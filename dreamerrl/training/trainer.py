from __future__ import annotations

import copy
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box
from torch import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from dreamerrl.debug.probe import DeterminismProbe
from dreamerrl.env.env_factory import make_env
from dreamerrl.logging.repo_logger import create_repro_logger
from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.replay_buffer.replay_buffer import ReplayBuffer
from dreamerrl.training.core import actor_critic_update, world_model_training_step
from dreamerrl.utils.seed import set_global_seeds
from dreamerrl.utils.types import DreamerConfig, LatentConfig, LRScheduleConfig, NetworkConfig, WorldModelMetrics

# Define LOG_DIR at module load so the path is computed once, from the actual file location, not from the caller's
# working directory. This guarantees a consistent top‑level logs/ directory under pytest, CLI runs, and
# multiprocessing.

# Path to trainer.py
trainer_file = Path(__file__).resolve()

# dreamerrl/training/trainer.py → dreamerrl/training → dreamerrl → project root
PROJECT_ROOT = trainer_file.parents[2]

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


class CosineWarmupScheduler:
    """Local shared LR schedule for world, actor, and critic optimizers."""

    def __init__(self, cfg: LRScheduleConfig):
        self.cfg = cfg

    def __call__(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return self.cfg.base_lr * (step / self.cfg.warmup_steps)

        progress = (step - self.cfg.warmup_steps) / max(1, self.cfg.total_steps - self.cfg.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        min_lr = self.cfg.base_lr * self.cfg.lr_floor
        return min_lr + (self.cfg.base_lr - min_lr) * cosine


class DreamerTrainer:
    """
    Dreamer‑V3 trainer.

    Wiring summary:
      • WorldModel: factored discrete latent RSSM + symlog reward/continue heads
      • Actor: policy over discrete actions from (h, z)
      • Critic: distributional value head over symlog bins
      • ReplayBuffer: stores raw env transitions, samples (B, L, ·) sequences
    """

    def __init__(self, cfg: DreamerConfig):
        for name, value in (
            ("seq_len", cfg.train.seq_len),
            ("collect_steps", cfg.train.collect_steps),
            ("max_episode_steps", cfg.env.max_episode_steps),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(cfg.ac.discount) or not 0 < cfg.ac.discount <= 1:
            raise ValueError("discount must be finite and in (0, 1]")
        if not math.isfinite(cfg.ac.lambda_) or not 0 <= cfg.ac.lambda_ <= 1:
            raise ValueError("lambda_ must be finite and in [0, 1]")
        if not math.isfinite(cfg.ac.replay_value_scale) or cfg.ac.replay_value_scale < 0:
            raise ValueError("replay_value_scale must be finite and nonnegative")
        self.ac_metrics: dict[str, torch.Tensor] = {}
        self.cfg = cfg
        self.sample_step = 0
        self.completed_episodes = 0
        self.run_state: dict[str, Any] = {}
        if not math.isfinite(cfg.train.action_cost) or cfg.train.action_cost < 0:
            raise ValueError("action_cost must be finite and nonnegative")

        self.on_episode: Callable[[dict[str, Any]], None] | None = None
        self.on_checkpoint: Callable[[dict[str, Any]], None] | None = None
        self.completed_updates = 0
        self.total_updates_target: int | None = None

        if cfg.train.enable_repro_log:
            self.repro_log = create_repro_logger(cfg.train.seed, cfg.env.seed, LOG_DIR)
        else:
            self.repro_log = None

        # Unified determinism probe
        self.probe = DeterminismProbe(
            logger=self.repro_log,
            every_n=self.cfg.train.repro_log_every_n,
        )

        if self.cfg.train.deterministic_env:
            self.cfg.train.use_amp = False
            self.cfg.train.enable_fused_adamw_kernels = False

            torch.use_deterministic_algorithms(True)

            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            os.environ["PYTORCH_DISABLE_FUSED_ADAMW"] = "1"
            os.environ["PYTORCH_CUDA_FUSER_DISABLE"] = "1"

        self.scaler = GradScaler() if cfg.train.use_amp else None

        logdir = os.path.join(cfg.log.tb_logdir, cfg.log.run_name)
        self.tb = SummaryWriter(log_dir=logdir)

        # -----------------------------------------------------
        # Seeding (must be first for reproducibility)
        # -----------------------------------------------------
        set_global_seeds(cfg.train.seed)

        # Request deterministic kernels for repeatability on the same stack.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda" if cfg.train.cuda and torch.cuda.is_available() else "cpu")

        # -----------------------------------------------------
        # Environment
        # -----------------------------------------------------
        self.env = make_env(cfg.env, device=self.device)
        obs_space = self.env.obs_space
        self.episode_steps = torch.zeros(self.env.batch_size, dtype=torch.long, device=self.device)
        if cfg.env.finite_horizon:
            if not isinstance(obs_space, Box) or len(obs_space.shape) != 1:
                self.env.venv.close()
                self.tb.close()
                raise ValueError("Finite-horizon training requires vector Box observations")
            obs_space = Box(
                np.append(obs_space.low, 0).astype(np.float32),
                np.append(obs_space.high, 1).astype(np.float32),
                dtype=np.float32,
            )
        self.action_dim = self.env.action_dim

        # -----------------------------------------------------
        # Latent + Network configs
        # -----------------------------------------------------
        latent = LatentConfig(
            deter_size=cfg.world.deter_size,
            stoch_size=cfg.world.stoch_size,
            num_classes=cfg.world.num_classes,
        )

        net_world = NetworkConfig(
            hidden_size=cfg.world.hidden_size,
            action_dim=self.action_dim,
            value_bins=cfg.world.value_bins,
            unimix=cfg.world.unimix,
        )

        # -----------------------------------------------------
        # World Model
        # -----------------------------------------------------
        self.world = WorldModel(
            obs_space=obs_space,
            dreamer_cfg=cfg,
            latent=latent,
            net=net_world,
            device=self.device,
            probe=self.probe,
        )
        self.world_state = self.world.init_state(self.env.batch_size)
        self.world_state = self.world_state.to(self.device)

        # -----------------------------------------------------
        # Actor + Critic
        # -----------------------------------------------------
        net_actor = NetworkConfig(
            hidden_size=cfg.ac.actor_hidden,
            unimix=cfg.ac.unimix,
            action_dim=self.action_dim,
        )

        net_critic = NetworkConfig(
            hidden_size=cfg.ac.critic_hidden,
            value_bins=cfg.world.value_bins,
        )

        self.actor = Actor(latent=latent, net=net_actor, probe=self.probe).to(self.device)
        self.critic = ValueHead(latent=latent, net=net_critic, probe=self.probe).to(self.device)
        self.slow_critic = copy.deepcopy(self.critic).requires_grad_(False).eval()
        self.slow_critic.probe = None
        if not 0 < cfg.ac.slow_critic_rate <= 1:
            raise ValueError("slow_critic_rate must be in (0, 1]")

        # -----------------------------------------------------
        # Replay Buffer
        # -----------------------------------------------------
        flat_obs_dim = self.world.flat_obs_dim
        self.replay = ReplayBuffer(cfg=cfg.train, obs_dim=flat_obs_dim, action_dim=self.action_dim, device=self.device)

        # -----------------------------------------------------
        # Optimizers
        # -----------------------------------------------------
        self.model_opt = torch.optim.Adam(
            self.world.parameters(), lr=cfg.train.model_lr, fused=self.cfg.train.enable_fused_adamw_kernels
        )
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.train.actor_lr, fused=self.cfg.train.enable_fused_adamw_kernels
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.train.critic_lr, fused=self.cfg.train.enable_fused_adamw_kernels
        )

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------
        self.wandb_run = None
        if cfg.log.enable_wandb:
            import wandb

            self.wandb_run = wandb.init(project="dreamer_v3", config=cfg.__dict__)

        self.env_state: Dict[str, Any] = self.prepare_env_output(self.env.reset(), reset=True)
        with torch.no_grad():
            self.world_state = self.world.observe_step(
                self.world_state,
                self.env_state["state"].to(self.device),
                torch.zeros(self.env.batch_size, self.action_dim, device=self.device),
                is_first=torch.ones(self.env.batch_size, dtype=torch.bool, device=self.device),
            )["post"].detach()
        self.total_env_steps: int = 0
        self.episode_returns = torch.zeros(self.env.batch_size, device=self.device)
        self.recent_returns = []

    @property
    def global_step(self) -> int:
        return self.total_env_steps

    def save_checkpoint(self, path: str | Path) -> None:
        """Atomically save a trusted local training checkpoint at an update boundary.

        Environment state and unfinished replay episodes are intentionally omitted.
        A resumed collector starts new episodes rather than joining unrelated states.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 4,
            "config": self.cfg,
            "world": self.world.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "slow_critic": self.slow_critic.state_dict(),
            "model_opt": self.model_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "completed_updates": self.completed_updates,
            "total_updates_target": self.total_updates_target,
            "total_env_steps": self.total_env_steps,
            "sample_step": self.sample_step,
            "completed_episodes": self.completed_episodes,
            "run_state": self.run_state,
            "recent_returns": self.recent_returns,
            "replay": self.replay.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            },
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
            ) as file:
                temporary = Path(file.name)
                torch.save(payload, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        if self.on_checkpoint is not None:
            self.on_checkpoint(
                dict(
                    event="checkpoint",
                    update=self.completed_updates,
                    env_steps=self.total_env_steps,
                    path=str(path),
                    bytes=path.stat().st_size,
                )
            )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a checkpoint produced by this project; only load files you trust."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._restore_checkpoint(payload)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        cuda: bool | None = None,
        log_dir: str | Path | None = None,
    ) -> DreamerTrainer:
        """Construct from the saved configuration, optionally selecting CPU/CUDA."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("version") in (1, 2):
            raise ValueError(
                "Pooled-latent checkpoint v1/v2 is incompatible with factor-preserving heads; start a new run"
            )
        if payload.get("version") == 3:
            raise ValueError("MLP-RSSM checkpoint v3 is incompatible with the GRU transition; start a new run")
        if payload.get("version") != 4:
            raise ValueError("Unsupported Dreamer checkpoint version")
        cfg = payload["config"]
        if log_dir is not None:
            cfg.log.tb_logdir = str(Path(log_dir) / "tensorboard")
            cfg.log.run_name = "train"
        if cuda is not None:
            cfg.train.cuda = cuda
        if not cfg.train.cuda or not torch.cuda.is_available():
            cfg.train.use_amp = False
            cfg.train.enable_fused_adamw_kernels = False
        trainer = cls(cfg)
        try:
            trainer._restore_checkpoint(payload)
        except Exception:
            trainer.env.venv.close()
            trainer.tb.close()
            raise
        return trainer

    def _restore_checkpoint(self, payload) -> None:
        if payload.get("version") in (1, 2):
            raise ValueError(
                "Pooled-latent checkpoint v1/v2 is incompatible with factor-preserving heads; start a new run"
            )
        if payload.get("version") == 3:
            raise ValueError("MLP-RSSM checkpoint v3 is incompatible with the GRU transition; start a new run")
        if payload.get("version") != 4:
            raise ValueError("Unsupported Dreamer checkpoint version")
        saved_cfg = payload["config"]
        if self.cfg.world != saved_cfg.world or self.cfg.ac != saved_cfg.ac or self.cfg.env != saved_cfg.env:
            raise ValueError("Checkpoint environment/model configuration does not match this trainer")
        # Validate replay dimensions before changing model/optimizer state.
        for key in ("seq_len", "obs_dim", "action_dim", "capacity"):
            if payload["replay"][key] != getattr(self.replay, key):
                raise ValueError(f"Checkpoint replay {key} does not match this trainer")
        for name in ("world", "actor", "critic", "model_opt", "actor_opt", "critic_opt"):
            getattr(self, name).load_state_dict(payload[name])
        self.slow_critic.load_state_dict(payload["slow_critic"])
        if self.device.type == "cpu":
            for optimizer in (self.model_opt, self.actor_opt, self.critic_opt):
                for group in optimizer.param_groups:
                    group["fused"] = False
        if self.scaler is not None and payload["scaler"] is not None:
            self.scaler.load_state_dict(payload["scaler"])
        self.replay.load_state_dict(payload["replay"])
        self.completed_updates = payload["completed_updates"]
        self.total_updates_target = payload["total_updates_target"]
        self.total_env_steps = payload["total_env_steps"]
        self.sample_step = payload["sample_step"]
        self.completed_episodes = payload.get("completed_episodes", len(self.replay.episodes))
        self.run_state = payload.get("run_state", {})
        self.recent_returns = list(payload["recent_returns"])
        self.episode_returns.zero_()
        # The simulator cannot be snapshotted. Reset both collector and RSSM.
        self.env_state = self.prepare_env_output(
            self.env.reset(seed=self.cfg.env.seed + self.total_env_steps),
            reset=True,
        )
        with torch.no_grad():
            self.world_state = self.world.observe_step(
                self.world.init_state(self.env.batch_size),
                self.env_state["state"].to(self.device),
                torch.zeros(self.env.batch_size, self.action_dim, device=self.device),
                is_first=torch.ones(self.env.batch_size, dtype=torch.bool, device=self.device),
            )["post"].detach()
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None and self.device.type == "cuda":
            for index, state in enumerate(rng["cuda"][: torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state.cpu(), index)

    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------
    def train(
        self,
        total_updates: int,
        *,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 100,
        on_update: Callable[[dict[str, Any]], bool | None] | None = None,
    ) -> None:
        """Train to an absolute update count, saving after completed updates."""
        if total_updates < self.completed_updates:
            raise ValueError("total_updates is below the checkpoint's completed update count")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        self.total_updates_target = total_updates
        lr_cfg = LRScheduleConfig(
            base_lr=self.cfg.train.model_lr,
            warmup_steps=self.cfg.train.warmup_steps,
            total_steps=total_updates,
            lr_floor=0.1,
        )
        lr_schedule = CosineWarmupScheduler(lr_cfg)

        for update_idx in range(self.completed_updates, total_updates):
            t0 = time.perf_counter()
            previous_env_steps = self.total_env_steps

            lr = lr_schedule(update_idx)
            for pg in self.model_opt.param_groups:
                pg["lr"] = lr
            for pg in self.actor_opt.param_groups:
                pg["lr"] = lr
            for pg in self.critic_opt.param_groups:
                pg["lr"] = lr

            self.collect_env_steps()
            if not self.replay.episodes:
                warmup_calls = math.ceil(self.cfg.env.max_episode_steps / self.cfg.train.collect_steps)
                for _ in range(warmup_calls - 1):
                    self.collect_env_steps()
                    if self.replay.episodes:
                        break
                if not self.replay.episodes:
                    raise RuntimeError(
                        "No completed replay episodes within max_episode_steps; "
                        "check environment endings and replay capacity"
                    )

            batch = self.replay.sample(
                batch_size=self.cfg.train.batch_size,
                seed=self.cfg.train.seed + self.sample_step,
            )
            self.sample_step += 1

            wm_metrics = self.update_world_model(batch, update_idx)

            if on_update is None and update_idx % self.cfg.train.console_log_interval == 0:
                print(
                    f"[train] "
                    f"update={update_idx}/{total_updates} "
                    f"episodes={len(self.replay.episodes)} "
                    f"replay_size={self.replay.size} "
                    f"env_steps={self.total_env_steps}"
                )

                print(
                    f"[wm] "
                    f"total={wm_metrics.total_loss.item():.4f} "
                    f"recon={wm_metrics.recon_loss.item():.4f} "
                    f"reward={wm_metrics.reward_loss.item():.4f} "
                    f"cont={wm_metrics.cont_loss.item():.4f} "
                    f"kl_dyn={wm_metrics.kl_dyn.item():.4f} "
                    f"kl_rep={wm_metrics.kl_rep.item():.4f}"
                )

                if wm_metrics.aux_losses:
                    aux_names = [obj.name for obj in self.world.aux_objectives]

                    print(
                        "[aux] "
                        + " ".join(f"{name}={loss.item():.4f}" for name, loss in zip(aux_names, wm_metrics.aux_losses))
                    )

            actor_loss, critic_loss = self.update_actor_critic(batch, update_idx)

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "loss/model": wm_metrics.total_loss.item(),
                        "loss/actor": actor_loss,
                        "loss/critic": critic_loss,
                        "time/update": time.perf_counter() - t0,
                    },
                    step=update_idx,
                )

            self.completed_updates = update_idx + 1
            elapsed = time.perf_counter() - t0
            record = dict(
                event="train",
                update=self.completed_updates,
                target_updates=total_updates,
                env_steps=self.total_env_steps,
                episodes=self.completed_episodes,
                replay_size=self.replay.size,
                replay_episodes=len(self.replay.episodes),
                replay_positive_reward_steps=self.replay.positive_reward_steps,
                replay_positive_reward_episodes=self.replay.positive_reward_episodes,
                credit_assignment={name: value.item() for name, value in self.ac_metrics.items()},
                reward_coverage={name: value.item() for name, value in wm_metrics.reward_metrics.items()},
                sequence_length=batch["obs"].shape[1],
                learning_rate=lr,
                model_tokens_per_env_step=batch["obs"].shape[0]
                * batch["obs"].shape[1]
                / max(1, self.total_env_steps - previous_env_steps),
                update_seconds=elapsed,
                env_steps_per_second=(self.total_env_steps - previous_env_steps) / max(elapsed, 1e-9),
                recent_return_mean=float(np.mean(self.recent_returns)) if self.recent_returns else None,
                world_loss=wm_metrics.total_loss.item(),
                recon_loss=wm_metrics.recon_loss.item(),
                reward_loss=wm_metrics.reward_loss.item(),
                continuation_loss=wm_metrics.cont_loss.item(),
                kl_dyn=wm_metrics.kl_dyn.item(),
                kl_rep=wm_metrics.kl_rep.item(),
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                aux_losses={
                    cfg.name: loss.item() for cfg, loss in zip(self.world.aux_objectives, wm_metrics.aux_losses)
                },
                cuda_memory_mb=torch.cuda.memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else None,
            )
            keep_running = on_update(record) if on_update is not None else None
            if checkpoint_path is not None and (
                self.completed_updates % checkpoint_every == 0
                or self.completed_updates == total_updates
                or keep_running is False
            ):
                self.save_checkpoint(checkpoint_path)
            if keep_running is False:
                break

    # -------------------------------------------------------------
    # Collect steps from environment
    # -------------------------------------------------------------
    @torch.no_grad()
    def collect_env_steps(self) -> None:
        """
        Collect multiple environment steps per call.
        Dreamer-V3 requires enough steps to finish episodes so the replay buffer
        can finalize and store them.
        """
        for _ in range(self.cfg.train.collect_steps):
            # 1. Choose discrete action
            if self.cfg.train.deterministic_env:
                # Fully deterministic for reproducibility tests
                actions_discrete = self.actor(self.world_state.h, self.world_state.z).argmax(dim=-1)
            else:
                # Normal Dreamer training behavior
                if self.global_step < self.cfg.train.random_exploration_steps:
                    actions_discrete = torch.randint(
                        low=0,
                        high=self.action_dim,
                        size=(self.env.batch_size,),
                        device=self.device,
                    )
                else:
                    actions_discrete, _ = self.actor.act(self.world_state)

            # 2. Step environment
            env_out = self.prepare_env_output(self.env.step(actions_discrete))
            self._check_consistency(env_out)

            # Move env outputs to CUDA
            for k, v in env_out.items():
                if torch.is_tensor(v):
                    env_out[k] = v.to(self.device)

            self.episode_returns += env_out["reward"]
            for i in env_out["is_last"].nonzero(as_tuple=True)[0].tolist():
                ep_return = self.episode_returns[i].item()
                self.completed_episodes += 1
                if self.on_episode is not None:
                    self.on_episode(
                        dict(
                            event="episode",
                            env_index=i,
                            episode=self.completed_episodes,
                            env_steps=self.total_env_steps + self.env.batch_size,
                            episode_return=ep_return,
                        )
                    )
                self.recent_returns.append(ep_return)
                self.recent_returns = self.recent_returns[-50:]
                self.tb.add_scalar("env/ep_return", ep_return, self.total_env_steps + self.env.batch_size)
                self.episode_returns[i] = 0
            if env_out["is_last"].any():
                self.tb.add_scalar(
                    "env/avg_return_50", np.mean(self.recent_returns), self.total_env_steps + self.env.batch_size
                )

            training_reward = env_out["reward"] - self.cfg.train.action_cost

            # 3. One-hot encode actions for RSSMCore
            actions_one_hot = F.one_hot(actions_discrete, num_classes=self.action_dim).float()

            # 4. Update latent state
            wm_out = self.world.observe_step(
                prev_state=self.world_state,
                obs=env_out.get("transition_state", env_out["state"]),
                action=actions_one_hot,
                reward=training_reward,
                is_first=None if "transition_state" in env_out else env_out["is_first"],
                is_last=env_out["is_last"],
                is_terminal=env_out["is_terminal"],
            )

            self.world_state = wm_out["post"].detach()
            if "transition_state" in env_out and env_out["is_last"].any():
                reset = self.world.observe_step(
                    self.world_state,
                    env_out["state"],
                    torch.zeros_like(actions_one_hot),
                    is_first=env_out["is_last"],
                )["post"]
                mask = env_out["is_last"].unsqueeze(-1)
                self.world_state.h = torch.where(mask, reset.h, self.world_state.h)
                self.world_state.z = torch.where(mask.unsqueeze(-1), reset.z, self.world_state.z)

            # 5. Store transition with the shaped training reward in replay.
            self.replay.add(
                obs=env_out.get("transition_state", env_out["state"]),
                initial_obs=self.env_state["state"],
                action=actions_discrete,
                reward=training_reward,
                done=env_out["is_last"].float().to(self.device),
                is_terminal=env_out["is_terminal"].float().to(self.device),
            )

            # 6. Update counters
            self.env_state = env_out
            self.total_env_steps += self.env.batch_size

    def prepare_env_output(self, out: Dict[str, Any], *, reset: bool = False) -> Dict[str, Any]:
        """Apply the chosen finite-task objective and its observable clock.

        Collection chunk boundaries do not end a task. Same-step autoreset
        adapters must preserve the final transition observation separately.
        """
        if not self.cfg.env.finite_horizon:
            return out
        out = dict(out)
        horizon = self.cfg.env.max_episode_steps
        if reset:
            self.episode_steps.zero_()
        else:
            if "transition_state" not in out:
                raise ValueError("Finite-horizon adapters must preserve transition_state across autoreset")
            self.episode_steps.add_(1)
            remaining = (horizon - self.episode_steps).clamp_min(0).float() / horizon
            out["transition_state"] = torch.cat(
                [out["transition_state"].to(self.device), remaining.unsqueeze(-1)],
                -1,
            )
            out["is_terminal"] = out["is_terminal"] | out["is_last"]
            self.episode_steps[out["is_last"].to(self.device)] = 0
        remaining = (horizon - self.episode_steps).clamp_min(0).float() / horizon
        out["state"] = torch.cat([out["state"].to(self.device), remaining.unsqueeze(-1)], -1)
        return out

    def _check_consistency(self, env_out: dict) -> None:
        state = env_out["state"]
        reward = env_out["reward"]
        is_last = env_out["is_last"]
        is_terminal = env_out["is_terminal"]
        prev_action = env_out.get("prev_action", None)

        # ✅ unified environment interface
        batch_size = self.env.batch_size
        obs_dim = self.world.flat_obs_dim
        action_dim = self.env.action_dim

        if state.shape != (batch_size, obs_dim):
            raise RuntimeError(f"State shape mismatch: got {state.shape}, expected ({batch_size}, {obs_dim})")

        if prev_action is not None and prev_action.shape != (batch_size, action_dim):
            raise RuntimeError(
                f"Prev_action shape mismatch: got {prev_action.shape}, expected ({batch_size}, {action_dim})"
            )

        if reward.shape != (batch_size,):
            raise RuntimeError(f"Reward shape mismatch: got {reward.shape}, expected ({batch_size},)")

        if not torch.isfinite(reward).all():
            raise RuntimeError("Non-finite reward detected")

        # ✅ use is_first so Ruff stops complaining
        for key in ("is_first", "is_last", "is_terminal"):
            flag = env_out[key]
            if flag.shape != (batch_size,):
                raise RuntimeError(f"{key} shape mismatch: got {flag.shape}, expected ({batch_size},)")
            if flag.dtype != torch.bool:
                raise RuntimeError(f"{key} must be boolean, got {flag.dtype}")

        if (is_terminal & ~is_last).any():
            raise RuntimeError("True terminal transitions must also end the episode")

    def update_world_model(self, batch: Dict[str, torch.Tensor], update_idx: int) -> WorldModelMetrics:
        # Zero optimizer gradients
        self.model_opt.zero_grad()

        # AMP (Automatic Mixed Precision):
        #   Uses FP16 for most matrix multiplications and FP32 for numerically sensitive ops. This speeds up training
        #   on modern GPUs and reduces memory use. AMP is nondeterministic because FP16 kernels vary across hardware
        #   and drivers. Therefore AMP is disabled automatically when deterministic_env=True.
        if self.cfg.train.use_amp:
            with autocast(device_type="cuda", dtype=torch.float16, enabled=self.cfg.train.use_amp):
                metrics = world_model_training_step(
                    world_model=self.world,
                    batch=batch,
                    kl_scale=self.cfg.world.kl_cfg.scale,
                )
        else:
            metrics = world_model_training_step(
                world_model=self.world,
                batch=batch,
                kl_scale=self.cfg.world.kl_cfg.scale,
            )

        if self.repro_log:
            self.repro_log.debug(
                f"[TRAIN] KL_DYN={metrics.kl_dyn.mean().item():.6f} KL_REP={metrics.kl_rep.mean().item():.6f}"
            )

        loss = metrics.total_loss

        if self.cfg.train.use_amp:
            assert self.scaler is not None, "GradScaler is None but use_amp is True"
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.model_opt)
            torch.nn.utils.clip_grad_norm_(self.world.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.model_opt)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world.parameters(), self.cfg.train.grad_clip)
            self.model_opt.step()

        # TensorBoard logging
        step = update_idx

        self.tb.add_scalar("wm/total_loss", metrics.total_loss.item(), step)
        self.tb.add_scalar("wm/recon_loss", metrics.recon_loss.item(), step)
        self.tb.add_scalar("wm/reward_loss", metrics.reward_loss.item(), step)
        self.tb.add_scalar("wm/cont_loss", metrics.cont_loss.item(), step)
        self.tb.add_scalar("wm/kl_dyn", metrics.kl_dyn.item(), step)
        self.tb.add_scalar("wm/kl_rep", metrics.kl_rep.item(), step)

        for name, value in metrics.reward_metrics.items():
            self.tb.add_scalar(f"wm/{name}", value.item(), step)

        for name, value in metrics.reconstruction_metrics.items():
            self.tb.add_scalar(f"wm/{name}", value.item(), step)

        for i, aux in enumerate(metrics.aux_losses):
            self.tb.add_scalar(f"wm/aux_loss_{i}", aux.item(), step)

        return metrics

    # -------------------------------------------------------------
    # Actor + Critic Update
    # -------------------------------------------------------------
    def update_actor_critic(self, batch: Dict[str, torch.Tensor], update_idx: int):
        self.ac_metrics = {}
        # Freeze actor/critic during reproducibility tests to prevent cross‑seed divergence.
        if update_idx < self.cfg.train.freeze_actor_critic_steps:
            return 0.0, 0.0

        remaining_steps = None
        if self.cfg.env.finite_horizon:
            obs = batch.get("obs", batch.get("state"))
            if obs is None:
                raise KeyError("Batch must contain obs or state")
            remaining_steps = (obs[..., -1] * self.cfg.env.max_episode_steps).round().long()

        if self.cfg.train.use_amp:
            with autocast(device_type="cuda", dtype=torch.float16, enabled=self.cfg.train.use_amp):
                actor_loss, critic_loss = actor_critic_update(
                    world_model=self.world,
                    actor=self.actor,
                    critic=self.critic,
                    batch=batch,
                    imagination_horizon=self.cfg.world.imagination_horizon,
                    discount=self.cfg.ac.discount,
                    lam=self.cfg.ac.lambda_,
                    deterministic_imagination=self.cfg.train.deterministic_imagination,
                    slow_critic=self.slow_critic,
                    slow_critic_scale=self.cfg.ac.slow_critic_scale,
                    entropy_scale=self.cfg.ac.entropy_scale,
                    remaining_steps=remaining_steps,
                    replay_value_scale=self.cfg.ac.replay_value_scale,
                    diagnostics=self.ac_metrics,
                )
        else:
            actor_loss, critic_loss = actor_critic_update(
                world_model=self.world,
                actor=self.actor,
                critic=self.critic,
                batch=batch,
                imagination_horizon=self.cfg.world.imagination_horizon,
                discount=self.cfg.ac.discount,
                lam=self.cfg.ac.lambda_,
                deterministic_imagination=self.cfg.train.deterministic_imagination,
                slow_critic=self.slow_critic,
                slow_critic_scale=self.cfg.ac.slow_critic_scale,
                entropy_scale=self.cfg.ac.entropy_scale,
                remaining_steps=remaining_steps,
                replay_value_scale=self.cfg.ac.replay_value_scale,
                diagnostics=self.ac_metrics,
            )

        # AMP (Automatic Mixed Precision):
        #   Uses FP16 for most matrix multiplications and FP32 for numerically sensitive ops. This speeds up
        #   training on modern GPUs and reduces memory use. AMP is nondeterministic because FP16 kernels vary
        #   across hardware and drivers. Therefore AMP is disabled automatically when deterministic_env=True.
        if self.cfg.train.use_amp:
            assert self.scaler is not None, "GradScaler is None but use_amp is True"

            self.actor_opt.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_opt)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.actor_opt)
            self.scaler.update()

            self.critic_opt.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_opt)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.critic_opt)
            self.scaler.update()
        else:
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.train.grad_clip)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.train.grad_clip)
            self.critic_opt.step()

        for name, value in self.ac_metrics.items():
            self.tb.add_scalar(f"ac/{name}", value.item(), update_idx)
        self.update_slow_critic()
        return float(actor_loss.item()), float(critic_loss.item())

    @torch.no_grad()
    def update_slow_critic(self) -> None:
        for slow, online in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow.lerp_(online, self.cfg.ac.slow_critic_rate)
        for slow, online in zip(self.slow_critic.buffers(), self.critic.buffers()):
            slow.copy_(online)
