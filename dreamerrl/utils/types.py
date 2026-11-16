from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import torch

from .transforms import symexp


@dataclass(frozen=True)
class AuxHeadConfig:
    """
    Shared config for auxiliary heads.

    in_dim:  dimension of concatenated (h_t, z_t)
    hidden:  hidden layer size
    out_dim: output dimension (1 for scalar, N for vector)
    """

    in_dim: int
    hidden: int
    out_dim: int


# ---------------------------------------------------------
# Auxiliary Objective Config
# ---------------------------------------------------------
@dataclass(frozen=True)
class AuxObjectiveConfig:
    """
    Environment‑agnostic auxiliary objective.

    Each objective provides a function:
        fn(batch, gamma) -> target tensor of shape (B, L)

    Examples:
        • affordance prediction
        • reachability prediction
        • novelty / event prediction
        • resource dynamics
        • skill discovery / options
    """

    name: str
    fn: Callable[[Dict[str, torch.Tensor], float], torch.Tensor]


# ---------------------------------------------------------
# KL Config (used in invariants + world model update)
# ---------------------------------------------------------
@dataclass(frozen=True)
class KLConfig:
    scale: float = 1.0
    free_nats: float = 1.0
    dynamics_scale: float = 1.0
    representation_scale: float = 0.1
    max_kl: float = 100.0
    min_kl: float = -1e-6
    require_nonzero: bool = True


# ---------------------------------------------------------
# World Model Config (RSSM + Encoder/Decoder)
# ---------------------------------------------------------
@dataclass
class WorldModelConfig:
    """
    Configuration for the Dreamer‑V3 world model.

    Maps directly to:
      • RSSMCore (deter_size, stoch_size, num_classes)
      • Prior / Posterior categorical distributions
      • ObsEncoder / Decoder MLP widths
      • Distributional reward/value heads (value_bins)
    """

    # Hidden sizes for RSSMCore, Prior, Posterior, Decoder.
    deter_size: int = 200
    stoch_size: int = 30
    num_classes: int = 32
    hidden_size: int = 200

    # Optional categorical MiniHack grid model; None preserves the vector MLP.
    glyph_shape: tuple[int, int] | None = None
    glyph_vocab_size: int = 0
    glyph_changed_weight: float = 1.0
    glyph_foreground_ids: tuple[int, ...] = ()
    glyph_loss_balance: str = "weighted"  # Legacy-compatible; new runner jobs default to groups.
    glyph_salient_fraction: float = 0.5
    reconstruction_scale: float = 1.0

    # Reward head hidden size.
    reward_hidden: int = 256

    kl_cfg: KLConfig = field(
        default_factory=lambda: KLConfig(
            max_kl=100.0,
            min_kl=-1e-6,
            require_nonzero=True,
        )
    )

    # Encoder/decoder MLP widths.
    encoder_hidden: int = 256
    decoder_hidden: int = 256

    # Imagination horizon for actor/critic rollout.
    imagination_horizon: int = 15

    unimix: float = 0.01

    # Distributional value/reward bins.
    value_bins: int = 41

    # Optional project extensions; baseline DreamerV3 does not require these.
    # Each callback produces a detached (B, L, D) target from replay.
    aux_objectives: List[AuxObjectiveConfig] = field(default_factory=list)
    aux_reward_scale: float = 0.1
    disable_aux_losses: bool = False


# ---------------------------------------------------------
# Actor/Critic Config
# ---------------------------------------------------------
@dataclass
class ActorCriticConfig:
    """
    Configuration for Dreamer‑V3 actor and critic networks.

    Maps directly to:
      • Actor MLP (actor_hidden)
      • Critic MLP (critic_hidden)
      • λ‑return computation (lambda_)
      • Discount factor for value targets
    """

    actor_hidden: int = 256
    critic_hidden: int = 256

    discount: float = 0.99
    lambda_: float = 0.95  # λ‑return smoothing
    unimix: float = 0.01
    entropy_scale: float = 3e-4
    slow_critic_rate: float = 0.02
    slow_critic_scale: float = 1.0
    replay_value_scale: float = 0.0  # Optional recorded-reward critic supervision.


# ---------------------------------------------------------
# Training Config
# ---------------------------------------------------------
@dataclass
class TrainingConfig:
    """
    Training hyperparameters for Dreamer‑V3.

    Maps directly to:
      • ReplayBuffer (replay_capacity, seq_len)
      • Trainer.update() batch sizes
      • LR warmup + cosine decay
      • Random exploration schedule
      • Gradient clipping
    """

    # Replay buffer + sequence sampling.
    replay_capacity: int = 200_000
    seq_len: int = 16
    burn_in: int = 16  # Preceding replay steps used only to reconstruct recurrent state.
    acting_context: int = -1  # Refresh after model updates: full live episode; 0 disables, positive caps history.
    batch_size: int = 64
    collect_steps: int = 100

    # Optional per-decision reward shaping; environment-return metrics stay raw.
    action_cost: float = 0.0

    # Learning rates for world model, actor, critic.
    model_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Gradient clipping.
    grad_clip: float = 100.0

    # Warmup for LR schedule.
    warmup_steps: int = 2_000

    # Random exploration before actor is used.
    random_exploration_steps: int = 2_500

    # Whether to use deterministic imagination (argmax) or sample from the prior.
    deterministic_imagination: bool = False

    # Enable / disable Weights & Biases logging.
    enable_wandb: bool = False

    # Deterministic env stepping (for reproducibility tests only).
    #
    # If True:
    #   • disables random exploration
    #   • disables stochastic policy sampling
    #   • uses argmax actions instead of sampled actions
    #   • ensures identical env trajectories across seeds
    #
    # Normal training behavior is unchanged.
    deterministic_env: bool = False

    enable_fused_adamw_kernels: bool = True  # Use fused AdamW kernels if available (faster training on GPU)
    use_amp: bool = True  # Use automatic mixed precision (AMP) for faster training on GPU

    # Device + seed.
    cuda: bool = True
    seed: int = 0

    # Legacy compatibility flag; positive lengths are always checked. Replay
    # windows may span collection calls and shorten to fit selected episodes.
    enforce_length_invariants: bool = False
    enable_repro_log: bool = False  # Set to true to enable reproducibility logging (repro.log)
    repro_log_every_n: int = 100

    freeze_actor_critic_steps: int = 2000
    console_log_interval: int = 10


# ---------------------------------------------------------
# Environment Config
# ---------------------------------------------------------
@dataclass
class EnvironmentConfig:
    """
    Environment configuration.

    Maps directly to:
      • PopGymVecWrapper
      • Trainer rollout loop
      • Curriculum scheduler
    """

    env_id: str = "popgym-RepeatPreviousEasy-v0"
    num_envs: int = 4
    max_episode_steps: int = 50
    deterministic: bool = False
    parallel: bool = False  # reproducibility mode
    seed: int = 0
    # True: episode ends define the task return, not a continuing-task cutoff.
    # The trainer appends a remaining-time feature and bounds imagined returns.
    finite_horizon: bool = False


# ---------------------------------------------------------
# Logging Config
# ---------------------------------------------------------
@dataclass
class LoggingConfig:
    """
    Logging + checkpoint configuration.

    Maps directly to:
      • TensorBoard logger
      • Checkpoint saving in Trainer
    """

    tb_logdir: str = "./tb_logs"
    checkpoint_dir: str = "./checkpoints"
    run_name: str = ""
    enable_wandb: bool = False


# ---------------------------------------------------------
# Debug Config (non-essential features for debugging)
# ---------------------------------------------------------
@dataclass
class DebugConfig:
    rich_dashboard: bool = False
    rich_update_interval: int = 10


# ---------------------------------------------------------
# Top-level Dreamer Config
# ---------------------------------------------------------
@dataclass
class DreamerConfig:
    """
    Top‑level Dreamer‑V3 configuration.

    This object is passed into:
      • Trainer
      • ReplayBuffer
      • WorldModel / Actor / Critic constructors
      • Environment wrappers
    """

    world: WorldModelConfig = field(default_factory=WorldModelConfig)
    ac: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    log: LoggingConfig = field(default_factory=LoggingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def init_run_name(self):
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log.run_name = f"{self.env.env_id}__dreamer__{ts}"


def make_default_config() -> DreamerConfig:
    cfg = DreamerConfig()
    cfg.init_run_name()
    return cfg


# ---------------------------------------------------------
# LatentConfig (shared latent geometry)
# ---------------------------------------------------------
@dataclass(frozen=True)
class LatentConfig:
    """
    Shared Dreamer‑V3 latent configuration.

    Maps directly to:
      • RSSMCore latent shapes
      • Prior/Posterior categorical distributions
      • Actor/Critic input shapes (h_t, z_t)
    """

    deter_size: int
    stoch_size: int
    num_classes: int

    @property
    def z_dim(self) -> int:
        return self.stoch_size * self.num_classes


# ---------------------------------------------------------
# NetworkConfig (shared MLP geometry)
# ---------------------------------------------------------
@dataclass(frozen=True)
class NetworkConfig:
    """
    Shared network configuration for Actor/Critic/Heads.

    Maps directly to:
      • Actor hidden size
      • Critic hidden size
      • Distributional reward/value bin supports
    """

    hidden_size: int
    value_bins: int | None = None
    bin_min: float = -10.0
    bin_max: float = 10.0
    discount: float = 0.99

    # Action dimension is required for Actor, optional for Critic.
    action_dim: int | None = None
    unimix: float = 0.01

    def make_bins(self, device=None):
        symlog_bins = torch.linspace(self.bin_min, self.bin_max, steps=self.value_bins)
        bins = symexp(symlog_bins)
        return bins if device is None else bins.to(device)


# ---------------------------------------------------------
# LR Schedule Config (shared LR schedule)
# ---------------------------------------------------------
@dataclass(frozen=True)
class LRScheduleConfig:
    """
    Linear warmup + cosine decay schedule.

    Maps directly to:
      • Trainer LR scheduler
      • World model / actor / critic optimizers
    """

    base_lr: float
    warmup_steps: int
    total_steps: int
    lr_floor: float = 0.1


# ---------------------------------------------------------
# World Model Metrics (returned by world model update)
# ---------------------------------------------------------
@dataclass
class WorldModelMetrics:
    total_loss: torch.Tensor
    recon_loss: torch.Tensor
    reward_loss: torch.Tensor
    cont_loss: torch.Tensor
    raw_kl_dyn: torch.Tensor
    raw_kl_rep: torch.Tensor
    kl_dyn: torch.Tensor
    kl_rep: torch.Tensor
    aux_losses: List
    reconstruction_metrics: Dict[str, torch.Tensor] = field(default_factory=dict)
    reward_metrics: Dict[str, torch.Tensor] = field(default_factory=dict)
