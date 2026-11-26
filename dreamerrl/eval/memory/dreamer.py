"""Read-only flat Dreamer checkpoint adapter using the production filter."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import torch
import torch.nn.functional as F

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel, WorldModelState
from dreamerrl.utils.types import LatentConfig, NetworkConfig


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def module_sha256(module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointSpec:
    path: Path
    sha256: str
    seed: int
    update: int
    env_steps: int
    version: int = 4


def verified_payload(spec: CheckpointSpec):
    """Validate provenance before deserializing a trusted, pinned local file."""
    if file_sha256(spec.path) != spec.sha256:
        raise ValueError(f"Checkpoint digest mismatch for {spec.path}")
    payload = torch.load(spec.path, map_location="cpu", mmap=True, weights_only=False)
    cfg = payload["config"]
    expected = (spec.version, spec.seed, spec.seed, spec.update, spec.env_steps)
    found = (payload["version"], cfg.train.seed, cfg.env.seed, payload["completed_updates"], payload["total_env_steps"])
    if found != expected:
        raise ValueError(f"Checkpoint identity mismatch: {found} != {expected}")
    return payload


def categorical_state(probs, mode, *, rng=None, noise=None):
    if mode == "argmax":
        if noise is not None:
            raise ValueError("Categorical noise is only meaningful for sampling")
        index = probs.argmax(-1)
    elif mode == "sample":
        if noise is None:
            if rng is None:
                raise ValueError("Sampling requires explicit RNG or paired noise")
            noise = gumbel_noise(probs.shape, rng, probs.device)
        if noise.shape != probs.shape:
            raise ValueError("Categorical noise and probabilities must have equal shapes")
        index = (probs.log() + noise).argmax(-1)
    else:
        raise ValueError(f"Unknown latent mode: {mode}")
    return F.one_hot(index, probs.shape[-1]).to(probs.dtype)


def gumbel_noise(shape, rng, device="cpu"):
    uniform = torch.rand(shape, generator=rng, device=device).clamp_min(torch.finfo(torch.float32).tiny)
    return -(-uniform.log()).log()


class DreamerAdapter:
    """Flat actor and online critic; the world state is the recurrent carry."""

    def __init__(self, cfg, world, actor, critic):
        self.config, self.world, self.actor, self.critic = cfg, world, actor, critic
        self.action_dim = world.net_cfg.action_dim
        for module in (world, actor, critic):
            if module is not None:
                module.eval().requires_grad_(False)
        self.world.posterior.deterministic_latent_for_tests = True
        self.world.prior.deterministic_latent_for_tests = True

    @classmethod
    def from_payload(cls, payload, obs_space, action_dim, *, load_critic=True):
        cfg = payload["config"]
        if payload["version"] != 4 or cfg.ac.hierarchical:
            raise ValueError("This diagnostic adapter supports flat v4 checkpoints")
        latent = LatentConfig(deter_size=cfg.world.deter_size, num_classes=cfg.world.num_classes,
                              stoch_size=cfg.world.stoch_size)
        net = NetworkConfig(hidden_size=cfg.world.hidden_size, action_dim=action_dim,
                            value_bins=cfg.world.value_bins, unimix=cfg.world.unimix)
        world = WorldModel(obs_space=obs_space, dreamer_cfg=cfg, latent=latent, net=net)
        actor = Actor(latent=latent, net=NetworkConfig(hidden_size=cfg.ac.actor_hidden,
                      action_dim=action_dim, unimix=cfg.ac.unimix))
        critic = ValueHead(latent=latent, net=NetworkConfig(hidden_size=cfg.ac.critic_hidden,
                           value_bins=cfg.world.value_bins)) if load_critic else None
        modules = {"world": world, "actor": actor, "critic": critic}
        for name, module in modules.items():
            if module is not None:
                module.load_state_dict(payload[name], strict=True)
        return cls(cfg, world, actor, critic)

    def initial_state(self, batch):
        return self.world.init_state(batch)

    @torch.inference_mode()
    def filter_step(self, state, obs, incoming, first, *, mode="argmax", rng=None, noise=None):
        world = self.world
        embedding = world.encoder(obs)
        # Capture the raw head output in the same production filter call. The
        # public logits field can instead hold the log of mixed probabilities.
        captured = []
        hook = world.posterior.fc_logits.register_forward_hook(lambda _m, _x, y: captured.append(y))
        try:
            state = world.filter_step(state, embedding, incoming, first)
        finally:
            hook.remove()
        q = state.post_stats["probs"]
        raw = captured[0].reshape_as(q)
        z = categorical_state(q, mode, rng=rng, noise=noise)
        state = self.with_z(state, z)
        p = world.prior(state.h)["probs"]
        return state, dict(observation=obs, encoder=embedding, raw_logits=raw, pre_unimix=raw.float().softmax(-1),
                           posterior=q, prior=p, z=z, h=state.h, hz=torch.cat((state.h, z.flatten(1)), dim=-1))

    @staticmethod
    def with_z(state, z):
        if z.shape != state.z.shape:
            raise ValueError("Replacement z must match recurrent carry")
        stats = None if state.post_stats is None else {**state.post_stats, "z": z}
        return WorldModelState(h=state.h, z=z, prior_stats=state.prior_stats, post_stats=stats)

    @torch.inference_mode()
    def imagine_step(self, state, action, *, mode="argmax", rng=None):
        successor = self.world.imagine_step(state, self.actor, action=action, deterministic_imagination=True)
        return self.with_z(successor, categorical_state(successor.prior_stats["probs"], mode, rng=rng))

    @torch.inference_mode()
    def readouts(self, state):
        logits = self.actor(state.h, state.z)
        result = {"action_logits": logits, "action_probability": logits.softmax(-1)}
        if self.critic is not None:
            result["value"] = self.critic.readout(self.critic(state.h, state.z))
        return result

    @torch.inference_mode()
    def reward(self, state):
        head = self.world.reward_heads.main
        return head.readout(head(state.h, state.z))
