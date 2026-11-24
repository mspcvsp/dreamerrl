"""Fresh paired initialization and fail-closed provenance gate for the encoder ablation."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from copy import deepcopy
from dataclasses import is_dataclass, replace
from pathlib import Path

import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.models.world_model import WorldModel
from dreamerrl.utils.seed import set_global_seeds
from dreamerrl.utils.types import LatentConfig, NetworkConfig, make_default_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "environments/cage2/host_aware_v1_config.json"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(*arguments, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def load_config(arm, seed, run):
    spec = json.loads(CONFIG.read_text())
    if arm not in spec["arms"] or seed not in spec["training_seeds"]:
        raise ValueError("Arm and seed must belong to the preregistered experiment")
    cfg = make_default_config()

    def assign(target, values):
        updates = {}
        for key, value in values.items():
            current = getattr(target, key)
            if is_dataclass(current):
                updates[key] = assign(current, value)
            else:
                updates[key] = tuple(value) if isinstance(current, tuple) else value
        return replace(target, **updates)

    cfg = assign(cfg, spec["dreamer"])
    cfg.train.seed = cfg.env.seed = seed
    cfg.world.obs_encoder = arm
    if arm == "cage2_host":
        cfg.world.cage2_host_order = tuple(spec["schema"]["hosts"])
        cfg.world.cage2_subnet_ids = tuple(spec["schema"]["subnet_ids"])
    cfg.log.tb_logdir, cfg.log.run_name = str(run / "tensorboard"), "train"
    return cfg


def state_hash_without_encoder(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        if not name.startswith("encoder."):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def initialize_paired(trainer):
    """Use only freshly generated compatible weights; never load a baseline checkpoint."""
    cfg = deepcopy(trainer.cfg)
    cfg.world.obs_encoder, cfg.world.cage2_host_order, cfg.world.cage2_subnet_ids = "flat", (), ()
    latent = LatentConfig(
        deter_size=cfg.world.deter_size, num_classes=cfg.world.num_classes, stoch_size=cfg.world.stoch_size
    )
    net = NetworkConfig(
        hidden_size=cfg.world.hidden_size,
        action_dim=trainer.action_dim,
        value_bins=cfg.world.value_bins,
        unimix=cfg.world.unimix,
    )
    devices = [trainer.device.index or 0] if trainer.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.train.seed)
        reference = WorldModel(
            obs_space=trainer.env.obs_space, dreamer_cfg=cfg, latent=latent, net=net, device=trainer.device
        )
        actor = Actor(
            latent=latent,
            net=NetworkConfig(hidden_size=cfg.ac.actor_hidden, action_dim=trainer.action_dim, unimix=cfg.ac.unimix),
        ).to(trainer.device)
        critic = ValueHead(
            latent=latent, net=NetworkConfig(hidden_size=cfg.ac.critic_hidden, value_bins=cfg.world.value_bins)
        ).to(trainer.device)
        weights = reference.state_dict()
        if trainer.cfg.world.obs_encoder == "cage2_host":
            weights = {name: value for name, value in weights.items() if not name.startswith("encoder.")}
        result = trainer.world.load_state_dict(weights, strict=False)
        assert not result.unexpected_keys and all(name.startswith("encoder.") for name in result.missing_keys)
        trainer.actor.load_state_dict(actor.state_dict())
        trainer.critic.load_state_dict(critic.state_dict())
        trainer.slow_critic.load_state_dict(critic.state_dict())
    assert all(not opt.state for opt in (trainer.model_opt, trainer.actor_opt, trainer.critic_opt))
    # Separate collection/training RNG from architecture-dependent construction draws.
    set_global_seeds(cfg.train.seed + 123456)
    with torch.no_grad():
        trainer.world_state = trainer.world.observe_step(
            trainer.world.init_state(1),
            trainer.env_state["state"],
            torch.zeros(1, trainer.action_dim, device=trainer.device),
            is_first=torch.ones(1, dtype=torch.bool, device=trainer.device),
        )["post"].detach()
    return {name: state_hash_without_encoder(getattr(trainer, name)) for name in ("world", "actor", "critic")}


def verify_gate(path, *, root=ROOT, require_approval=True):
    """The manifest may live in a separate reviewed Git checkout, never only in runs/."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if not manifest.get("source_commit") or not manifest.get("source_fingerprints"):
        raise ValueError("Unsealed experiment: review, commit, annotate a new tag and record fingerprints first")
    if require_approval and not manifest.get("launch_approval"):
        raise ValueError("Explicit user approval is required before formal training")
    manifest_root = Path(git("rev-parse", "--show-toplevel", root=path.parent))
    relative = str(path.relative_to(manifest_root))
    committed = subprocess.check_output(["git", "-C", str(manifest_root), "show", "HEAD:" + relative])
    if hashlib.sha256(committed).hexdigest() != sha256(path):
        raise ValueError("Execution manifest must be reviewed and committed, not only generated")
    tag = manifest["source_tag"]
    if git("rev-parse", tag, root=root) != manifest["tag_object"]:
        raise ValueError("Annotated tag object differs from the recorded experiment")
    if git("cat-file", "-t", tag, root=root) != "tag":
        raise ValueError("Experiment tag must be annotated")
    if (
        git("rev-parse", tag + "^{}", root=root) != manifest["source_commit"]
        or git("rev-parse", "HEAD", root=root) != manifest["source_commit"]
    ):
        raise ValueError("Training must run from exactly the recorded tagged source commit")
    if git("status", "--porcelain", root=root):
        raise ValueError("Training source checkout is not clean")
    inventory = {name for name in git("ls-files", root=root).splitlines() if (root / name).is_file()}
    if set(manifest["source_fingerprints"]) != inventory:
        raise ValueError("Source fingerprint inventory does not cover the complete tagged checkout")
    for relative, expected in manifest["source_fingerprints"].items():
        if sha256(root / relative) != expected:
            raise ValueError("Source/configuration/dependency fingerprint mismatch: " + relative)
    config_path = root / "environments/cage2/host_aware_v1_config.json"
    if sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("Experimental configuration fingerprint mismatch")
    import dreamerrl

    if not Path(dreamerrl.__file__).resolve().is_relative_to(root.resolve()):
        raise ValueError("Imported Dreamer source is outside the tagged checkout")
    dependency = manifest["dependencies"]
    vendor = root / "environments/cage2/vendor/cage-challenge-2"
    if git("rev-parse", "HEAD", root=vendor) != dependency["cyborg_revision"] or git(
        "status", "--porcelain", root=vendor
    ):
        raise ValueError("CybORG submodule is not clean and pinned")
    import CybORG

    if not Path(CybORG.__file__).resolve().is_relative_to(vendor.resolve()):
        raise ValueError("Imported CybORG source is outside the pinned submodule")
    if sys.version_info[:3] != (3, 10, 19) or platform.machine() != "aarch64":
        raise ValueError("Formal v1 experiment requires the verified Spark Python/architecture")
    installed = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in importlib.metadata.distributions()}
    if installed != dependency["package_versions"]:
        raise ValueError("Installed package set differs from the sealed manifest")
    for name, expected in dependency["package_versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("Installed dependency mismatch: " + name)
    native = root / dependency["native_wheel_path"]
    if sha256(native) != dependency["native_wheel_sha256"]:
        raise ValueError("Native wheel fingerprint mismatch")
    return manifest
