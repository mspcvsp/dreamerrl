"""Read-only-model synthetic verification for the auxiliary-objective audit; no optimizer steps."""

import argparse
import copy
import hashlib
import io
import json
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from dreamerrl.models.aux_objectives import AUX_OBJECTIVES
from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.core.world_model_update import world_model_training_step
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig

ROOT = Path(__file__).resolve().parents[2]


def main(output):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    cfg = DreamerConfig()
    cfg.world.aux_objectives = list(AUX_OBJECTIVES.values())
    latent = LatentConfig(deter_size=16, num_classes=3, stoch_size=4)
    net = NetworkConfig(hidden_size=16, action_dim=3, value_bins=21, bin_min=-5, bin_max=5)

    def make(config):
        return WorldModel(
            obs_space=gym.spaces.Box(-100, 100, (4,), np.float32), dreamer_cfg=config, latent=latent, net=net
        )

    wm = make(cfg)
    batch = dict(
        obs=torch.randn(2, 4, 4),
        action=torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]]),
        reward=torch.randn(2, 4),
        is_terminal=torch.zeros(2, 4),
        is_first=torch.zeros(2, 4, dtype=torch.bool),
        action_mask=torch.ones(2, 4, 3),
    )
    results = {}
    for name, objective in AUX_OBJECTIVES.items():
        wm.zero_grad(set_to_none=True)
        wm.aux_objectives = [objective]
        captured = []

        def capture(module, args, result):
            for value in args:
                value.retain_grad()
            captured.append(args)

        hook = wm.aux_heads[name].register_forward_hook(capture)
        loss = world_model_training_step(wm, batch).aux_losses[0]
        hook.remove()
        loss.backward()
        gradients = {}
        for key in ("encoder", "rssm", "posterior", "prior", "decoder", "reward_heads", "continue_head"):
            gradients[key] = float(
                sum(p.grad.abs().sum().item() for p in getattr(wm, key).parameters() if p.grad is not None)
            )
        gradients["aux_head"] = float(
            sum(p.grad.abs().sum().item() for p in wm.aux_heads[name].parameters() if p.grad is not None)
        )
        gradients["h"] = float(captured[-1][0].grad.abs().sum())
        gradients["z"] = float(captured[-1][1].grad.abs().sum())
        assert torch.isfinite(loss)
        assert all(gradients[key] > 0 for key in ("encoder", "rssm", "posterior", "aux_head", "h", "z"))
        assert all(gradients[key] == 0 for key in ("prior", "decoder", "reward_heads", "continue_head"))
        results[name] = dict(loss=float(loss.detach()), absolute_gradient_sums=gradients)
    wm.aux_objectives = [AUX_OBJECTIVES["novelty"]]
    terminal_only = {key: value.clone() for key, value in batch.items()}
    terminal_only["is_terminal"][:, :-1] = 1
    terminal_only_loss = world_model_training_step(wm, terminal_only).aux_losses[0]
    flagged_last = dict(terminal_only, is_last=terminal_only["is_terminal"].bool())
    correctly_masked = world_model_training_step(wm, flagged_last).aux_losses[0]
    masked_loss_value = float(correctly_masked.detach())
    assert terminal_only_loss > 0 and masked_loss_value == 0
    state_only = {key: value for key, value in batch.items() if key != "obs"}
    state_only["state"] = batch["obs"]
    try:
        world_model_training_step(wm, state_only)
    except KeyError as exc:
        state_only_error = str(exc)
    else:
        raise AssertionError("Expected novelty callback's obs-only schema limitation")
    wm.aux_objectives = list(AUX_OBJECTIVES.values())
    snapshot = io.BytesIO()
    torch.save(dict(config=cfg, world=wm.state_dict()), snapshot)
    snapshot.seek(0)
    payload = torch.load(snapshot, weights_only=False)
    clone = make(payload["config"])
    clone.load_state_dict(payload["world"], strict=True)
    assert all(torch.equal(value, clone.state_dict()[key]) for key, value in wm.state_dict().items())
    disabled = copy.deepcopy(cfg)
    disabled.world.disable_aux_losses = True
    try:
        make(disabled).load_state_dict(payload["world"], strict=True)
    except RuntimeError:
        incompatible_rejected = True
    else:
        incompatible_rejected = False
    assert incompatible_rejected
    results["checks"] = dict(
        optimizer_steps=0,
        synthetic_seed=42,
        terminal_only_novelty_loss=float(terminal_only_loss.detach()),
        last_flag_masked_novelty_loss=masked_loss_value,
        state_alias_novelty_error=state_only_error,
        enabled_serialization_roundtrip=True,
        disabled_schema_rejects_enabled_weights=True,
    )
    results["source_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    results["uv_lock_sha256"] = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    results["verification_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
