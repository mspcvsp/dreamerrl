"""Record/verify disabled-core equivalence across auxiliary cleanup; synthetic backward only."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from dreamerrl.models.world_model import WorldModel
from dreamerrl.training.core.world_model_update import world_model_training_step
from dreamerrl.utils.types import DreamerConfig, LatentConfig, NetworkConfig


def snapshot() -> dict[str, Any]:
    torch.manual_seed(731)
    cfg = DreamerConfig()
    cfg.world.disable_aux_losses = True
    cfg.train.deterministic_env = True
    wm = WorldModel(
        obs_space=gym.spaces.Box(-100, 100, (4,), np.float32),
        dreamer_cfg=cfg,
        latent=LatentConfig(deter_size=16, num_classes=3, stoch_size=4),
        net=NetworkConfig(hidden_size=16, action_dim=3, value_bins=21, bin_min=-5, bin_max=5),
    )
    rng = torch.get_rng_state().clone()
    batch = dict(
        obs=torch.randn(2, 4, 4),
        action=torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]]),
        reward=torch.randn(2, 4),
        is_terminal=torch.zeros(2, 4),
        is_first=torch.zeros(2, 4, dtype=torch.bool),
    )
    result = world_model_training_step(wm, batch)
    result.total_loss.backward()
    return dict(
        weights={key: value.detach().clone() for key, value in wm.state_dict().items()},
        gradients={key: value.grad.clone() for key, value in wm.named_parameters() if value.grad is not None},
        loss=result.total_loss.detach(),
        rng=rng,
    )


def checkpoint_checks(paths, output):
    from dreamerrl.training.trainer import DreamerTrainer

    records = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        trainer = DreamerTrainer.from_checkpoint(path, cuda=False, log_dir=output.parent / path.parent.name)
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            assert not trainer.world.aux_objectives
            assert all(torch.equal(value, trainer.world.state_dict()[key]) for key, value in saved["world"].items())
            records[str(path)] = dict(
                update=trainer.completed_updates,
                sha256=digest.hexdigest(),
                exact_world_weights=True,
                replay_episodes=len(trainer.replay.episodes),
            )
        finally:
            trainer.env.venv.close()
            trainer.tb.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    print("PASS: historical checkpoint weights and replay restored without updates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.checkpoint:
        if args.output is None:
            parser.error("--checkpoint requires --output")
        checkpoint_checks(args.checkpoint, args.output)
        raise SystemExit(0)
    if args.snapshot is None:
        parser.error("--snapshot or --checkpoint required")
    actual = snapshot()
    if args.record:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.save(actual, args.snapshot)
        print("Recorded pre-cleanup disabled baseline")
    else:
        expected = torch.load(args.snapshot, weights_only=True)
        for group in ("weights", "gradients"):
            assert actual[group].keys() == expected[group].keys()
            for key in actual[group]:
                torch.testing.assert_close(actual[group][key], expected[group][key], rtol=0, atol=0)
        for key in ("loss", "rng"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        print("PASS: exact disabled weights, state_dict keys, RNG, loss and gradients unchanged")
