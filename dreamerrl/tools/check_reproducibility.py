#!/usr/bin/env python3
"""
End‑to‑end reproducibility check for Dreamer.

Runs two short Dreamer training runs with the same seed and verifies that:
  • world model losses match
  • actor losses match
  • critic losses match
  • replay sampling matches
  • latent transitions match
  • action sampling matches

If anything diverges, the script prints a diff and exits with code 1.
"""

import copy

import torch

from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.seed import set_global_seeds


def run_once(cfg, steps=10):
    """Run a short deterministic Dreamer loop and record metrics."""
    set_global_seeds(cfg.train.seed)

    trainer = DreamerTrainer(cfg)

    logs = {
        "wm_loss": [],
        "actor_loss": [],
        "critic_loss": [],
        "replay_samples": [],
        "latent_states": [],
        "actions": [],
    }

    for update_idx in range(steps):
        # deterministic environment interaction
        trainer.collect_env_steps()

        # deterministic replay sampling
        batch = trainer.replay.sample(
            batch_size=cfg.train.batch_size,
            seq_len=cfg.train.seq_len,
            device=trainer.device,
        )
        logs["replay_samples"].append(copy.deepcopy(batch))

        # world model update
        wm_loss = trainer.update_world_model(batch, update_idx)
        logs["wm_loss"].append(wm_loss)

        # actor‑critic update
        actor_loss, critic_loss = trainer.update_actor_critic(batch, update_idx)
        logs["actor_loss"].append(actor_loss)
        logs["critic_loss"].append(critic_loss)

        # latent transition (RSSM observe_step)
        # use the current world_state and a dummy env step
        # this gives deterministic latent evolution
        with torch.no_grad():
            dummy = {
                "state": trainer.env_state["state"],
                "reward": trainer.env_state["reward"],
                "is_first": trainer.env_state["is_first"],
                "is_last": trainer.env_state["is_last"],
                "is_terminal": trainer.env_state["is_terminal"],
                "info": trainer.env_state["info"],
            }

            action_dim = trainer.world.net_cfg.action_dim
            assert action_dim is not None, "action_dim must be specified in config for reproducibility check"

            action = torch.zeros(
                (trainer.world_state.h.shape[0], action_dim),
                device=trainer.device,
            )

            latent_out = trainer.world.observe_step(trainer.world_state, dummy["state"], action=action)
            logs["latent_states"].append(
                {
                    "h": latent_out["state"]["h"].detach().cpu(),
                    "z": latent_out["state"]["z"].detach().cpu(),
                }
            )

        # deterministic action sampling
        with torch.no_grad():
            actions, _ = trainer.actor.act(trainer.world_state)
            logs["actions"].append(actions.cpu())

    return logs


def assert_same(a, b, name):
    """Recursively verify equality."""
    if isinstance(a, torch.Tensor):
        if not torch.allclose(a, b, atol=1e-6, rtol=1e-6):
            print(f"\n❌ Divergence in {name}")
            print("A:", a)
            print("B:", b)
            raise SystemExit(1)
    elif isinstance(a, dict):
        for k in a:
            assert_same(a[k], b[k], f"{name}.{k}")
    elif isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{name}[{i}]")
    else:
        if a != b:
            print(f"\n❌ Divergence in {name}: {a} != {b}")
            raise SystemExit(1)


def main():
    from dreamerrl.utils.types import make_default_config

    cfg = make_default_config()
    cfg.train.seed = 123
    cfg.train.batch_size = 4
    cfg.train.seq_len = 16

    print("Running reproducibility check...\n")

    logs1 = run_once(cfg)
    logs2 = run_once(cfg)

    print("Comparing logs...")

    assert_same(logs1["wm_loss"], logs2["wm_loss"], "world_model_loss")
    assert_same(logs1["actor_loss"], logs2["actor_loss"], "actor_loss")
    assert_same(logs1["critic_loss"], logs2["critic_loss"], "critic_loss")
    assert_same(logs1["replay_samples"], logs2["replay_samples"], "replay_samples")
    assert_same(logs1["latent_states"], logs2["latent_states"], "latent_states")
    assert_same(logs1["actions"], logs2["actions"], "actions")

    print("\n✅ Dreamer reproducibility verified.")


if __name__ == "__main__":
    main()
