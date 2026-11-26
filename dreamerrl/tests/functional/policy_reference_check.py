"""Bounded flat PopGym reference snapshot, runnable against old or current source."""

import argparse
import json
from pathlib import Path

import torch

import dreamerrl
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import DreamerConfig


def snapshot(logdir):
    cfg = DreamerConfig()
    cfg.env.env_id = "popgym-RepeatPreviousEasy-v0"
    cfg.env.num_envs = 2
    cfg.env.max_episode_steps = 6
    cfg.env.parallel = False
    cfg.env.seed = 19
    cfg.world.deter_size = 12
    cfg.world.num_classes = 3
    cfg.world.stoch_size = 4
    cfg.world.hidden_size = 16
    cfg.world.imagination_horizon = 5
    cfg.world.disable_aux_losses = True
    cfg.ac.actor_hidden = cfg.ac.critic_hidden = 16
    cfg.train.cuda = False
    cfg.train.use_amp = False
    cfg.train.enable_fused_adamw_kernels = False
    cfg.train.seed = 19
    cfg.train.freeze_actor_critic_steps = 0
    cfg.train.random_exploration_steps = 0
    cfg.train.deterministic_env = True
    cfg.train.deterministic_imagination = True
    cfg.train.collect_steps = 12
    cfg.train.seq_len = 4
    cfg.train.burn_in = 3
    cfg.train.batch_size = 2
    cfg.log.tb_logdir = str(logdir)
    cfg.log.run_name = "flat_reference"
    cfg.log.enable_wandb = False
    trainer = DreamerTrainer(cfg)

    def weights():
        return {
            name: {key: value.detach().clone() for key, value in getattr(trainer, name).state_dict().items()}
            for name in ("world", "actor", "critic", "slow_critic")
        }

    try:
        initial = weights()
        rng = torch.get_rng_state().clone()
        trainer.collect_env_steps()
        batch = trainer.replay.sample(2, seed=3)
        metrics = trainer.update_world_model(batch, 0)
        a, c = trainer.update_actor_critic(batch, 0)
        gradients = {
            name: {key: p.grad.clone() for key, p in getattr(trainer, name).named_parameters() if p.grad is not None}
            for name in ("world", "actor", "critic")
        }
        return dict(
            initial=initial,
            post_initialization_rng=rng,
            batch=batch,
            losses=torch.tensor([metrics.total_loss.item(), a, c]),
            final=weights(),
            gradients=gradients,
            counters=trainer._optimizer_steps,
            final_rng=torch.get_rng_state().clone(),
        )
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


def assert_equal(actual, expected):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_equal(actual[key], expected[key])
    else:
        assert actual == expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    actual = snapshot(args.output.parent / "tensorboard")
    if args.reference is not None:
        assert_equal(actual, torch.load(args.reference, weights_only=True))
        print("PASS exact flat initialization, RNG, replay, losses, gradients, optimizer update and final weights")
    torch.save(actual, args.output)
    args.output.with_suffix(".source.json").write_text(json.dumps({"imported_source": dreamerrl.__file__}) + "\n")
