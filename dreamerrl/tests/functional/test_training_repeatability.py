"""Short same-seed training repeatability check; runs in the regular suite."""

import pytest
import torch

from dreamerrl.tests.manual.utils import minihack_config
from dreamerrl.training.trainer import DreamerTrainer


def training_trace(seed, logdir):
    trainer = DreamerTrainer(minihack_config(seed, logdir))
    modules = {
        "world": trainer.world,
        "actor": trainer.actor,
        "critic": trainer.critic,
        "slow_critic": trainer.slow_critic,
    }
    initial = {name: [p.detach().clone() for p in module.parameters()] for name, module in modules.items()}
    trace = []
    try:
        for update in range(3):
            trainer.collect_env_steps()
            batch = trainer.replay.sample(2)
            trace.append({k: v.cpu().clone() for k, v in batch.items()})
            metrics = trainer.update_world_model(batch, update)
            actor_loss, critic_loss = trainer.update_actor_critic(batch, update)
            losses = torch.tensor([metrics.total_loss.item(), actor_loss, critic_loss])
            assert torch.isfinite(losses).all()
            trace.append({"losses": losses})
        for name, module in modules.items():
            assert any(not torch.equal(a, b) for a, b in zip(initial[name], module.parameters())), name
        trace.append(
            {
                f"{name}.{key}": tensor.detach().cpu().clone()
                for name, module in modules.items()
                for key, tensor in module.state_dict().items()
            }
        )
        trace.append({"returns": torch.tensor(trainer.recent_returns)})
        return trace
    finally:
        trainer.env.venv.close()
        trainer.tb.close()


@pytest.mark.reproducibility
def test_same_seed_training_repeatability(tmp_path):
    first = training_trace(7, tmp_path / "first")
    second = training_trace(7, tmp_path / "second")
    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, msg=key)
