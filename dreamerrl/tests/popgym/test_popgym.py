"""Short PopGym training contract check, run in the regular suite."""

import pytest
import torch

from dreamerrl.tests.manual.utils import minihack_config
from dreamerrl.training.trainer import DreamerTrainer


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.popgym_functional
def test_popgym_training_contract(tmp_path, monkeypatch, parallel):
    cfg = minihack_config(0, tmp_path)
    cfg.env.env_id = "popgym-RepeatPreviousEasy-v0"
    cfg.env.parallel = parallel
    trainer = DreamerTrainer(cfg)
    try:
        # This is a prerequisite, not proof of memory-task learning.
        original_step = trainer.env.step

        def checked_step(actions):
            out = original_step(actions)
            assert "transition_state" in out
            assert (~out["is_terminal"] | out["is_last"]).all()
            return out

        monkeypatch.setattr(trainer.env, "step", checked_step)
        trainer.collect_env_steps()
        batch = trainer.replay.sample(cfg.train.batch_size)
        metrics = trainer.update_world_model(batch, 0)
        actor_loss, critic_loss = trainer.update_actor_critic(batch, 0)
        assert torch.isfinite(torch.tensor([metrics.total_loss.item(), actor_loss, critic_loss])).all()
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
