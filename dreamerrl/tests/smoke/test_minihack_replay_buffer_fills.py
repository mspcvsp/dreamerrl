import pytest

from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


@pytest.mark.smoke
@pytest.mark.minihack_smoke
def test_minihack_replay_buffer_fills():
    cfg = make_default_config()

    cfg.env.env_id = "MiniHack-Room-5x5-v0"
    cfg.env.num_envs = 4
    cfg.env.max_episode_steps = 50

    cfg.train.seed = 0
    cfg.train.cuda = False

    # Ensure an episode can complete.
    cfg.train.collect_steps = 60

    trainer = DreamerTrainer(cfg)

    trainer.collect_env_steps()

    assert len(trainer.replay.episodes) > 0
