import pytest
import torch

from dreamerrl.trainer import LSTMPPOTrainer
from dreamerrl.trainer_state import TrainerState
from dreamerrl.types import Config
from tests.helpers.fake_state import TrainerStateProtocol, make_fake_state
from tests.helpers.runtime_env import make_runtime_env_info


@pytest.fixture
def trainer_state():
    cfg = Config()
    cfg.trainer.debug_mode = True

    state = TrainerState(cfg)
    state.env_info = make_runtime_env_info()
    return state


@pytest.fixture
def fake_state() -> TrainerStateProtocol:
    return make_fake_state()


@pytest.fixture
def deterministic_trainer():
    trainer = LSTMPPOTrainer.for_validation()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer.policy.eval()
    trainer.state.cfg.trainer.debug_mode = True

    return trainer
