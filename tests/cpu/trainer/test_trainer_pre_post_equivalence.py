import pytest
import torch

from dreamerrl.buffer import RecurrentRolloutBuffer
from dreamerrl.policy import LSTMPPOPolicy
from dreamerrl.trainer_state import TrainerState
from dreamerrl.types import PolicyEvalInput, PolicyInput, RolloutStep

pytestmark = pytest.mark.cpu


def test_trainer_pre_post_equivalence(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    B = trainer_state.cfg.buffer_config.num_envs
    T = trainer_state.cfg.buffer_config.rollout_steps
    H = trainer_state.cfg.lstm.lstm_hidden_size

    policy = LSTMPPOPolicy(trainer_state)
    buffer = RecurrentRolloutBuffer(trainer_state, device="cpu")

    # Fake rollout
    obs = torch.randn(T, B, trainer_state.env_info.flat_obs_dim)
    actions = torch.randint(0, trainer_state.env_info.action_dim, (T, B))
    h = torch.zeros(B, H)
    c = torch.zeros(B, H)

    for t in range(T):
        out = policy.forward(PolicyInput(obs=obs[t], hxs=h, cxs=c))
        buffer.add(
            RolloutStep(
                obs=obs[t],
                actions=actions[t],
                rewards=torch.zeros(B),
                values=out.values,
                logprobs=torch.zeros(B),
                terminated=torch.zeros(B, dtype=torch.bool),
                truncated=torch.zeros(B, dtype=torch.bool),
                hxs=h,
                cxs=c,
                gates=out.gates,
            )
        )
        h, c = out.new_hxs, out.new_cxs

    # Full unroll
    full = policy.evaluate_actions_sequence(
        PolicyEvalInput(
            obs=obs,
            hxs=buffer.hxs[0],
            cxs=buffer.cxs[0],
            actions=actions,
        )
    )

    # Compare buffer PRE-STEP to full PRE-STEP
    assert torch.allclose(buffer.hxs, full.pre_hxs, atol=1e-6)
    assert torch.allclose(buffer.cxs, full.pre_cxs, atol=1e-6)
