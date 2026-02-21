"""
Auxiliary prediction head shape invariants.

Why this matters:
-----------------
The auxiliary heads (next-observation and next-reward prediction)
provide dense supervision for the LSTM. Their shapes must remain
stable across refactors:

    pred_obs: (B, T, obs_dim)
    pred_rew: (B, T, 1)

If these shapes drift, TBPTT breaks, auxiliary losses misalign,
and PPO training becomes unstable.
"""

import torch

from dreamerrl.policy import LSTMPPOPolicy
from dreamerrl.trainer_state import TrainerState
from dreamerrl.types import PolicyInput


def test_aux_head_shapes(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 7
    trainer_state.env_info.action_dim = 3

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 4, 6
    D = trainer_state.env_info.flat_obs_dim
    H = trainer_state.cfg.lstm.lstm_hidden_size

    obs = torch.randn(B, T, D)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    # pred_obs: (B, T, obs_dim)
    assert out.pred_obs.shape == (B, T, D)

    # pred_rew: (B, T, 1)
    assert out.pred_raw.shape == (B, T, 1)
