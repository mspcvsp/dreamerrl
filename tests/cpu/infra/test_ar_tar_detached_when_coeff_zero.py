"""
AR/TAR must remain detached even when their coefficients are zero.

Why this matters:
-----------------
If ar_coef or tar_coef = 0, the model should still produce:
    • scalar AR/TAR values
    • non-negative values
    • tensors that do NOT require gradients

If AR/TAR ever require gradients when coef=0, it means the
regularization path is leaking autograd state, which would
corrupt PPO training and violate the detachment invariant.
"""

import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput


def test_ar_tar_detached_when_coeff_zero(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    # Explicitly disable AR/TAR
    trainer_state.cfg.lstm.lstm_ar_coef = 0.0
    trainer_state.cfg.lstm.lstm_tar_coef = 0.0

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 2, 4
    H = trainer_state.cfg.lstm.lstm_hidden_size
    D = trainer_state.env_info.flat_obs_dim

    obs = torch.randn(B, T, D)
    h0 = torch.zeros(B, H)
    c0 = torch.zeros(B, H)

    out = policy.forward(PolicyInput(obs=obs, hxs=h0, cxs=c0))

    # AR/TAR must be scalar, non-negative, and detached
    assert out.ar_loss.dim() == 0
    assert out.tar_loss.dim() == 0

    assert out.ar_loss >= 0
    assert out.tar_loss >= 0

    assert out.ar_loss.requires_grad is False
    assert out.tar_loss.requires_grad is False
