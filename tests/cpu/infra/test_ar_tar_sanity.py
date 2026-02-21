"""
This test MUST use the real LSTMPPOPolicy and real TrainerState.

Why:
-----
Activation Regularization (AR) and Temporal Activation Regularization (TAR)
are scalar penalties applied to the LSTM hidden state:

    • AR  = λ * ||h_t||²
    • TAR = λ * ||h_t - h_{t-1}||²

Correct behavior:
    • AR and TAR must be scalars (0‑dim tensors)
    • They must always be ≥ 0
    • They must be detached from the computation graph
    • They must reflect real LSTM dynamics, not synthetic mocks

If AR/TAR ever become:
    • vectors or matrices → PPO loss breaks
    • negative → mathematically impossible
    • requiring gradients → PPO backprop explodes

This is a sentinel test for LSTM regularization correctness.
Do not replace the real model here.
"""

import torch

from dreamerrl.policy import LSTMPPOPolicy
from dreamerrl.trainer_state import TrainerState
from dreamerrl.types import PolicyInput


def test_ar_tar_sanity(trainer_state: TrainerState):
    assert trainer_state.env_info is not None
    trainer_state.env_info.flat_obs_dim = 4
    trainer_state.env_info.action_dim = 3

    # Enable AR/TAR explicitly
    trainer_state.cfg.lstm.lstm_ar_coef = 0.1
    trainer_state.cfg.lstm.lstm_tar_coef = 0.1

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    B, T = 3, 6
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
