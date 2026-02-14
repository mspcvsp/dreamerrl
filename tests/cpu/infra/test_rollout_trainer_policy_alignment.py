"""
Rollout buffer → trainer → policy shape alignment.

This test ensures that:
    • rollout buffer produces (T, B, ...)
    • trainer slices TBPTT chunks correctly
    • policy forward accepts (B, K, ...)
    • returned shapes match expected invariants

If any of these drift, TBPTT breaks and PPO gradients become misaligned.
"""

import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.trainer_state import TrainerState
from lstmppo.types import PolicyInput
from tests.helpers.fake_rollout import FakeRolloutBuilder


def test_rollout_trainer_policy_alignment(trainer_state: TrainerState):
    cfg = trainer_state.cfg

    T = cfg.trainer.rollout_steps
    B = cfg.env.num_envs
    obs_dim = trainer_state.env_info.flat_obs_dim  # ← FIXED

    rollout = FakeRolloutBuilder(T, B, obs_dim).build()

    policy = LSTMPPOPolicy(trainer_state)
    policy.eval()

    assert rollout.obs.shape == (T, B, obs_dim)
    assert rollout.next_obs.shape == (T, B, obs_dim)

    for t0 in range(0, T, cfg.trainer.tbptt_chunk_len):
        t1 = min(t0 + cfg.trainer.tbptt_chunk_len, T)

        obs_chunk = rollout.obs[t0:t1]  # (K, B, D)
        K = obs_chunk.shape[0]

        obs_batch = obs_chunk.transpose(0, 1)  # (B, K, D)

        H = cfg.lstm.lstm_hidden_size
        h0 = torch.zeros(B, H)
        c0 = torch.zeros(B, H)

        out = policy.forward(
            PolicyInput(
                obs=obs_batch,
                hxs=h0,
                cxs=c0,
            )
        )

        assert out.logits.shape[:2] == (B, K)
        assert out.values.shape[:2] == (B, K)
        assert out.pred_obs.shape[:2] == (B, K)
        assert out.pred_raw.shape[:2] == (B, K)
        assert out.gates.i_gates.shape[:2] == (B, K)
