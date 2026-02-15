import torch

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import LSTMGates, RolloutStep


def test_rollout_buffer_pre_step_alignment():
    """
    PRE‑STEP Hidden State Alignment Test

    Ensures the rollout buffer stores the PRE‑STEP LSTM state (h_t, c_t)
    used to produce action[t].
    """

    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    device = trainer.device
    buf = trainer.buffer

    T = buf.cfg.rollout_steps
    B = buf.cfg.num_envs
    H = buf.cfg.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim

    obs = torch.randn(T, B, obs_dim, device=device)

    # Initial PRE‑STEP state
    h = torch.randn(B, H, device=device)
    c = torch.randn(B, H, device=device)

    buf.reset()

    pre_h_list = []
    pre_c_list = []

    for t in range(T):
        pre_h = h.clone()
        pre_c = c.clone()

        logits, value, h_new, c_new, _ = policy.forward_step(obs[t], h, c)

        dummy_gates = LSTMGates(
            i_gates=torch.zeros(B, 1, H, device=device),
            f_gates=torch.zeros(B, 1, H, device=device),
            g_gates=torch.zeros(B, 1, H, device=device),
            o_gates=torch.zeros(B, 1, H, device=device),
            c_gates=torch.zeros(B, 1, H, device=device),
            h_gates=torch.zeros(B, 1, H, device=device),
        )

        step = RolloutStep(
            obs=obs[t],
            actions=torch.zeros(B, device=device),
            rewards=torch.zeros(B, device=device),
            values=value.squeeze(-1).detach(),  # FIXED SHAPE (B,)
            logprobs=torch.zeros(B, device=device),
            terminated=torch.zeros(B, dtype=torch.bool, device=device),
            truncated=torch.zeros(B, dtype=torch.bool, device=device),
            hxs=pre_h,
            cxs=pre_c,
            gates=dummy_gates,
        )
        buf.add(step)

        pre_h_list.append(pre_h)
        pre_c_list.append(pre_c)

        h, c = h_new.detach(), c_new.detach()

    pre_h_stack = torch.stack(pre_h_list, dim=0)
    pre_c_stack = torch.stack(pre_c_list, dim=0)

    assert torch.allclose(buf.hxs, pre_h_stack, atol=1e-6)
    assert torch.allclose(buf.cxs, pre_c_stack, atol=1e-6)
