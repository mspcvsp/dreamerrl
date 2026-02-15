import torch

from lstmppo.policy import LSTMPPOPolicy
from tests.helpers.fake_buffer_loader import load_rollout_into_buffer
from tests.helpers.fake_rollout import FakeRolloutBuilder
from tests.helpers.fake_state import make_fake_state


def test_lstm_pre_step_to_post_step_state_flow():
    # 1. Build a fake trainer state and policy
    state = make_fake_state(
        rollout_steps=4,
        num_envs=2,
        obs_dim=4,
        hidden_size=4,
    )
    policy = LSTMPPOPolicy(state)  # type: ignore[arg-type]

    # 2. Build a simple rollout and load it into the buffer
    rollout = FakeRolloutBuilder(T=4, B=2, obs_dim=4, device="cpu").build()
    buf = load_rollout_into_buffer(state, rollout, device="cpu")

    # 3. Take a single timestep t
    t = 0
    obs_t = buf.obs[t]  # (B, D)
    h_t = buf.hxs[t]  # PRE-STEP (B, H)
    c_t = buf.cxs[t]  # PRE-STEP (B, H)

    # 4. Run the policy one step from (h_t, c_t)
    _, _, hxs, cxs, _ = policy.forward_step(obs_t, h_t, c_t)
    h_tp1 = hxs[-1]  # POST-STEP (B, H)
    c_tp1 = cxs[-1]  # POST-STEP (B, H)

    # 5. Re-run the underlying LSTM directly and check we get the same post-step state
    lstm = state.lstm
    lstm_input = obs_t.unsqueeze(1)  # (B, 1, D)
    h0 = h_t.unsqueeze(0)  # (1, B, H)
    c0 = c_t.unsqueeze(0)  # (1, B, H)

    _, (h1_direct, c1_direct) = lstm(lstm_input, (h0, c0))

    assert torch.allclose(h_tp1, h1_direct.squeeze(0))
    assert torch.allclose(c_tp1, c1_direct.squeeze(0))

    # 6. And the buffer must still contain the PRE-STEP state
    assert torch.allclose(buf.hxs[t], h_t)
    assert torch.allclose(buf.cxs[t], c_t)
