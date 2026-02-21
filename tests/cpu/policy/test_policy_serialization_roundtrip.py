import torch

from dreamerrl.policy import LSTMPPOPolicy
from dreamerrl.types import PolicyEvalInput
from tests.helpers.fake_batch import make_fake_batch
from tests.helpers.fake_rollout import FakeRolloutBuilder
from tests.helpers.fake_state import make_fake_state


def test_policy_serialization_roundtrip():
    """
    policy -> state_dict -> new policy must preserve all outputs.
    """

    # --- Build fake state ---
    state = make_fake_state(
        rollout_steps=8,
        num_envs=2,
        obs_dim=4,
        hidden_size=4,
    )

    # --- Build policies (ignore protocol mismatch) ---
    policy1 = LSTMPPOPolicy(state)  # type: ignore[arg-type]
    sd = policy1.state_dict()

    policy2 = LSTMPPOPolicy(state)  # type: ignore[arg-type]
    policy2.load_state_dict(sd)

    # --- Build a fake rollout + batch ---
    rollout = FakeRolloutBuilder(T=8, B=2, obs_dim=4).build()
    batch = make_fake_batch(state, rollout)

    # --- Build PolicyEvalInput ---
    h0 = batch.hxs[0]  # (B,H)
    c0 = batch.cxs[0]  # (B,H)

    inp = PolicyEvalInput(
        obs=batch.obs,  # (T,B,obs_dim)
        hxs=h0,  # (B,H)
        cxs=c0,  # (B,H)
        actions=batch.actions,  # (T,B)
    )

    # --- Evaluate both policies ---
    out1 = policy1.evaluate_actions_sequence(inp)
    out2 = policy2.evaluate_actions_sequence(inp)

    # --- Assertions ---
    def assert_close(a, b, name):
        assert torch.allclose(a, b, atol=1e-6), f"{name} mismatch"

    assert_close(out1.values, out2.values, "values")
    assert_close(out1.logprobs, out2.logprobs, "logprobs")
    assert_close(out1.new_hxs, out2.new_hxs, "new_hxs")
    assert_close(out1.new_cxs, out2.new_cxs, "new_cxs")
