import pytest
import torch

from lstmppo.policy import LSTMPPOPolicy
from lstmppo.types import PolicyEvalInput
from tests.helpers.fake_batch import make_fake_batch
from tests.helpers.fake_rollout import FakeRolloutBuilder
from tests.helpers.fake_state import make_fake_state


@pytest.mark.gpu
def test_policy_serialization_roundtrip_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    # --- Fake state stays on CPU ---
    state = make_fake_state(
        rollout_steps=8,
        num_envs=2,
        obs_dim=4,
        hidden_size=4,
    )

    # --- Policies move to GPU ---
    policy1 = LSTMPPOPolicy(state).to(device)  # type: ignore[arg-type]
    sd = policy1.state_dict()

    policy2 = LSTMPPOPolicy(state).to(device)  # type: ignore[arg-type]
    policy2.load_state_dict(sd)

    # --- Fake rollout + batch on GPU ---
    rollout = FakeRolloutBuilder(T=8, B=2, obs_dim=4, device="cuda").build()
    batch = make_fake_batch(state, rollout, device="cuda")

    # --- Build PolicyEvalInput ---
    h0 = batch.hxs[0].to(device)
    c0 = batch.cxs[0].to(device)

    inp = PolicyEvalInput(
        obs=batch.obs.to(device),
        hxs=h0,
        cxs=c0,
        actions=batch.actions.to(device),
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
