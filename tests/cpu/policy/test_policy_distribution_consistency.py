import torch
from torch.distributions import Categorical

from dreamerrl.types import PolicyEvalInput
from tests.helpers.fake_policy import make_fake_policy


def test_sequence_logprob_entropy_match_logits():
    """
    Ensures evaluate_actions_sequence() produces logprobs and entropy
    exactly matching torch.distributions.Categorical(logits).
    """

    policy = make_fake_policy()

    T, B, D, H = 5, 3, 4, 4
    obs = torch.randn(T, B, D)
    h0 = torch.randn(B, H)
    c0 = torch.randn(B, H)

    # First pass: get logits
    # Use dummy actions; we will replace them later
    dummy_actions = torch.zeros(T, B, dtype=torch.long)
    inp = PolicyEvalInput(obs=obs, hxs=h0, cxs=c0, actions=dummy_actions)
    out = policy.evaluate_actions_sequence(inp)

    logits = out.logits  # (T, B, A)

    # Sample actions from the same logits to avoid mismatch
    dist = Categorical(logits=logits)
    actions = dist.sample()  # (T, B)

    # Re-run with real actions
    inp = PolicyEvalInput(obs=obs, hxs=h0, cxs=c0, actions=actions)
    out = policy.evaluate_actions_sequence(inp)

    # Manual distribution
    dist_manual = Categorical(logits=logits)
    manual_logprobs = dist_manual.log_prob(actions)
    manual_entropy = dist_manual.entropy()

    # Exact equality checks
    assert torch.allclose(out.logprobs, manual_logprobs)
    assert torch.allclose(out.entropy, manual_entropy)

    # Sanity: logits must be finite
    assert torch.isfinite(logits).all()
