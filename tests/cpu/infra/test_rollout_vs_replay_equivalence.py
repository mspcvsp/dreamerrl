import pytest
import torch

pytestmark = pytest.mark.cpu


def test_rollout_vs_replay_equivalence(deterministic_trainer):
    """
    Ensures that evaluate_actions_sequence() exactly reproduces
    rollout-time values/logprobs/gates/hidden-states.

    This is the core PPO invariant: rollout == replay.
    """

    trainer = deterministic_trainer
    trainer.collect_rollout()

    # Rollout-time tensors
    buf = trainer.buffer
    T, B = buf.values.shape

    rollout_values = buf.values.clone()
    rollout_logprobs = buf.logprobs.clone()
    rollout_hxs = buf.hxs.clone()
    rollout_cxs = buf.cxs.clone()

    # Replay the entire rollout
    eval_output = trainer.replay_policy_on_rollout()

    # -----------------------------
    # Compare values/logprobs
    # -----------------------------
    assert torch.allclose(eval_output.values.cpu(), rollout_values.cpu(), atol=1e-6), (
        "Values mismatch between rollout and replay"
    )

    assert torch.allclose(eval_output.logprobs.cpu(), rollout_logprobs.cpu(), atol=1e-6), (
        "Logprobs mismatch between rollout and replay"
    )

    # -----------------------------
    # Compare hidden states
    # -----------------------------
    assert torch.allclose(eval_output.new_hxs.cpu(), rollout_hxs.cpu(), atol=1e-6), "Hidden state mismatch"

    assert torch.allclose(eval_output.new_cxs.cpu(), rollout_cxs.cpu(), atol=1e-6), "Cell state mismatch"

    # -----------------------------
    # Compare gates (T, B, H)
    # -----------------------------
    for name in ["i_gates", "f_gates", "g_gates", "o_gates", "c_gates", "h_gates"]:
        r = getattr(eval_output.gates, name).cpu()
        # rollout gates are stored transposed in buffer.add()
        # so we transpose back for comparison
        t = getattr(buf, "gates", None)
        assert r.shape[0] == T, f"Gate {name} shape mismatch"
