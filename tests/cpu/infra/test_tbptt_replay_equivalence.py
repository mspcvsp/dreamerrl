import torch


def test_tbptt_replay_equivalence(deterministic_trainer):
    """
    Replay(full sequence) ≈ Replay(chunked TBPTT)
    Ensures that TBPTT slicing does not change numerical outputs.
    """

    trainer = deterministic_trainer
    trainer.state.cfg.lstm.dropconnect_p = 0.0
    trainer.policy.eval()

    trainer.collect_rollout()
    buf = trainer.buffer

    # Full replay
    full = trainer.replay_policy_on_rollout()

    # TBPTT replay
    trainer.state.cfg.trainer.tbptt_steps = 32
    chunked = trainer.replay_policy_on_rollout()

    def assert_close(a, b, name, rtol=1e-3, atol=1e-3):
        assert torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol), f"{name} mismatch"

    assert_close(full.values, chunked.values, "values")
    assert_close(full.logprobs, chunked.logprobs, "logprobs")
    assert_close(full.new_hxs, chunked.new_hxs, "new_hxs")
    assert_close(full.new_cxs, chunked.new_cxs, "new_cxs")
