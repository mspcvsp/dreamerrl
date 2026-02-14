import torch


def test_rollout_vs_replay_equivalence_gpu(deterministic_trainer):
    """
    GPU version of rollout vs replay equivalence.
    Mirrors CPU test exactly.
    """

    trainer = deterministic_trainer
    trainer.policy.to("cuda")
    trainer.state.cfg.lstm.dropconnect_p = 0.0
    trainer.policy.eval()

    # Collect rollout on GPU
    trainer.collect_rollout()

    # Full replay (CPU test uses this)
    full = trainer.replay_policy_on_rollout()

    # Replay again to check determinism
    replay = trainer.replay_policy_on_rollout()

    def assert_close(a, b, name, atol=1e-5, rtol=1e-5):
        assert torch.allclose(a.cpu(), b.cpu(), atol=atol, rtol=rtol), f"{name} mismatch"

    assert_close(full.values, replay.values, "values")
    assert_close(full.logprobs, replay.logprobs, "logprobs")
    assert_close(full.new_hxs, replay.new_hxs, "new_hxs")
    assert_close(full.new_cxs, replay.new_cxs, "new_cxs")
