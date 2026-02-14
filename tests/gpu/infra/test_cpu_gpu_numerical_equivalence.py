import pytest
import torch

pytestmark = pytest.mark.gpu


def _clone_buffer_to_device(buf, device):
    """Clone rollout buffer tensors to a new device."""
    for name in [
        "obs",
        "actions",
        "rewards",
        "values",
        "logprobs",
        "terminated",
        "truncated",
        "hxs",
        "cxs",
        "returns",
        "advantages",
    ]:
        t = getattr(buf, name)
        setattr(buf, name, t.to(device))
    buf.device = device
    return buf


def test_cpu_gpu_eval_equivalence(deterministic_trainer):
    trainer = deterministic_trainer

    # Force CPU mode
    trainer.state.cfg.trainer.cuda = False
    trainer.device = torch.device("cpu")

    # 1) Rollout on CPU
    trainer.collect_rollout()
    cpu_eval = trainer.replay_policy_on_rollout()

    # 2) Move policy + buffer to GPU
    device = torch.device("cuda")
    trainer.policy = trainer.policy.to(device)
    trainer.buffer = _clone_buffer_to_device(trainer.buffer, device)
    trainer.device = device

    gpu_eval = trainer.replay_policy_on_rollout()

    def assert_close(a, b, name, rtol=1e-3, atol=1e-4):
        assert torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol), f"{name} mismatch"

    # Core PPO outputs
    assert_close(cpu_eval.values, gpu_eval.values, "values")
    assert_close(cpu_eval.logprobs, gpu_eval.logprobs, "logprobs")
    assert_close(cpu_eval.entropy, gpu_eval.entropy, "entropy")

    # Hidden states
    assert_close(cpu_eval.new_hxs, gpu_eval.new_hxs, "new_hxs")
    assert_close(cpu_eval.new_cxs, gpu_eval.new_cxs, "new_cxs")

    # Gates
    for name in ["i_gates", "f_gates", "g_gates", "o_gates", "c_gates", "h_gates"]:
        assert_close(getattr(cpu_eval.gates, name), getattr(gpu_eval.gates, name), f"gates.{name}")
