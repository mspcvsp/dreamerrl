import pytest
import torch

pytestmark = pytest.mark.gpu


def test_cpu_gpu_eval_equivalence(deterministic_trainer):
    """
    Given the same rollout and the same policy weights,
    CPU and GPU evaluate_actions_sequence must produce
    numerically equivalent outputs.
    """

    trainer = deterministic_trainer
    trainer.state.cfg.trainer.cuda = False
    trainer.device = torch.device("cpu")

    # 1) Rollout on CPU
    trainer.collect_rollout()
    buf = trainer.buffer

    # Extract inputs for replay
    obs = buf.obs  # (T, B, D)
    actions = buf.actions  # (T, B, A)
    hxs = buf.hxs[0]  # initial h (B, H)
    cxs = buf.cxs[0]  # initial c (B, H)

    policy = trainer.policy

    # 2) CPU eval
    cpu_eval = policy.evaluate_actions_sequence(
        obs=obs,
        actions=actions,
        hxs=hxs,
        cxs=cxs,
    )

    # 3) Move policy + inputs to GPU
    device = torch.device("cuda")
    policy_gpu = policy.to(device)

    obs_gpu = obs.to(device)
    actions_gpu = actions.to(device)
    hxs_gpu = hxs.to(device)
    cxs_gpu = cxs.to(device)

    gpu_eval = policy_gpu.evaluate_actions_sequence(
        obs=obs_gpu,
        actions=actions_gpu,
        hxs=hxs_gpu,
        cxs=cxs_gpu,
    )

    def assert_close(a, b, name, rtol=1e-3, atol=1e-4):
        assert torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol), f"{name} mismatch"

    # Core outputs
    assert_close(cpu_eval.values, gpu_eval.values, "values")
    assert_close(cpu_eval.logprobs, gpu_eval.logprobs, "logprobs")
    assert_close(cpu_eval.entropy, gpu_eval.entropy, "entropy")

    # Hidden states
    assert_close(cpu_eval.new_hxs, gpu_eval.new_hxs, "new_hxs")
    assert_close(cpu_eval.new_cxs, gpu_eval.new_cxs, "new_cxs")

    # Gates
    for name in ["i_gates", "f_gates", "g_gates", "o_gates", "c_gates", "h_gates"]:
        assert_close(getattr(cpu_eval.gates, name), getattr(gpu_eval.gates, name), f"gates.{name}")
