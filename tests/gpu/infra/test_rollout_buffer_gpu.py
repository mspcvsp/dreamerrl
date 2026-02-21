import pytest
import torch

pytestmark = pytest.mark.gpu


def _make_gpu_buffer(trainer):
    trainer.state.cfg.trainer.cuda = True
    device = torch.device("cuda")
    buf = trainer.buffer.__class__(trainer.state, device)
    return trainer.state.cfg, device, buf


def test_gpu_buffer_initialization_shapes(deterministic_trainer):
    cfg, _, buf = _make_gpu_buffer(deterministic_trainer)

    T = cfg.trainer.rollout_steps
    B = cfg.env.num_envs
    D = deterministic_trainer.state.env_info.flat_obs_dim
    H = cfg.lstm.lstm_hidden_size

    expected = {
        "obs": (T, B, D),
        "actions": (T, B, 1),
        "rewards": (T, B),
        "values": (T, B),
        "logprobs": (T, B),
        "terminated": (T, B),
        "truncated": (T, B),
        "hxs": (T, B, H),
        "cxs": (T, B, H),
        "mask": (T, B),
    }

    for name, shape in expected.items():
        t = getattr(buf, name)
        assert t.shape == shape
        assert t.device.type == "cuda"


def test_gpu_buffer_add_and_copy_semantics(deterministic_trainer):
    cfg, device, buf = _make_gpu_buffer(deterministic_trainer)

    B = cfg.env.num_envs
    D = deterministic_trainer.state.env_info.flat_obs_dim
    H = cfg.lstm.lstm_hidden_size

    obs = torch.randn(B, D, device=device)
    act = torch.randn(B, device=device)
    rew = torch.randn(B, device=device)
    val = torch.randn(B, device=device)
    logp = torch.randn(B, device=device)
    term = torch.zeros(B, dtype=torch.bool, device=device)
    trunc = torch.zeros(B, dtype=torch.bool, device=device)
    hxs = torch.randn(B, H, device=device)
    cxs = torch.randn(B, H, device=device)

    from dreamerrl.types import LSTMGates, RolloutStep

    gates = LSTMGates(
        i_gates=hxs,
        f_gates=hxs,
        g_gates=hxs,
        o_gates=hxs,
        c_gates=hxs,
        h_gates=hxs,
    )

    step = RolloutStep(
        obs=obs,
        actions=act,
        rewards=rew,
        values=val,
        logprobs=logp,
        terminated=term,
        truncated=trunc,
        hxs=hxs,
        cxs=cxs,
        gates=gates,
    )

    buf.add(step)

    assert buf.step == 1
    assert torch.allclose(buf.obs[0], obs)
    assert buf.obs[0].data_ptr() != obs.data_ptr()  # copy, not reference
    assert buf.obs.device.type == "cuda"


def test_gpu_mask_logic(deterministic_trainer):
    _, device, buf = _make_gpu_buffer(deterministic_trainer)

    buf.terminated[0] = True
    buf.truncated[1] = True

    assert buf.mask.device.type == "cuda"
    assert (buf.mask[0] == 0).all()
    assert (buf.mask[1] == 0).all()
    assert (buf.mask[2:] == 1).all()


def test_gpu_reset_clears_state(deterministic_trainer):
    _, device, buf = _make_gpu_buffer(deterministic_trainer)

    buf.terminated[0] = True
    buf.truncated[0] = True
    buf.obs[0] = 123.0

    buf.reset()

    assert buf.step == 0
    assert (buf.terminated == 0).all()
    assert (buf.truncated == 0).all()
    assert (buf.mask == 1).all()
    assert (buf.obs == 0).all()
    assert buf.obs.device.type == "cuda"


def test_gpu_hidden_state_alignment(deterministic_trainer):
    trainer = deterministic_trainer
    trainer.state.cfg.trainer.cuda = True

    trainer.collect_rollout()
    batch = next(trainer.buffer.get_recurrent_minibatches())

    assert batch.hxs.device.type == "cuda"
    assert batch.cxs.device.type == "cuda"

    assert batch.hxs.shape[0] == trainer.rollout_steps
    assert batch.hxs.shape[1] == trainer.num_envs
