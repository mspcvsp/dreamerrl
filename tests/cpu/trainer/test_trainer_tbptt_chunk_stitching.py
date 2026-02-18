import torch

from lstmppo.trainer import LSTMPPOTrainer


def test_trainer_tbptt_chunk_stitching():
    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    state = trainer.state
    device = trainer.device

    T = 16
    B = 2
    H = state.cfg.lstm.lstm_hidden_size
    obs_dim = state.env_info.flat_obs_dim
    chunk_size = state.cfg.trainer.tbptt_chunk_len

    obs = torch.randn(T, B, obs_dim, device=device)
    h0 = torch.zeros(B, H, device=device)
    c0 = torch.zeros(B, H, device=device)

    # Reference: full unroll
    full = policy.forward_sequence(obs, h0, c0)
    ref_h = full.hxs  # (T, B, H)
    ref_c = full.cxs  # (T, B, H)

    # TBPTT-style manual chunking at trainer level
    h = h0
    c = c0
    stitched_h = []
    stitched_c = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        out_chunk = policy.forward_sequence(obs[start:end], h, c)
        stitched_h.append(out_chunk.hxs)
        stitched_c.append(out_chunk.cxs)
        # carry last POST-STEP state into next chunk
        h = out_chunk.new_hxs[-1]
        c = out_chunk.new_cxs[-1]

    h_tb = torch.cat(stitched_h, dim=0)
    c_tb = torch.cat(stitched_c, dim=0)

    assert h_tb.shape == ref_h.shape
    assert c_tb.shape == ref_c.shape
    assert torch.allclose(h_tb, ref_h, atol=1e-5)
    assert torch.allclose(c_tb, ref_c, atol=1e-5)
