import torch

from tests.helpers.fake_policy import make_fake_policy


def test_tbptt_chunked_matches_full_sequence():
    """
    TBPTT forward_tbptt must be numerically identical to full forward_sequence
    for the same obs, h0, c0.
    """
    policy = make_fake_policy()

    T, B, D, H = 12, 3, 4, 4
    obs = torch.randn(T, B, D)
    h0 = torch.randn(B, H)
    c0 = torch.randn(B, H)

    # Full unroll
    full = policy.forward_sequence(obs, h0, c0)
    logits_full = full.logits  # (T, B, A)
    value_full = full.value  # (T, B)
    hn_full = full.hn  # (T, B, H)
    cn_full = full.cn  # (T, B, H)

    # Chunked TBPTT
    chunk_size = 4
    tbptt = policy.forward_tbptt(obs, h0, c0, chunk_size=chunk_size)
    logits_tb = tbptt.logits  # (T, B, A)
    value_tb = tbptt.value  # (T, B)
    hn_tb = tbptt.hn  # (T, B, H)
    cn_tb = tbptt.cn  # (T, B, H)

    # Bit-exact alignment
    assert torch.allclose(logits_tb, logits_full)
    assert torch.allclose(value_tb, value_full)
    assert torch.allclose(hn_tb, hn_full)
    assert torch.allclose(cn_tb, cn_full)
