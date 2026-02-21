import torch

from dreamerrl.trainer import LSTMPPOTrainer
from dreamerrl.types import PolicyEvalInput


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
    actions = torch.randint(0, state.env_info.action_dim, (T, B), device=device)
    h0 = torch.zeros(B, H, device=device)
    c0 = torch.zeros(B, H, device=device)

    # Reference: full unroll using evaluate_actions_sequence
    full = policy.evaluate_actions_sequence(PolicyEvalInput(obs=obs, hxs=h0, cxs=c0, actions=actions))
    ref_pre = full.pre_hxs  # (T, B, H)
    ref_post = full.new_hxs  # (T, B, H)

    # TBPTT-style manual chunking
    h = h0
    c = c0
    stitched_pre = []
    stitched_post = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)

        out_chunk = policy.evaluate_actions_sequence(
            PolicyEvalInput(obs=obs[start:end], hxs=h, cxs=c, actions=actions[start:end])
        )

        stitched_pre.append(out_chunk.pre_hxs)
        stitched_post.append(out_chunk.new_hxs)

        # carry POST-STEP state into next chunk
        h = out_chunk.new_hxs[-1]
        c = out_chunk.new_cxs[-1]

    pre_tb = torch.cat(stitched_pre, dim=0)
    post_tb = torch.cat(stitched_post, dim=0)

    assert pre_tb.shape == ref_pre.shape
    assert post_tb.shape == ref_post.shape

    assert torch.allclose(pre_tb, ref_pre, atol=1e-5)
    assert torch.allclose(post_tb, ref_post, atol=1e-5)
