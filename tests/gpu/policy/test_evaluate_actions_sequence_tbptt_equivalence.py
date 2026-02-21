import torch

from dreamerrl.policy import PolicyEvalInput
from dreamerrl.trainer import LSTMPPOTrainer


def test_evaluate_actions_sequence_tbptt_equivalence():
    trainer = LSTMPPOTrainer.for_validation()
    policy = trainer.policy
    device = trainer.device

    T = 24
    B = 3
    H = trainer.state.cfg.lstm.lstm_hidden_size
    obs_dim = trainer.state.env_info.flat_obs_dim
    act_dim = trainer.state.env_info.action_dim

    obs = torch.randn(T, B, obs_dim, device=device)
    h0 = torch.randn(B, H, device=device)
    c0 = torch.randn(B, H, device=device)
    actions = torch.randint(0, act_dim, (T, B, 1), device=device)

    # Full‑sequence evaluation
    full_in = PolicyEvalInput(obs=obs, hxs=h0, cxs=c0, actions=actions)
    full_out = policy.evaluate_actions_sequence(full_in)

    # TBPTT: chunked evaluation using forward_tbptt + evaluate_actions_sequence
    chunk_size = 5
    logits_chunks = []
    values_chunks = []

    h = h0
    c = c0
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)

        chunk_obs = obs[start:end]
        chunk_actions = actions[start:end]

        chunk_in = PolicyEvalInput(
            obs=chunk_obs,
            hxs=h,
            cxs=c,
            actions=chunk_actions,
        )
        chunk_out = policy.evaluate_actions_sequence(chunk_in)

        logits_chunks.append(chunk_out.logits)
        values_chunks.append(chunk_out.values)

        # carry final hidden state
        h = chunk_out.new_hxs[-1]
        c = chunk_out.new_cxs[-1]

    logits_tbptt = torch.cat(logits_chunks, dim=0)
    values_tbptt = torch.cat(values_chunks, dim=0)

    assert torch.allclose(full_out.logits, logits_tbptt, atol=1e-6)
    assert torch.allclose(full_out.values, values_tbptt, atol=1e-6)
