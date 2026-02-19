import torch

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import PolicyEvalInput


def test_tbptt_chunk_matches_full_sequence_eval(require_popgym_env):
    env_id = "popgym-RepeatPreviousEasy-v0"
    require_popgym_env(env_id)

    trainer = LSTMPPOTrainer.for_validation(env_id=env_id)
    trainer.collect_rollout()

    # Validation mode → exactly one minibatch
    minibatches = list(trainer.buffer.get_recurrent_minibatches())
    assert len(minibatches) == 1
    batch = minibatches[0]

    K = trainer.state.cfg.trainer.tbptt_chunk_len

    # -----------------------------
    # Full-sequence evaluation
    # -----------------------------
    full_eval = trainer.policy.evaluate_actions_sequence(
        PolicyEvalInput(
            obs=batch.obs,
            hxs=batch.hxs[0],
            cxs=batch.cxs[0],
            actions=batch.actions,
        )
    )

    # -----------------------------
    # TBPTT chunked evaluation
    # -----------------------------
    logits_chunks = []
    values_chunks = []
    h_chunks = []
    c_chunks = []

    for mb in batch.iter_chunks(K):
        out = trainer.policy.evaluate_actions_sequence(
            PolicyEvalInput(
                obs=mb.obs,
                hxs=mb.hxs0,
                cxs=mb.cxs0,
                actions=mb.actions,
            )
        )
        logits_chunks.append(out.logits)
        values_chunks.append(out.values)
        h_chunks.append(out.new_hxs)
        c_chunks.append(out.new_cxs)

    logits_tb = torch.cat(logits_chunks, dim=0)
    values_tb = torch.cat(values_chunks, dim=0)
    h_tb = torch.cat(h_chunks, dim=0)
    c_tb = torch.cat(c_chunks, dim=0)

    # -----------------------------
    # Invariants
    # -----------------------------
    assert torch.allclose(logits_tb, full_eval.logits, atol=1e-6)
    assert torch.allclose(values_tb, full_eval.values, atol=1e-6)
    assert torch.allclose(h_tb, full_eval.new_hxs, atol=1e-6)
    assert torch.allclose(c_tb, full_eval.new_cxs, atol=1e-6)
