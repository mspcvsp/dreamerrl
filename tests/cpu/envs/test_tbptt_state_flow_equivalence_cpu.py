import torch

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import PolicyEvalInput


def test_tbptt_state_flow_equivalence(require_popgym_env) -> None:
    env_id = "popgym-RepeatPreviousEasy-v0"
    require_popgym_env(env_id)

    trainer = LSTMPPOTrainer.for_validation(env_id=env_id)
    trainer.collect_rollout()

    minibatches = list(trainer.buffer.get_recurrent_minibatches())
    assert len(minibatches) == 1
    batch = minibatches[0]

    K = trainer.state.cfg.trainer.tbptt_chunk_len

    # Full‑sequence state‑flow
    full_eval = trainer.policy.evaluate_actions_sequence(
        PolicyEvalInput(
            obs=batch.obs,
            hxs=batch.hxs[0],
            cxs=batch.cxs[0],
            actions=batch.actions,
        )
    )
    h_full = full_eval.new_hxs
    c_full = full_eval.new_cxs

    # TBPTT state‑flow stitched across chunks
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
        h_chunks.append(out.new_hxs)
        c_chunks.append(out.new_cxs)

    h_tb = torch.cat(h_chunks, dim=0)
    c_tb = torch.cat(c_chunks, dim=0)

    assert torch.allclose(h_tb, h_full, atol=1e-6)
    assert torch.allclose(c_tb, c_full, atol=1e-6)
