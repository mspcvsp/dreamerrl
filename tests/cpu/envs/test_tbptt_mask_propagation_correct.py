import torch

from lstmppo.trainer import LSTMPPOTrainer


def test_tbptt_mask_propagation_correct(require_popgym_env) -> None:
    env_id = "popgym-RepeatPreviousEasy-v0"
    require_popgym_env(env_id)

    trainer = LSTMPPOTrainer.for_validation(env_id=env_id)
    trainer.collect_rollout()

    minibatches = list(trainer.buffer.get_recurrent_minibatches())
    assert len(minibatches) == 1
    batch = minibatches[0]

    # Full-sequence mask (T, B)
    batch_mask = 1.0 - (batch.terminated | batch.truncated).float()

    K = trainer.state.cfg.trainer.tbptt_chunk_len

    for mb in batch.iter_chunks(K):
        t0, t1 = mb.t0, mb.t1

        # Chunk mask (T_chunk, B)
        mb_mask = mb.mask

        # Invariant: chunk mask must be an exact slice of the full mask
        assert torch.allclose(mb_mask, batch_mask[t0:t1])
