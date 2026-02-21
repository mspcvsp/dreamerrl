import torch

from dreamerrl.trainer import LSTMPPOTrainer


def test_tbptt_minibatch_preserves_sequence_order(require_popgym_env):
    env_id = "popgym-RepeatPreviousEasy-v0"
    require_popgym_env(env_id)

    trainer = LSTMPPOTrainer.for_validation(env_id=env_id)

    # Fill rollout buffer
    trainer.collect_rollout()

    # Validation mode → exactly one minibatch
    minibatches = list(trainer.buffer.get_recurrent_minibatches())
    assert len(minibatches) == 1
    batch = minibatches[0]

    K = trainer.state.cfg.trainer.tbptt_chunk_len

    # For each TBPTT chunk, verify obs/actions/etc. are contiguous slices
    for mb in batch.iter_chunks(K):
        t0, t1 = mb.t0, mb.t1

        # Sequence order must be preserved
        assert torch.allclose(mb.obs, batch.obs[t0:t1])
        assert torch.allclose(mb.actions, batch.actions[t0:t1])
        assert torch.allclose(mb.returns, batch.returns[t0:t1])
        assert torch.allclose(mb.advantages, batch.advantages[t0:t1])
