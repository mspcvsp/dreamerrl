import torch

from dreamerrl.trainer import LSTMPPOTrainer


def test_tbptt_chunk_boundaries_correct(require_popgym_env):
    env_id = "popgym-RepeatPreviousEasy-v0"
    require_popgym_env(env_id)

    trainer = LSTMPPOTrainer.for_validation(env_id=env_id)

    # Fill the rollout buffer
    trainer.collect_rollout()

    # Get the single minibatch (validation mode = 1 env, 1 minibatch)
    minibatches = list(trainer.buffer.get_recurrent_minibatches())
    assert len(minibatches) == 1
    batch = minibatches[0]

    K = trainer.state.cfg.trainer.tbptt_chunk_len

    for mb in batch.iter_chunks(K):
        t0 = mb.t0
        # PRE‑STEP hidden state must match rollout buffer
        assert torch.allclose(mb.hxs0, batch.hxs[t0])
        assert torch.allclose(mb.cxs0, batch.cxs[t0])
