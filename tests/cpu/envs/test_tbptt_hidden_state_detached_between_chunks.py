from dreamerrl.trainer import LSTMPPOTrainer
from dreamerrl.types import PolicyEvalInput


def test_tbptt_hidden_state_detached_between_chunks(require_popgym_env):
    env_id = "popgym-RepeatPreviousEasy-v0"
    require_popgym_env(env_id)

    trainer = LSTMPPOTrainer.for_validation(env_id=env_id)
    trainer.collect_rollout()

    minibatches = list(trainer.buffer.get_recurrent_minibatches())
    assert len(minibatches) == 1
    batch = minibatches[0]

    K = trainer.state.cfg.trainer.tbptt_chunk_len

    for mb in batch.iter_chunks(K):
        # --- Invariant 1: hidden state at chunk start must be detached ---
        assert mb.hxs0.requires_grad is False
        assert mb.cxs0.requires_grad is False

        # --- Invariant 2: rollout data must NOT require grad ---
        assert mb.obs.requires_grad is False
        assert mb.actions.requires_grad is False
        assert mb.advantages.requires_grad is False
        assert mb.returns.requires_grad is False

        # --- Invariant 3: evaluate_actions_sequence must NOT create grad-enabled hidden states ---
        eval_out = trainer.policy.evaluate_actions_sequence(
            PolicyEvalInput(
                obs=mb.obs,
                hxs=mb.hxs0,
                cxs=mb.cxs0,
                actions=mb.actions,
            )
        )

        assert eval_out.new_hxs.requires_grad is False
        assert eval_out.new_cxs.requires_grad is False
