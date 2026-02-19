from lstmppo.trainer import LSTMPPOTrainer


def test_learning_smoke(require_popgym_env) -> None:
    """
    Minimal learning smoke test:
    - ensures rollout + PPO update run without error
    - ensures reward trends upward across a few iterations
    """

    env_id = "popgym-RepeatPreviousEasy-v0"
    require_popgym_env(env_id)

    trainer = LSTMPPOTrainer.for_validation(env_id=env_id)

    rewards = []

    for _ in range(3):
        trainer.collect_rollout()
        trainer.optimize_policy()

        # Compute mean episode reward from rollout buffer
        # rewards: (T, B)
        r = trainer.buffer.rewards

        # Sum rewards until termination for each env
        # For a smoke test, a simple mean over all rewards is fine
        mean_reward = float(r.mean().item())
        rewards.append(mean_reward)

    # Should trend upward even in a tiny smoke test
    assert rewards[-1] >= rewards[0]
