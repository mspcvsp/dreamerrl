import pytest
import torch

from dreamerrl.models.actor import Actor
from dreamerrl.models.value_head import ValueHead
from dreamerrl.training.replay_buffer import DreamerReplayBuffer
from dreamerrl.training.test_trainer import TestDreamerTrainer, lambda_return


@pytest.mark.trainer
def test_actor_critic_training_step(world_model, device):
    B = 1
    obs_dim = world_model.flat_obs_dim
    deter = world_model.deter_size
    stoch = world_model.stoch_size
    action_dim = 4

    buffer = DreamerReplayBuffer(
        num_envs=B,
        obs_dim=obs_dim,
        capacity_episodes=10,
        device=device,
    )

    # Build a small episode
    for t in range(6):
        buffer.add(
            state=torch.randn(obs_dim, device=device),
            action=torch.randint(0, action_dim, (1,), device=device),
            reward=torch.randn((), device=device),
            is_first=torch.tensor(t == 0, device=device),
            is_last=torch.tensor(t == 5, device=device),
            is_terminal=torch.tensor(False, device=device),
        )

    actor = Actor(deter, stoch, hidden_size=128, action_dim=action_dim).to(device)
    critic = ValueHead(deter, stoch, hidden_size=128).to(device)

    # Instantiate to assert the trainer wiring is valid; the test computes losses manually.
    TestDreamerTrainer(
        world_model=world_model,
        actor=actor,
        critic=critic,
        replay_buffer=buffer,
        device=device,
    )

    batch = buffer.sample(batch_size=2, seq_len=3, device=device)

    # Inline batch size to avoid unused variable
    state = world_model.init_state(batch["state"].size(0))
    horizon = 5

    imagined = []
    for _ in range(horizon):
        state = world_model.imagine_step(state)
        imagined.append(state)

    rewards = torch.stack([world_model.predict_reward(s) for s in imagined], dim=0)  # (T, B, 1)
    values = torch.stack([critic(s.h, s.z) for s in imagined], dim=0)  # (T, B, 1)

    logits = torch.stack([actor(s.h, s.z) for s in imagined], dim=0)  # (T, B, A)
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    logp = dist.log_prob(actions)  # (T, B)

    # Dummy λ-return target
    returns = lambda_return(
        reward=rewards.squeeze(-1).transpose(0, 1),  # (B, T)
        value=values.squeeze(-1).transpose(0, 1),  # (B, T)
        discount=0.99,
        lam=0.95,
    ).transpose(0, 1)  # back to (T, B)

    actor_loss = -(logp * returns.detach()).mean()
    critic_loss = (values.squeeze(-1) - returns.detach()).pow(2).mean()

    assert torch.isfinite(actor_loss)
    assert torch.isfinite(critic_loss)
    assert actor_loss.dim() == 0
    assert critic_loss.dim() == 0
