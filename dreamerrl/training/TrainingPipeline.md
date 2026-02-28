                   ┌──────────────────────────────┐
                   │         Environment           │
                   │      (PopGymVecEnv)           │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │  collect_env_steps()   │
                     │  - sample action       │
                     │  - env.step()          │
                     │  - world.observe_step  │
                     │  - replay.add_batch    │
                     └──────────────┬─────────┘
                                    │
                                    ▼
                     ┌────────────────────────┐
                     │   Replay Buffer        │
                     │ DreamerReplayBuffer    │
                     └──────────────┬─────────┘
                                    │ sample batch
                                    ▼
                     ┌────────────────────────┐
                     │ world_model_training   │
                     │  world_model_update()  │
                     │    (core/)             │
                     └──────────────┬─────────┘
                                    │
                                    ▼
                     ┌────────────────────────┐
                     │ actor_critic_update()  │
                     │    (core/)             │
                     │ - imagine latent traj  │
                     │ - predict rewards      │
                     │ - predict values       │
                     │ - bootstrap V          │
                     │ - λ-return             │
                     │ - actor loss           │
                     │ - critic loss          │
                     └──────────────┬─────────┘
                                    │
                                    ▼
                     ┌────────────────────────┐
                     │   Optimizers (Adam)    │
                     │   - world model        │
                     │   - actor              │
                     │   - critic             │
                     └────────────────────────┘
