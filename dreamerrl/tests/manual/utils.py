"""Small configurations and held-out evaluation for the MiniHack experiments."""

from __future__ import annotations

from dreamerrl.models.aux_objectives import AUX_OBJECTIVES
from dreamerrl.utils.types import make_default_config


def minihack_config(seed, logdir, novelty=False):
    cfg = make_default_config()
    cfg.env.env_id = "MiniHack-Room-5x5-v0"
    cfg.env.num_envs = 2
    cfg.env.max_episode_steps = 6
    cfg.env.parallel = False
    cfg.env.seed = seed
    cfg.world.disable_aux_losses = not novelty
    cfg.world.aux_objectives = [AUX_OBJECTIVES["novelty"]] if novelty else []
    cfg.world.deter_size = 16
    cfg.world.num_classes = 3
    cfg.world.stoch_size = 4
    cfg.world.hidden_size = 16
    cfg.world.imagination_horizon = 3
    cfg.ac.actor_hidden = cfg.ac.critic_hidden = 16
    cfg.train.seed = seed
    cfg.train.cuda = False
    cfg.train.use_amp = False
    cfg.train.enable_fused_adamw_kernels = False
    cfg.train.deterministic_env = False
    cfg.train.deterministic_imagination = False
    cfg.train.freeze_actor_critic_steps = 0
    cfg.train.random_exploration_steps = 0
    cfg.train.batch_size = 2
    cfg.train.collect_steps = 12
    cfg.train.seq_len = 4
    cfg.log.enable_wandb = False
    cfg.log.tb_logdir = str(logdir)
    cfg.log.run_name = f"seed{seed}_{'novelty' if novelty else 'baseline'}"
    return cfg
