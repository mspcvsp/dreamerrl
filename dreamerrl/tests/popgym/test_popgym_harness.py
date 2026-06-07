import numpy as np
import torch
from scipy.stats import entropy

from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


def run_popgym_seed(env_id, seed, steps=5000):
    cfg = make_default_config()
    cfg.env.env_id = env_id
    cfg.env.num_envs = 4
    cfg.env.seed = seed
    cfg.train.seed = seed
    cfg.train.cuda = True
    cfg.log.enable_wandb = False

    trainer = DreamerTrainer(cfg)

    returns = []
    wm_losses = []
    actor_losses = []
    critic_losses = []
    action_logits = []

    for step in range(steps):
        trainer.collect_env_steps()
        batch = trainer.replay.sample(cfg.train.batch_size)

        wm_losses.append(trainer.update_world_model(batch, step))
        a_loss, c_loss = trainer.update_actor_critic(batch, step)
        actor_losses.append(a_loss)
        critic_losses.append(c_loss)

        if trainer.env_state["is_last"].any():
            returns.append(trainer.env_state["reward"].sum().item())

        if step >= steps - 50:
            with torch.no_grad():
                logits = trainer.actor(trainer.world_state.h, trainer.world_state.z)
                action_logits.append(logits)

    return {
        "returns": np.array(returns),
        "wm_loss": np.array(wm_losses),
        "actor_loss": np.array(actor_losses),
        "critic_loss": np.array(critic_losses),
        "action_logits": torch.stack(action_logits),
    }


def kl_between_seeds(logits_a, logits_b):
    pa = torch.softmax(logits_a, dim=-1)
    pb = torch.softmax(logits_b, dim=-1)
    pa_np = pa.view(-1)[::10].cpu().numpy()
    pb_np = pb.view(-1)[::10].cpu().numpy()
    return float(entropy(pa_np, pb_np))


def summarize(arr_list):
    arr = np.stack(arr_list)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    cv = std.mean() / abs(mean.mean())
    return mean, std, cv


def test_popgym(env_id="popgym-RepeatFirstEasy-v0", seeds=[0, 1, 2], steps=5000):
    results = [run_popgym_seed(env_id, s, steps) for s in seeds]

    wm_mean, wm_std, wm_cv = summarize([r["wm_loss"] for r in results])
    actor_mean, actor_std, actor_cv = summarize([r["actor_loss"] for r in results])
    critic_mean, critic_std, critic_cv = summarize([r["critic_loss"] for r in results])

    kl_vals = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            kl_vals.append(kl_between_seeds(results[i]["action_logits"], results[j]["action_logits"]))

    print("World Model CV:", wm_cv)
    print("Actor CV:", actor_cv)
    print("Critic CV:", critic_cv)
    print("Action KL mean:", np.mean(kl_vals))
    print("Mean Return:", np.mean([r["returns"].mean() for r in results]))

    assert wm_cv < 1e-3
    assert 0.5 < actor_cv < 3.0
    assert 0.05 < critic_cv < 1.0
    assert 0.1 < np.mean(kl_vals) < 2.0
    assert np.mean([r["returns"].mean() for r in results]) > 0.8
