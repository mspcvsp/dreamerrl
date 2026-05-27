import numpy as np
import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from scipy.stats import entropy

from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.seed import set_global_seeds
from dreamerrl.utils.types import make_default_config


def run_training(seed, steps, progress, task_id):
    cfg = make_default_config()
    cfg.train.seed = seed
    cfg.train.enable_wandb = False
    cfg.train.cuda = True

    set_global_seeds(seed)
    trainer = DreamerTrainer(cfg)

    returns = []
    wm_losses = []
    actor_losses = []
    critic_losses = []
    action_logits = []

    for step in range(steps):
        progress.update(task_id, advance=1)

        trainer.collect_env_steps()
        batch = trainer.replay.sample(cfg.train.batch_size)

        wm_losses.append(trainer.update_world_model(batch, step))
        a_loss, c_loss = trainer.update_actor_critic(batch, step)
        actor_losses.append(a_loss)
        critic_losses.append(c_loss)

        with torch.no_grad():
            logits = trainer.actor(trainer.world_state.h, trainer.world_state.z)
            action_logits.append(logits.cpu())

        if trainer.env_state["is_last"].any():
            returns.append(trainer.env_state["reward"].sum().item())

    return {
        "returns": np.array(returns),
        "wm_loss": np.array(wm_losses),
        "actor_loss": np.array(actor_losses),
        "critic_loss": np.array(critic_losses),
        "action_logits": torch.stack(action_logits),
    }


def run_all_seeds(seeds, steps):
    results = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        # One task per seed
        seed_tasks = {
            seed: progress.add_task(f"Seed {i + 1}/{len(seeds)}", total=steps) for i, seed in enumerate(seeds)
        }

        for seed in seeds:
            results.append(run_training(seed, steps, progress, seed_tasks[seed]))

    return results


def kl_between_seeds(logits_a, logits_b):
    pa = torch.softmax(logits_a, dim=-1)
    pb = torch.softmax(logits_b, dim=-1)
    return float(entropy(pa.flatten(), pb.flatten()))


def summarize(metric_list):
    arr = np.stack(metric_list)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    cv = std.mean() / abs(mean.mean())
    return mean, std, cv


def main():
    seeds = [0, 1, 2, 3, 4]
    results = run_all_seeds(seeds, steps=2000)

    wm_mean, wm_std, wm_cv = summarize([r["wm_loss"] for r in results])
    actor_mean, actor_std, actor_cv = summarize([r["actor_loss"] for r in results])
    critic_mean, critic_std, critic_cv = summarize([r["critic_loss"] for r in results])

    print("World Model CV:", wm_cv)
    print("Actor CV:", actor_cv)
    print("Critic CV:", critic_cv)

    # KL between action distributions
    kl_vals = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            kl_vals.append(kl_between_seeds(results[i]["action_logits"], results[j]["action_logits"]))

    print("Action KL mean:", np.mean(kl_vals))

    # Regression criteria
    if wm_cv < 0.05 and actor_cv < 0.05 and critic_cv < 0.05 and np.mean(kl_vals) < 0.05:
        print("\n✅ Statistical reproducibility PASSED.")
    else:
        print("\n❌ Statistical reproducibility FAILED.")


if __name__ == "__main__":
    main()
