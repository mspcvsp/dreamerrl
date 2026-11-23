"""Replay known R2 diagnostic routes and compare paired one-step predictions."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.wrappers import TimeLimit

from dreamerrl.eval.corridor_diagnostics import CORRIDOR_R2, compass_actions, stair_action, winning_action_stats
from dreamerrl.eval.location_probe import load_frozen_agent
from dreamerrl.training.run_monitor import write_json


def route_prefix(env, seed, actions, moves):
    env.unwrapped.seed(core=seed, disp=seed, reseed=False, lgen=seed)
    obs, _ = env.reset(seed=seed)
    glyphs = [obs["glyphs"].ravel().copy()]
    for action in actions:
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise ValueError(f"Route {seed} ended before its specified winning step")
        glyphs.append(obs["glyphs"].ravel().copy())
    return obs, glyphs, stair_action(obs, moves)


def verify_route(env, route, moves):
    seed = int(route["seed"])
    *prefix, win = [int(step["action_id"]) for step in route["trace"]]
    obs, glyphs, stair = route_prefix(env, seed, prefix, moves)
    origin = tuple(map(int, obs["blstats"][:2]))
    if stair != win:
        raise ValueError(f"Route {seed}, step {len(prefix) + 1}: expected stair action {win}, found {stair}")
    terminal, reward, terminated, truncated, info = env.step(win)
    if reward != 1 or not terminated or info["end_status"].name != "TASK_SUCCESSFUL":
        raise ValueError(f"Route {seed} did not terminate with the +1 stair reward")
    terminal_glyphs = terminal["glyphs"].ravel().copy()
    dx, dy = moves[win]
    reverse = next(action for action, offset in moves.items() if offset == (-dx, -dy))
    for alternative in dict.fromkeys([reverse, *moves]):
        dx, dy = moves[alternative]
        if alternative == win or (dx and dy):
            continue
        _, alternate_prefix, _ = route_prefix(env, seed, prefix, moves)
        np.testing.assert_array_equal(alternate_prefix, glyphs)
        obs, reward, done, timeout, _ = env.step(alternative)
        destination = tuple(map(int, obs["blstats"][:2]))
        if reward == 0 and not done and not timeout and destination == (origin[0] + dx, origin[1] + dy):
            return (
                {
                    "seed": seed,
                    "step": len(prefix) + 1,
                    "origin": origin,
                    "winning_action": win,
                    "alternative_action": alternative,
                    "outcome": "TASK_SUCCESSFUL",
                    "terminal_reward": 1.0,
                    "terminal_truncated": bool(truncated),
                    "alternative_reward": 0.0,
                },
                [*glyphs, terminal_glyphs],
                prefix,
            )
    raise ValueError(f"Route {seed} has no verified zero-reward alternative")


@torch.no_grad()
def probe_route(agent, route, glyphs, prefix, draws, sampling_seed):
    world, actor, critic = agent.world, agent.actor, agent.critic
    device = next(world.parameters()).device
    torch.manual_seed(sampling_seed + route["seed"])
    observations = torch.as_tensor(np.stack(glyphs), dtype=torch.float32, device=device)
    horizon = agent.cfg.env.max_episode_steps
    if agent.cfg.env.finite_horizon:
        clock = (1 - torch.arange(len(glyphs), device=device) / horizon).unsqueeze(-1)
        observations = torch.cat([observations, clock], dim=-1)
    embeds = world.encoder(observations)
    action_dim = world.net_cfg.action_dim
    ids = [0, *prefix, route["winning_action"], route["alternative_action"]]
    actions = F.one_hot(torch.tensor(ids, device=device), action_dim).float()
    state = world.init_state(draws)
    first = torch.ones(draws, device=device, dtype=torch.bool)
    for embed, action in zip(embeds[:-1], actions[:-2]):
        state = world.filter_step(state, embed.expand(draws, -1), action.expand(draws, -1), first)
        first = torch.zeros_like(first)
    probability, rank = winning_action_stats(actor(state.h, state.z), route["winning_action"])
    win_action, alternative_action = (action.expand(draws, -1) for action in actions[-2:])
    rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
    win = world.imagine_step(state, actor, action=win_action)
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng, device)
    else:
        torch.set_rng_state(rng)
    alternative = world.imagine_step(state, actor, action=alternative_action)
    observed = world.filter_step(state, embeds[-1].expand(draws, -1), win_action)

    head = world.reward_heads.main
    posterior_reward = head.readout(head(observed.h, observed.z))
    win_reward = head.readout(head(win.h, win.z))
    alternative_reward = head.readout(head(alternative.h, alternative.z))
    report = {
        **route,
        "actor_winning_action_probability": probability.mean().item(),
        "actor_winning_action_rank": rank.float().mean().item(),
        "posterior_terminal_reward": posterior_reward.mean().item(),
        "prior_winning_reward": win_reward.mean().item(),
        "prior_alternative_reward": alternative_reward.mean().item(),
        "paired_prior_reward_gap": (win_reward - alternative_reward).mean().item(),
    }
    if critic is not None:
        gamma = agent.cfg.ac.discount
        within_horizon = not agent.cfg.env.finite_horizon or route["step"] < horizon

        def one_step(successor, reward):
            cont_head = world.continue_head
            continuation = cont_head.readout(cont_head(successor.h, successor.z))
            value = critic.readout(critic(successor.h, successor.z))
            return {
                "reward": reward.mean().item(),
                "continuation": continuation.mean().item(),
                "successor_value": value.mean().item(),
                "return": (reward + gamma * within_horizon * continuation * value).mean().item(),
            }

        report["one_step_critic_diagnostic"] = {
            "discount": gamma,
            "return_aggregation": "mean of per-draw one-step returns",
            "finite_horizon_bootstrap_allowed": within_horizon,
            "winning": one_step(win, win_reward),
            "alternative": one_step(alternative, alternative_reward),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--routes", type=Path, required=True, help="Scripted route JSON containing seed and trace per episode"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--sampling-seed", type=int, default=4997)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--critic", action="store_true", help="Include the online critic one-step diagnostic")
    args = parser.parse_args()
    if args.draws < 1 or args.sampling_seed < 0:
        parser.error("Use a positive draw count and a nonnegative sampling seed")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    routes = json.loads(args.routes.read_text())["episodes"]
    if not routes or len({route["seed"] for route in routes}) != len(routes):
        parser.error("Provide nonempty diagnostic routes with unique seeds")
    torch.set_num_threads(2)
    agent = load_frozen_agent(args.checkpoint, sample_latents=True, load_critic=args.critic)
    if agent.cfg.env.env_id != CORRIDOR_R2:
        parser.error("The probe requires a MiniHack Corridor R2 checkpoint")
    agent.world.to(args.device)
    agent.actor.to(args.device)
    if agent.critic is not None:
        agent.critic.to(args.device)
    agent.world.prior.deterministic_latent_for_tests = False
    results = []
    with TimeLimit(gym.make(CORRIDOR_R2), max_episode_steps=agent.cfg.env.max_episode_steps) as env:
        moves = compass_actions(getattr(env.unwrapped, "actions"))
        for route in routes:
            verified, glyphs, prefix = verify_route(env, route, moves)
            results.append(probe_route(agent, verified, glyphs, prefix, args.draws, args.sampling_seed))
    write_json(
        args.output,
        {
            "checkpoint": agent.checkpoint,
            "update": agent.update,
            "probe_seeds": [route["seed"] for route in results],
            "draws_per_route": args.draws,
            "sampling_seed": args.sampling_seed,
            "critic": args.critic,
            "routes": results,
        },
    )


if __name__ == "__main__":
    main()
