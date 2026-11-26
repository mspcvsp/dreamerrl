"""Archived Blue decision frames and separately keyed offline simulator labels."""
from dataclasses import replace
import json
import random

import numpy as np
import torch

from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv, boundary_flags
from dreamerrl.eval.memory.dreamer import file_sha256
from dreamerrl.eval.memory.interfaces import Episode, Labels


def active_host_labels(state, host_order):
    if not set(host_order) <= state.hosts.keys():
        raise ValueError("Simulator hosts disagree with the Blue schema")
    active = {session.host for session in state.sessions["Red"].values() if session.active}
    return np.asarray([host in active for host in host_order], dtype=bool)


class CAGE2Replay:
    """Replay archived Blue actions. Truth never enters the policy frame stream."""

    def __init__(self, cfg, rows, scenario_hash, expected_actions):
        self.cfg, self.rows = cfg, rows
        self.scenario_hash, self.expected_actions = scenario_hash, expected_actions
        self.by_episode = {}

    def model_input(self, observations):
        return observations.float()

    def labels(self, episode_id):
        values = self.by_episode[episode_id]
        return Labels(values, np.ones_like(values, dtype=bool))

    def _check_env(self, env):
        cfg = self.cfg
        if (env.obs_dim, env.action_dim, env.host_order, env.subnet_ids) != (
            52, 145, tuple(cfg.world.cage2_host_order), tuple(cfg.world.cage2_subnet_ids)
        ) or env.action_mapping != self.expected_actions or file_sha256(env.scenario) != self.scenario_hash:
            raise ValueError("Checkpoint, scenario or action mapping mismatch")

    def collect(self, episode_ids):
        result = []
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                for seed in episode_ids:
                    row = self.rows[seed]
                    if row["policy"] != "actor" or row["policy_seed"] != seed + 100_000 or row["steps"] != 30:
                        raise ValueError("Unexpected historical evaluation settings")
                    cfg = replace(self.cfg.env, seed=seed)
                    env = Cage2BlueEnv(cfg, torch.device("cpu"))
                    try:
                        first = env.reset(seed=seed)["state"][0]
                        self._check_env(env)
                        native = env.wrapper
                        controller = native.get_attr("environment_controller")
                        observations, labels, rewards = [], [], []
                        actions = [action["index"] for action in row["actions"]]
                        for t, action in enumerate(actions):
                            if env.action_mapping[action] != row["actions"][t]:
                                raise ValueError("Archived action mapping differs")
                            observations.append(first.clone())
                            labels.append(active_host_labels(controller.state, env.host_order))
                            following, reward, done, info = native.step(action)
                            if info.get("error") is not None:
                                raise ValueError("Simulator reported an error during Blue replay")
                            last, terminated, truncated = boundary_flags(done, info, t+1, cfg.max_episode_steps)
                            if (last, terminated, truncated) != (t == 29, False, t == 29):
                                raise ValueError("Replay boundary differs from historical horizon")
                            if row["native_reward_components"][t] is not None:
                                components = json.loads(json.dumps(native.get_reward_breakdown("Blue"), default=str))
                                if components != row["native_reward_components"][t]:
                                    raise ValueError("Native reward components differ during replay")
                            rewards.append(reward)
                            first = torch.as_tensor(np.array(following, copy=True), dtype=torch.float32)
                        np.testing.assert_allclose(rewards, row["rewards"], rtol=0, atol=1e-12)
                        np.testing.assert_allclose(sum(rewards), row["total_return"], rtol=0, atol=1e-12)
                        self.by_episode[seed] = np.stack(labels)
                        result.append(Episode(seed, seed, torch.stack(observations), torch.tensor(actions),
                                              torch.tensor(rewards), terminated, truncated))
                    finally:
                        env.close()
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        return result
