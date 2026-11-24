"""Serial Challenge-2 Blue adapter; native observations/actions and unshaped rewards."""

import inspect
import random
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from gym.spaces import Box as LegacyBox
from gym.spaces import Discrete
from gymnasium.spaces import Box

from dreamerrl.env.env import EnvInterface


class ActionMapping(TypedDict):
    index: int
    name: str
    description: str


def make_blue_simulator(seed):
    """Use the CAGE2 package's own Scenario2 and supplied B-line opponent."""
    from CybORG import CybORG
    from CybORG.Agents import B_lineAgent

    random.seed(seed)
    np.random.seed(seed)
    scenario = Path(inspect.getfile(CybORG)).parent / "Shared" / "Scenarios" / "Scenario2.yaml"
    simulator = CybORG(str(scenario), "sim", agents={"Red": B_lineAgent})
    simulator.set_seed(seed)
    return simulator, scenario


def make_blue_wrapper(seed, horizon):
    from CybORG.Agents.Wrappers import ChallengeWrapper

    simulator, scenario = make_blue_simulator(seed)
    return ChallengeWrapper(agent_name="Blue", env=simulator, max_steps=horizon), scenario


def boundary_flags(done, info, step, horizon):
    """Pinned wrapper preserves native done in info, before imposing max_steps.

    If an older wrapper omits it, done at the cap is assumed a truncation;
    done before the cap is terminal. The assumption is explicit in episode info.
    """
    native = bool(info.get("done", done and step < horizon))
    return bool(done), native, bool(done and not native)


class Cage2BlueEnv(EnvInterface):
    """One Blue collector with same-step reset and a separate final observation."""

    def __init__(self, env_cfg, device, probe=None):
        if env_cfg.num_envs != 1 or env_cfg.parallel:
            raise ValueError("The first CAGE2 integration supports one serial Blue environment")
        self.cfg, self.device, self.probe = env_cfg, device, probe
        self.wrapper, self.scenario = make_blue_wrapper(env_cfg.seed, env_cfg.max_episode_steps)
        initial = np.asarray(self.wrapper.reset())
        self.native_dtype = str(initial.dtype)
        self._obs_dim = initial.size
        assert isinstance(self.wrapper.action_space, Discrete)
        assert isinstance(self.wrapper.observation_space, LegacyBox)
        self._action_dim = int(self.wrapper.action_space.n)
        self._obs_space = Box(
            np.asarray(self.wrapper.observation_space.low, dtype=np.float32),
            np.asarray(self.wrapper.observation_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_mapping: list[ActionMapping] = [
            ActionMapping(index=i, name=type(action).__name__, description=str(action))
            for i, action in enumerate(self.wrapper.get_attr("possible_actions"))
        ]
        assert len(self.action_mapping) == self._action_dim
        self._next_seed = env_cfg.seed
        self.journal: list[dict[str, Any]] = []
        self.venv = self  # Existing trainer resource-cleanup contract.

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def batch_size(self):
        return 1

    def _reset_episode(self):
        self.episode_seed = self._next_seed
        self._next_seed += 1
        self.wrapper, _ = make_blue_wrapper(self.episode_seed, self.cfg.max_episode_steps)
        obs = np.asarray(self.wrapper.reset())
        assert isinstance(self.wrapper.action_space, Discrete)
        assert obs.shape == self._obs_space.shape and int(self.wrapper.action_space.n) == self.action_dim
        self.steps = 0
        self.current = torch.as_tensor(obs.copy(), dtype=torch.float32, device=self.device).reshape(1, -1)
        return self.current.clone()

    def reset(self, seed=None) -> dict[str, Any]:
        if seed is not None:
            self._next_seed = seed
        self.journal = []
        state = self._reset_episode()
        return dict(
            state=state,
            reward=torch.zeros(1, device=self.device),
            is_first=torch.ones(1, dtype=torch.bool, device=self.device),
            is_last=torch.zeros(1, dtype=torch.bool, device=self.device),
            is_terminal=torch.zeros(1, dtype=torch.bool, device=self.device),
            prev_action=torch.zeros(1, self.action_dim, device=self.device),
            info={},
        )

    def step(self, actions) -> dict[str, Any]:
        action = int(actions.reshape(-1)[0].item())
        if not 0 <= action < self.action_dim:
            raise ValueError("Action outside installed ChallengeWrapper action space")
        before = self.current.clone()
        obs, reward, done, info = self.wrapper.step(action)
        if info.get("error") is not None:
            raise RuntimeError(info.get("error_msg", str(info["error"])))
        self.steps += 1
        last, terminal, truncated = boundary_flags(done, info, self.steps, self.cfg.max_episode_steps)
        transition = torch.as_tensor(np.asarray(obs).copy(), dtype=torch.float32, device=self.device).reshape(1, -1)
        self.journal.append(
            dict(
                seed=self.episode_seed,
                step=self.steps,
                action=action,
                action_description=self.action_mapping[action]["description"],
                observation_before=before[0].cpu().tolist(),
                observation_after=transition[0].cpu().tolist(),
                reward=float(reward),
                wrapper_done=last,
                native_done=terminal,
                truncated=truncated,
            )
        )
        state = self._reset_episode() if last else transition.clone()
        self.current = state
        previous = torch.nn.functional.one_hot(torch.tensor([action], device=self.device), self.action_dim).float()
        if last:
            previous.zero_()
        return dict(
            state=state,
            transition_state=transition,
            reward=torch.tensor([reward], dtype=torch.float32, device=self.device),
            is_first=torch.tensor([last], device=self.device),
            is_last=torch.tensor([last], device=self.device),
            is_terminal=torch.tensor([terminal], device=self.device),
            prev_action=previous,
            info={
                **info,
                "adapter_truncated": truncated,
                "horizon_assumption": "native done retained in info; otherwise cap means truncation",
            },
        )

    def close(self):
        """The pure simulator holds no external process or socket resources."""
