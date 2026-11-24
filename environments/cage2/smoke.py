"""Fail-fast Spark CUDA backward check and one native 30-step CAGE2 Blue episode."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import inspect
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import dreamerrl
from dreamerrl.models.actor import Actor
from dreamerrl.utils.types import LatentConfig, NetworkConfig

PROJECT = Path(__file__).resolve().parent
CAGE_REVISION = "26ce1c1253fa9e2e73f25e6a7f2da32860c11257"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--learner-cycle", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    assert os.environ.get("CONDA_PREFIX") is None, "Deactivate Conda before this verification"
    assert Path(sys.executable).parent == PROJECT / ".venv" / "bin", sys.executable
    assert "miniconda" not in sys.base_prefix.lower()
    assert platform.machine() == "aarch64", "This lock targets the Spark's Linux aarch64 platform"
    vendor = PROJECT / "vendor" / "cage-challenge-2"
    revision = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    assert revision == CAGE_REVISION, "CAGE2 submodule differs from the pinned revision"
    subprocess.run(["git", "-C", str(vendor), "diff", "--exit-code"], check=True, capture_output=True)
    assert torch.__version__ == "2.9.1+cu130" and torch.version.cuda == "13.0"
    assert torch.cuda.is_available(), "A real GPU test is required; CPU fallback is forbidden"
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    actor = Actor(
        latent=LatentConfig(deter_size=8, num_classes=2, stoch_size=3), net=NetworkConfig(hidden_size=16, action_dim=4)
    ).to("cuda")
    h = torch.randn(4, 8, device="cuda", requires_grad=True)
    z = torch.nn.functional.one_hot(torch.randint(3, (4, 2), device="cuda"), 3).float()
    loss = actor(h, z).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    gradients = [p.grad for p in actor.parameters() if p.requires_grad]
    assert h.device.type == loss.device.type == "cuda"
    assert h.grad is not None and h.grad.device.type == "cuda" and torch.isfinite(h.grad).all()
    assert all(g is not None and g.device.type == "cuda" and torch.isfinite(g).all() for g in gradients)
    gradient_l1 = sum(float(g.abs().sum()) for g in gradients if g is not None)
    assert math.isfinite(float(loss.detach())) and gradient_l1 > 0

    if args.learner_cycle:
        if args.run_dir is None:
            parser.error("--learner-cycle requires --run-dir")
        from learner_cycle import main as learner_main

        return learner_main(["--run-dir", str(args.run_dir), "--seed", str(args.seed)])

    import CybORG as cyborg_package
    from CybORG import CybORG
    from CybORG.Agents import BlueMonitorAgent

    assert Path(cyborg_package.__file__).resolve().is_relative_to(vendor.resolve())
    scenario = Path(inspect.getfile(CybORG)).parent / "Shared" / "Scenarios" / "Scenario2.yaml"
    assert scenario.is_file(), "Use the bundled CAGE2 scenario, not a different CybORG generation"
    random.seed(args.seed)
    np.random.seed(args.seed)
    from dreamerrl.env.cage2.cage2_env_wrapper import make_blue_simulator

    env, actual_scenario = make_blue_simulator(args.seed)
    assert actual_scenario == scenario
    blue = BlueMonitorAgent()
    observation = env.reset(agent="Blue")
    blue.set_initial_values(observation=observation.observation, action_space=observation.action_space)
    rewards = []
    for step in range(30):
        action = blue.get_action(observation.observation, observation.action_space)
        observation = env.step(agent="Blue", action=action)
        assert not observation.has_error(), observation.error_msg
        assert observation.reward is not None and math.isfinite(float(observation.reward))
        rewards.append(float(observation.reward))
        assert not observation.done or step == 29, "Episode ended before the requested 30 steps"
    blue.end_episode()
    assert len(rewards) == 30
    report = dict(
        python_executable=sys.executable,
        python_version=platform.python_version(),
        architecture=platform.machine(),
        base_prefix=sys.base_prefix,
        conda_prefix=os.environ.get("CONDA_PREFIX"),
        dreamerrl_module=dreamerrl.__file__,
        cyborg_module=cyborg_package.__file__,
        cyborg_version=metadata.version("cyborg"),
        cage2_revision=revision,
        scenario=str(scenario),
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0),
        gpu_capability=list(torch.cuda.get_device_capability(0)),
        tensor_device=str(h.device),
        loss=float(loss.detach()),
        gradient_l1=gradient_l1,
        finite_cuda_gradients=True,
        blue_agent="BlueMonitorAgent",
        red_agent="B_lineAgent",
        seed=args.seed,
        steps=len(rewards),
        total_blue_return=sum(rewards),
        rewards=rewards,
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
