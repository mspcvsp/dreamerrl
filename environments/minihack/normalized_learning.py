"""Training-only calibration and provenance for the frozen-RSSM experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dreamerrl.eval.location_probe import load_frozen_agent
from dreamerrl.eval.posterior_diagnostics import collect_trace, filter_trace
from dreamerrl.models.goal_autoencoder import fit_goal_scale, goal_features
from host_experiment_support import git, sha256

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("director_room5x5.json")


def main_root():
    return Path(git("rev-parse", "--path-format=absolute", "--git-common-dir", root=ROOT)).parent


def world_fingerprint(weights):
    checksum = hashlib.sha256()
    for name, value in sorted(weights.items()):
        checksum.update(name.encode())
        checksum.update(value.detach().cpu().numpy().tobytes())
    return checksum.hexdigest()


def source_path(spec):
    path = main_root() / spec["source_checkpoint"]
    if sha256(path) != spec["source_checkpoint_sha256"]:
        raise ValueError("Pretrained checkpoint fingerprint mismatch")
    return path


def calibration_path(spec):
    path = main_root() / spec["normalizer"]
    if sha256(path) != spec["normalizer_sha256"]:
        raise ValueError("Frozen calibration fingerprint mismatch")
    return path


def calibration_arrays(spec):
    with np.load(calibration_path(spec), allow_pickle=False) as data:
        if data["source_world"].item() != spec["source_world_sha256"]:
            raise ValueError("Calibration was fitted with a different world model")
        expected = np.arange(spec["calibration_seed_start"], spec["calibration_seed_start"] + spec["calibration_levels"])
        if not np.array_equal(data["seeds"], expected):
            raise ValueError("Calibration seed mismatch")
        return torch.from_numpy(data["center"]), torch.from_numpy(data["scale"])


def worker_inputs(spec):
    source = torch.load(source_path(spec), weights_only=False, map_location="cpu")
    if world_fingerprint(source["world"]) != spec["source_world_sha256"]:
        raise ValueError("World tensor fingerprint mismatch")
    # Replay, optimizer states and trained policies are never transferred.
    return {k: source[k] for k in ("world", "config")}, calibration_arrays(spec)


@torch.no_grad()
def calibrate(spec, output):
    agent = load_frozen_agent(source_path(spec), allow_hierarchical=True)
    if agent.cfg.env.env_id != spec["environment"] or agent.update != spec["source_checkpoint_update"]:
        raise ValueError("Unexpected calibration environment or checkpoint")
    seeds = list(range(spec["calibration_seed_start"], spec["calibration_seed_start"] + spec["calibration_levels"]))
    monitors = set(range(spec["eval_seed_start"], spec["eval_seed_start"] + spec["eval_episodes"]))
    heldout = set(range(spec["diagnostic_seed_start"], spec["diagnostic_seed_start"] + spec["diagnostic_levels"]))
    if set(seeds) & (monitors | heldout):
        raise ValueError("Calibration includes evaluation levels")
    torch.set_num_threads(2)
    trace = collect_trace(agent, seeds, max_steps=spec["episode_steps"], sampling_seed=spec["calibration_rng_seed"])
    states = filter_trace(agent, trace, sampled=True, seed=spec["calibration_rng_seed"])
    features = goal_features(states["h"], states["z"])[~trace["is_last"]]
    center, scale = fit_goal_scale(features, spec["normalization_floor"])
    world_hash = world_fingerprint(agent.world.state_dict())
    with output.open("xb") as stream:
        np.savez(stream, center=center.numpy(), scale=scale.numpy(), seeds=np.array(seeds),
                 samples=len(features), source_world=world_hash)
    print(json.dumps(dict(samples=len(features), source_world_sha256=world_hash, normalizer_sha256=sha256(output),
                          seed_start=seeds[0], seed_end=seeds[-1])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    calibrate(json.loads(CONFIG.read_text()), args.output)
