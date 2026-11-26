"""Extend the archived CAGE2 frozen evaluation with the three flat replications."""

import argparse
import importlib.metadata
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch

from frozen_opponents import ROOT, VENDOR, git, load_policy, run_cell, sha256, write

ARCHIVE = ROOT / "runs/cage2_frozen_opponents_20261125T213227Z"
FLAT = ROOT / "runs/cage2_flat_replication_20261126T033743Z/group"
HIERARCHY = ROOT / "runs/hierarchy_v3_20261125T153127Z/group"
TAGS = {"flat": "cage2-host-aware-flat-replication-v1.1", "hierarchical": "dreamerv3-hierarchy-experiment-v3"}


def read(path):
    return json.loads(path.read_text())


def metrics(values):
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values), n=len(values))


def cohort(spec):
    items = []
    for arm, group in (("flat", FLAT), ("hierarchical", HIERARCHY)):
        report = read(group / "final_report.json")
        for record in report["runs"]:
            job = record["job"]
            run = Path(job["run"])
            seed = job["seed"]
            manifest = read(run / "manifest.json")
            path = run / "latest.pt"
            if (
                job["state"] != "completed" or job["exit_code"] != 0
                or job["updates"] != spec["completed_updates"] or job["env_steps"] != spec["env_steps"]
                or manifest["source_tag"] != TAGS[arm]
                or git("rev-parse", TAGS[arm] + "^{}") != manifest["source_commit"]
                or git("rev-parse", TAGS[arm]) != manifest["tag_object"]
                or sha256(path) != record["checkpoint_sha256"]
            ):
                raise ValueError(f"Checkpoint identity or completion mismatch: {path}")
            items.append(dict(id=f"{arm}_seed{seed}", arm=arm, training_seed=seed,
                              path=str(path.relative_to(ROOT)), sha256=record["checkpoint_sha256"],
                              source=manifest["source_commit"], version=4 if arm == "flat" else 5))
        if sorted(i["training_seed"] for i in items if i["arm"] == arm) != [0, 1, 2]:
            raise ValueError(f"Incomplete seed cohort: {arm}")
    return items


def validate_log(path, archive, checkpoint, opponent, seeds, policy_seeds):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if [row["seed"] for row in rows] != seeds or [row["policy_seed"] for row in rows] != policy_seeds:
        raise ValueError(f"Seed mismatch: {path}")
    totals = []
    for row in rows:
        total = math.fsum(row["rewards"])
        if (row["steps"] != 30 or len(row["rewards"]) != 30 or row["policy"] != "actor"
                or not math.isclose(total, row["total_return"], abs_tol=1e-9)):
            raise ValueError(f"Incompatible episode: {path}")
        totals.append(total)
    stat = metrics(totals)
    if archive and (
        not math.isclose(archive["mean_return"], stat["mean"], abs_tol=1e-9)
        or not math.isclose(archive["episode_sd"], stat["sample_sd"], abs_tol=1e-9)
        or archive["episodes"] != stat["n"]
    ):
        raise ValueError(f"Archived summary mismatch: {path}")
    return dict(checkpoint=checkpoint, opponent=opponent, mean_return=stat["mean"],
                episode_sd=stat["sample_sd"], episodes=stat["n"], log=str(path.relative_to(ROOT)),
                log_sha256=sha256(path), archived=archive is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != ROOT / "runs":
        parser.error("Use a fresh directory under the main runs directory")
    output.mkdir(exist_ok=False)
    prov = read(ARCHIVE / "provenance.json")
    spec = prov["manifest"]
    seeds, policy_seeds = prov["seeds"], prov["policy_seeds"]
    import CybORG

    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    if (
        spec["horizon"] != 30 or len(seeds) != spec["episodes_per_cell"]
        or seeds != list(range(spec["first_seed"], spec["first_seed"] + spec["episodes_per_cell"]))
        or policy_seeds != [seed + 100_000 for seed in seeds]
        or prov["policy"] != "greedy, deterministic posterior modes" or prov["device"] != "cpu"
        or prov["python"] != sys.version or packages != prov["packages"]
        or prov["pythonhashseed"] != os.environ.get("PYTHONHASHSEED")
        or prov["lock_sha256"] != sha256(ROOT / "uv.lock")
        or prov["cyborg_revision"] != git("rev-parse", "HEAD", root=VENDOR)
        or not Path(CybORG.__file__).resolve().is_relative_to(VENDOR)
        or git("status", "--porcelain", "--untracked-files=no", root=VENDOR)
        or (ARCHIVE / "exit_status").read_text().strip() != "0"
    ):
        raise ValueError("Frozen evaluation runtime or archive provenance mismatch")
    torch.set_num_threads(prov["torch_threads"])
    torch.use_deterministic_algorithms(True)
    items = cohort(spec)
    old = {i["id"]: i for i in spec["checkpoints"] if i["version"] == 5}
    old_cells = {(r["checkpoint"], r["opponent"]): r for r in read(ARCHIVE / "report.json")["cells"]}
    rows = []
    for item in items:
        if item["arm"] == "hierarchical":
            recorded = old[item["id"]]
            if any(item[k] != recorded[k] for k in ("version", "source", "sha256", "training_seed")):
                raise ValueError(f"Archived hierarchy identity mismatch: {item['id']}")
            for opponent in spec["opponents"]:
                cell = old_cells[item["id"], opponent]
                rows.append(validate_log(ARCHIVE / cell["log"], cell, item["id"], opponent, seeds, policy_seeds))
        print(f"Verified {item['id']} {item['sha256']}", flush=True)
    write(output / "provenance.json", dict(archive=str(ARCHIVE.relative_to(ROOT)), archive_sha256=sha256(ARCHIVE / "provenance.json"),
        checkpoints=items, seeds=seeds, policy_seeds=policy_seeds, horizon=spec["horizon"], opponents=spec["opponents"],
        policy=prov["policy"], device=prov["device"], torch_threads=prov["torch_threads"], lock_sha256=prov["lock_sha256"],
        cyborg_revision=prov["cyborg_revision"], scenario_sha256=prov["scenario_sha256"],
        source_revision=git("rev-parse", "HEAD"), command=sys.argv))
    (output / "source.diff").write_text(git("diff", "--no-ext-diff", "--binary", "HEAD", "--", "environments/cage2") + "\n")
    for item in items:
        if item["arm"] != "flat":
            continue
        policy, scenario = load_policy(item, spec, torch.device("cpu"))
        if sha256(scenario) != prov["scenario_sha256"]:
            raise ValueError("Scenario differs from the archived one")
        for opponent in spec["opponents"]:
            cell = run_cell(policy, item, opponent, seeds, output)
            rows.append(validate_log(output / cell["log"], None, item["id"], opponent, seeds, policy_seeds))
            print(f"{item['id']} / {opponent}: {cell['mean_return']:.3f}", flush=True)
    for item in items:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Checkpoint changed during evaluation: {item['id']}")
    summary = {}
    for opponent in spec["opponents"]:
        means = {r["checkpoint"]: r["mean_return"] for r in rows if r["opponent"] == opponent}
        flat = [means[f"flat_seed{s}"] for s in range(3)]
        hierarchy = [means[f"hierarchical_seed{s}"] for s in range(3)]
        deltas = [h - f for f, h in zip(flat, hierarchy)]
        summary[opponent] = dict(flat=metrics(flat), hierarchical=metrics(hierarchy),
                                 hierarchical_minus_flat=metrics(deltas),
                                 deltas_by_training_seed=dict(enumerate(deltas)))
    write(output / "report.json", dict(cells=rows, by_opponent=summary))


if __name__ == "__main__":
    main()
