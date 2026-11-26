"""Variance-aware offline goal autoencoders on the archived Room 5x5 split."""

import argparse
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from dreamerrl.eval.location_probe import load_frozen_agent
from dreamerrl.models.goal_autoencoder import GoalAutoencoder
from goal_representation import (
    CONFIG as BASE_CONFIG, consecutive_pairs, digest, dump, episode_split, fit_model,
    geometry, mean_sd, occupancy, provenance, spatial_probe,
)

TAG = "dreamerv3-director-normalized-goals-v1"
CONFIG = Path(__file__).with_suffix(".json")


def fit_scale(target, fit, floor):
    fitting = target[fit]
    center = fitting.mean(0)
    spread = fitting.std(dim=0, unbiased=False)
    return center, spread.clamp_min(floor), spread < floor


def prepare(features, pairs, split, hidden, arm, floor):
    width = hidden if arm["h_only"] else features.shape[1]
    starts = torch.from_numpy(features[pairs[:, 0], :width].copy())
    futures = torch.from_numpy(features[pairs[:, 1:], :width].copy())
    target = futures[:, -1]
    if arm["displacement"]:
        target = target - starts
    center, scale, floored = fit_scale(target, split["fit"], floor)
    vectors = (target - center) / scale if arm["standardized"] else target
    return dict(starts=starts, futures=futures, target=target, center=center, scale=scale,
                floored=floored, vectors=vectors)


@torch.no_grad()
def assess(model, data, arm, context, pairs, split, hidden, alphas):
    logits, decoded = model(data["vectors"])
    prototypes = model.decode_ids(torch.arange(model.codes))
    scale, center = data["scale"], data["center"]
    if arm["standardized"]:
        decoded = decoded * scale + center
        prototypes = prototypes * scale + center
    test = split["test"]
    starts, futures = data["starts"][test], data["futures"][test]
    target = data["target"][test]
    error = decoded[test] - target
    base_error = center - target
    forecast = prototypes[None, :, :]
    baseline = center[None, None, :]
    if arm["displacement"]:
        forecast = forecast + starts[:, None]
        baseline = baseline + starts[:, None]
    distance = (forecast[:, :, None] - futures[:, None]).square().mean(-1)
    distance_h = (forecast[:, :, None, :hidden] - futures[:, None, :, :hidden]).square().mean(-1)
    base_distance = (baseline[:, :, None] - futures[:, None]).square().mean(-1)
    base_distance_h = (baseline[:, :, None, :hidden] - futures[:, None, :, :hidden]).square().mean(-1)
    codes = logits.argmax(-1)[test].numpy()
    selected = distance[np.arange(len(codes)), codes]
    mse = float(error.square().mean())
    normalized = float((error / scale).square().mean())
    mean_mse = float(base_error.square().mean())
    mean_normalized = float((base_error / scale).square().mean())
    return dict(
        width=int(target.shape[-1]), parameters=sum(p.numel() for p in model.parameters()),
        original_mse=mse, original_h_mse=float(error[:, :hidden].square().mean()),
        normalized_mse=normalized, constant_mse=mean_mse,
        constant_h_mse=float(base_error[:, :hidden].square().mean()),
        constant_normalized_mse=mean_normalized,
        original_skill=1 - mse / max(mean_mse, 1e-12),
        normalized_skill=1 - normalized / max(mean_normalized, 1e-12),
        floored_dimensions=int(data["floored"].sum()),
        occupancy=occupancy(codes, model.codes),
        geometry=geometry(prototypes.numpy()),
        normalized_geometry=geometry(((prototypes - center) / scale).numpy()),
        decoded_normalized_variance=float(((decoded[test] - center) / scale).var(dim=0, unbiased=False).mean()),
        target_normalized_variance=float(((target - center) / scale).var(dim=0, unbiased=False).mean()),
        nearest_mse=float(distance.min(-1).values.mean()),
        nearest_h_mse=float(distance_h.min(-1).values.mean()),
        constant_nearest_mse=float(base_distance.min(-1).values.mean()),
        constant_nearest_h_mse=float(base_distance_h.min(-1).values.mean()),
        encoded_nearest_mse=float(selected.min(-1).values.mean()),
        probe=spatial_probe(context, pairs, split, decoded.numpy(), alphas),
    )


def rmse(row):
    return sum(row["probe"][axis]["rmse"] for axis in ("x", "y")) / 2


def compare(models):
    control, frozen = models["A"], models["Frozen"][0]
    result = {}
    for arm in "BCD":
        pairs = []
        for a, b in zip(control, models[arm]):
            if a["seed"] != b["seed"]:
                raise ValueError("Unmatched initialization")
            pairs.append(dict(
                seed=a["seed"], rmse_delta_vs_a=rmse(b) - rmse(a), rmse_delta_vs_frozen=rmse(b) - rmse(frozen),
                effective_codes=b["occupancy"]["hard_effective"], normalized_skill=b["normalized_skill"],
                per_axis_rmse_delta={axis: b["probe"][axis]["rmse"] - a["probe"][axis]["rmse"] for axis in ("x", "y")},
            ))
        result[arm] = dict(
            paired=pairs, wins_vs_a=sum(r["rmse_delta_vs_a"] < 0 for r in pairs),
            wins_vs_frozen=sum(r["rmse_delta_vs_frozen"] < 0 for r in pairs),
            meets_success=all(
                r["rmse_delta_vs_a"] < 0 and r["rmse_delta_vs_frozen"] < 0
                and r["effective_codes"] >= 2 and r["normalized_skill"] > 0 for r in pairs
            ),
        )
    return result


def plot(output, models, controls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = "ABCD"
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for col, name in enumerate(arms):
        representative = models[name][0]
        counts = representative["occupancy"]["counts"]
        axes[0, col].bar(range(len(counts)), counts)
        axes[0, col].set(title=f"{name}, seed {representative['seed']}", xlabel="Code", ylabel="Test count")
        im = axes[1, col].imshow(representative["geometry"]["cosine"], cmap="coolwarm", vmin=-1, vmax=1)
        axes[1, col].set(xlabel="Code", ylabel="Code")
    fig.colorbar(im, ax=axes[1, :].tolist(), label="Original-unit cosine")
    fig.savefig(output / "geometry_and_utilization.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for index, name in enumerate(arms):
        axes[0].scatter([index] * len(models[name]), [rmse(r) for r in models[name]])
        axes[1].scatter([index] * len(models[name]), [r["occupancy"]["hard_effective"] for r in models[name]])
    axes[0].axhline(sum(controls[a]["rmse"] for a in ("x", "y")) / 2, linestyle="--", color="gray")
    axes[1].axhline(2, linestyle="--", color="gray")
    for ax, label in zip(axes, ("Mean X/Y displacement RMSE", "Effective codes")):
        ax.set(xticks=list(range(4)), xticklabels=list(arms), ylabel=label)
    fig.savefig(output / "initialization_variability.png", dpi=140)
    plt.close(fig)


def report(result, arms):
    models, features, comparisons = result["models"], result["feature_probes"], result["comparisons"]
    text = [
        "# Variance-aware Director goal representations", "",
        f"Room 5x5, {result['pair_count']} within-episode four-action pairs, "
        f"{result['pair_counts']['test']} test pairs from {len(result['levels']['test'])} held-out levels.",
        f"Five matched initialization seeds. Fit-level standard deviations floored at {result['floor']:.2f} RSSM units. "
        "The table reports mean ± sample SD across initialization seeds. The frozen reference is one checkpoint.",
        "", "## Reconstruction and geometry", "",
        "| Arm | Original MSE | Original h MSE | Normalized MSE | Mean baseline MSE | Mean baseline normalized MSE | Original skill | Normalized skill | Effective codes | Decoded rank | Pairwise cosine |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rows in models.items():
        label = f"{name}. {arms[name]['name']}" if name in arms else "Frozen historical absolute"
        cols = [[r[k] for r in rows] for k in ("original_mse", "original_h_mse", "normalized_mse", "constant_mse",
                                                 "constant_normalized_mse", "original_skill", "normalized_skill")]
        cols += [[r["occupancy"]["hard_effective"] for r in rows]]
        cols += [[r["geometry"][k] for r in rows] for k in ("effective_rank", "mean_pairwise_cosine")]
        text.append("| " + label + " | " + " | ".join(mean_sd(c) for c in cols) + " |")
    text += [
        "", "Normalized error uses the fitting scale for the corresponding target, including control A. Absolute and displacement errors are inverted to future-state units for original MSE. D covers only h, so use the h column for comparisons with D. Each mean baseline uses its own fitting target mean. Positive skill beats that baseline.",
        "The displacement decoder's geometry is measured in unscaled displacement units. Matching code labels across initializations is unidentifiable, so the geometry figure uses the first seed. Full hard counts and both geometries are in report.json.",
        "", "![Code counts and geometry](geometry_and_utilization.png)", "",
        "## Held-out spatial probes", "",
        "All models predict actual displacement with start position and timestep controls. Added goal features are decoded native state or displacement vectors. Position enters only in these offline probes. Full RSSM and deterministic h are visited-future information references.",
        "", "| Features | X RMSE | Y RMSE | X R² | Y R² |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = {k: [v] for k, v in features.items()}
    rows.update({k: [r["probe"] for r in v] for k, v in models.items()})
    for name, probes in rows.items():
        cols = []
        for key in ("rmse", "r2"):
            for axis in ("x", "y"):
                values = [row[axis][key] for row in probes if row[axis][key] is not None]
                cols.append(mean_sd(values) if values else "—")
        text.append(f"| {name} | " + " | ".join(cols) + " |")
    text += [
        "", "## Recorded horizon proximity", "",
        "Average per-code squared distance to the closest recorded state 1–4 actions after the same start. The displacement arm anchors each code at that start. These distances do not establish worker controllability.",
        "", "| Arm | Code MSE | Mean baseline MSE | Code h MSE | Mean baseline h MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, rows in models.items():
        text.append(f"| {name} | " + " | ".join(mean_sd([r[key] for r in rows]) for key in (
            "nearest_mse", "constant_nearest_mse", "nearest_h_mse", "constant_nearest_h_mse"
        )) + " |")
    text += [
        "", "## Initialization sensitivity and decision", "",
        "Success requires a reduction in average X/Y RMSE versus both absolute references, at least two effective codes, and positive normalized reconstruction skill on every matched seed.",
        "", "| Arm | Paired RMSE change vs A | Better than A | Better than frozen | Meets all criteria |",
        "|---|---:|---:|---:|---|",
    ]
    for name, summary in comparisons.items():
        pairs = summary["paired"]
        text.append(f"| {name} | {mean_sd([r['rmse_delta_vs_a'] for r in pairs])} | {summary['wins_vs_a']}/{len(pairs)} | {summary['wins_vs_frozen']}/{len(pairs)} | {'Yes' if summary['meets_success'] else 'No'} |")
    candidates = [k for k, v in comparisons.items() if v["meets_success"]]
    text += ["", "![Across-seed variation](initialization_variability.png)", ""]
    if candidates:
        winner = min(candidates, key=lambda k: np.mean([rmse(r) for r in models[k]]))
        text.append(f"Recommend {winner}, {arms[winner]['name'].lower()}, for a subsequent controlled Room 5x5 learning experiment against A. Keep the RSSM frozen and preregister paired worker interventions so that the experiment measures actual controllability.")
    else:
        best = min(comparisons, key=lambda k: np.mean([rmse(r) for r in models[k]]))
        text.append(f"No arm met all criteria. {best}, {arms[best]['name'].lower()}, had the lowest mean RMSE among normalized arms. Keep it as the single candidate for a larger level-disjoint offline replication before reviewing a learning experiment.")
    text += [
        "", "D has fewer input and output features and parameters. It keeps the existing hidden width, code count and optimizer budget. Full-width models share identical initial weights within each seed; D uses the same seed and dimensions native to h.",
        "", "The study changes only offline autoencoders. Corridor R2 remains deferred. Exact reproduction uses launch_command with a fresh output directory.", "",
    ]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ready-channel")
    args = parser.parse_args()
    spec = json.loads(BASE_CONFIG.read_text())
    config = json.loads(CONFIG.read_text())
    arms, floor = config["arms"], config["standard_deviation_floor"]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    record = provenance(args.base, spec, tag=TAG, config=CONFIG, entry=Path(__file__))
    agent = load_frozen_agent(args.base / spec["checkpoint"], allow_hierarchical=True, load_critic=True)
    actor = agent.actor
    if agent.cfg.env.env_id != "MiniHack-Room-5x5-v0" or agent.update != spec["checkpoint_update"] or actor.interval != 4:
        raise ValueError("Wrong reference environment, update, or horizon")
    if any(p.requires_grad for mod in (agent.world, actor, agent.critic) for p in mod.parameters()):
        raise ValueError("Reference agent parameters are not frozen")
    reference = actor.goal_autoencoder
    with np.load(args.base / spec["dataset"], allow_pickle=False) as archive:
        context, features = archive["context"], archive["features"]
        with torch.no_grad():
            codes = reference.encoder(torch.from_numpy(features)).argmax(-1).numpy()
        if not np.array_equal(codes, archive["encoder_codes"]):
            raise ValueError("Archive and checkpoint encoder assignments differ")
    pairs = consecutive_pairs(context, actor.interval)
    split, levels = episode_split(context[pairs[:, 0], 0], spec["split_seed"])
    if not np.array_equal(np.unique(context[:, 0]), record["dataset_seeds"]):
        raise ValueError("Archive levels differ from the provenance record")
    hidden = agent.cfg.world.deter_size
    data = {name: prepare(features, pairs, split, hidden, arm, floor) for name, arm in arms.items()}
    dump(args.output / "provenance.json", dict(record, study=config, inherited_config_sha256=digest(BASE_CONFIG)))
    dump(args.output / "started.json", dict(pairs=len(pairs), test_pairs=int(split["test"].sum()), arms=list(arms)))
    print("Verified frozen reference, level split, and fit-only normalizers", flush=True)
    if args.ready_channel:
        subprocess.run(["tmux", "wait-for", "-S", args.ready_channel], check=True)
    shared = (context, pairs, split, hidden, spec["ridge_alphas"])
    models = {"Frozen": [assess(reference, data["A"], arms["A"], *shared)], **{name: [] for name in arms}}
    for seed in spec["initialization_seeds"]:
        torch.manual_seed(seed)
        full = GoalAutoencoder(features.shape[-1], reference.codes, reference.encoder[0].out_features)
        torch.manual_seed(seed)
        deter = GoalAutoencoder(hidden, reference.codes, reference.encoder[0].out_features)
        for name, arm in arms.items():
            fitting = data[name]
            initial = deter if arm["h_only"] else full
            model, fitted = fit_model(copy.deepcopy(initial), fitting["vectors"], split, spec)
            models[name].append(dict(assess(model, fitting, arm, *shared), seed=seed, fit=fitted))
            torch.save(dict(model=model.state_dict(), center=fitting["center"], scale=fitting["scale"], arm=arm),
                       args.output / f"{name}_seed{seed}.pt")
        print(f"Completed four arms for matched initialization {seed}", flush=True)
    future = features[pairs[:, -1]]
    probes = {name: spatial_probe(context, pairs, split, value, spec["ridge_alphas"])
              for name, value in (("Start and time", None), ("Full RSSM", future), ("Deterministic h", future[:, :hidden]))}
    for rows in models.values():
        for row in rows:
            row["incremental_spatial_r2"] = {
                axis: row["probe"][axis]["r2"] - probes["Start and time"][axis]["r2"]
                if row["probe"][axis]["r2"] is not None and probes["Start and time"][axis]["r2"] is not None else None
                for axis in ("x", "y")
            }
    result = dict(models=models, feature_probes=probes, comparisons=compare(models), pair_count=len(pairs),
                  pair_counts={name: int(mask.sum()) for name, mask in split.items()},
                  levels={name: values.tolist() for name, values in levels.items()}, floor=floor)
    for key in ("checkpoint", "dataset"):
        if digest(args.base / spec[key]) != spec[f"{key}_sha256"]:
            raise ValueError(f"Input changed during analysis: {key}")
    plot(args.output, models, probes["Start and time"])
    dump(args.output / "report.json", result)
    (args.output / "final_report.md").write_text(report(result, arms))


if __name__ == "__main__":
    main()
