"""Compare offline Room 5x5 goal autoencoders on archived, level-disjoint RSSM pairs."""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from dreamerrl.eval.location_probe import load_frozen_agent
from dreamerrl.models.goal_autoencoder import GoalAutoencoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "environments/cage2"))
from latent_probe import episode_split, metrics, ridge

TAG = "dreamerv3-director-goal-representation-v1"
CONFIG = Path(__file__).with_suffix(".json")


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", ROOT, *args], text=True).strip()


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def provenance(base, spec, *, tag=TAG, config=CONFIG, entry=Path(__file__)):
    if git("cat-file", "-t", tag) != "tag" or git("rev-parse", tag + "^{}") != git("rev-parse", "HEAD"):
        raise ValueError("Use the annotated experiment checkout")
    if git("status", "--porcelain"):
        raise ValueError("Experiment checkout has local changes")
    for key in ("source_manifest", "checkpoint", "dataset", "dataset_record"):
        if digest(base / spec[key]) != spec[f"{key}_sha256"]:
            raise ValueError(f"Fingerprint mismatch: {key}")
    old = json.loads((base / spec["source_manifest"]).read_text())
    records = json.loads((base / spec["dataset_record"]).read_text())
    revision = git("rev-parse", spec["reference_tag"] + "^{}")
    if (old["source_commit"] != revision or records["revision"] != revision
        or old["tag_object"] != git("rev-parse", spec["reference_tag"])
        or records["checkpoint_sha256"] != spec["checkpoint_sha256"]
        or records["manifest_sha256"] != spec["source_manifest_sha256"]
        or digest(ROOT / "uv.lock") != records["lock_sha256"]):
        raise ValueError("Archived dataset and checkpoint provenance disagree")
    for scope in ("dreamerrl", "environments/cage2/latent_probe.py", "uv.lock"):
        for path in git("ls-files", "--", scope).splitlines():
            if digest(ROOT / path) != old["source_fingerprints"][path]:
                raise ValueError(f"Changed reference implementation: {path}")
    import dreamerrl
    if not Path(dreamerrl.__file__).resolve().is_relative_to(ROOT):
        raise ValueError("Dreamer was imported from a different checkout")
    return dict(spec=spec, tag=tag, revision=git("rev-parse", "HEAD"), tag_object=git("rev-parse", tag),
                reference_revision=revision, config_sha256=digest(config), entry_sha256=digest(entry),
                python=sys.version, torch=torch.__version__, lock_sha256=records["lock_sha256"], device="cpu",
                dataset_seeds=records["seeds"], command=sys.argv)


def consecutive_pairs(context, horizon):
    """Return indices [t, t+1, ..., t+K] from the same uninterrupted trace."""
    order = np.lexsort((context[:, 1], context[:, 0]))
    result = []
    for level in np.unique(context[:, 0]):
        ids = order[context[order, 0] == level]
        times = context[ids, 1]
        if not np.array_equal(times, np.arange(len(times))):
            raise ValueError(f"Duplicate timestep or gap in archived level {level}")
        result.extend(ids[i:i + horizon + 1] for i in range(len(ids) - horizon))
    if not result:
        raise ValueError("No complete in-episode future pairs")
    return np.stack(result)


def fit_model(model, vectors, split, spec):
    fit, validation = vectors[split["fit"]], vectors[split["validation"]]
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
    weights = torch.ones(len(fit))
    best = None
    for step in range(1, spec["optimizer_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        objective, _, _ = model.reconstruction(fit, weights)
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            _, decoded = model(validation)
            error = float((decoded - validation).square().mean())
        if not np.isfinite(error):
            raise ValueError("Nonfinite validation loss")
        if best is None or error < best[0]:
            best = (error, step, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    return model.requires_grad_(False).eval(), dict(validation_mse=best[0], selected_update=best[1],
                                                   optimizer_steps=spec["optimizer_steps"])


def spatial_probe(context, pairs, split, features, alphas):
    initial, future = context[pairs[:, 0]], context[pairs[:, -1]]
    # Position labels only enter here, after autoencoder fitting has finished.
    displacement = future[:, 4:6] - initial[:, 4:6]
    start_position = initial[:, 2:4] + initial[:, 4:6]
    control = np.column_stack((initial[:, 1], start_position))
    matrix = control if features is None else np.column_stack((control, features))
    result = {}
    for i, axis in enumerate(("x", "y")):
        y = displacement[:, i]
        fit = ridge(matrix, y, split, alphas)
        result[axis] = metrics(y[split["test"]], fit["predictions"][split["test"]])
    return result


def geometry(prototypes):
    centered = prototypes - prototypes.mean(0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    energy = np.square(s)
    p = energy[energy > 0] / max(energy.sum(), 1e-12)
    rank = float(np.exp(-(p * np.log(p)).sum())) if energy.sum() > 1e-12 else 0.
    unit = prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)
    cosine = (unit @ unit.T).clip(-1, 1)
    pairs = cosine[np.triu_indices(len(unit), 1)]
    return dict(effective_rank=rank, mean_pairwise_cosine=float(pairs.mean()),
                pairs_cosine_above_098=int((pairs > .98).sum()), cosine=cosine.tolist(),
                pca=(u[:, :2] * s[:2]).tolist())


def occupancy(codes, count):
    counts = np.bincount(codes, minlength=count)
    p = counts[counts > 0] / counts.sum()
    return dict(counts=counts.tolist(), hard_active=int((counts > 0).sum()),
                hard_effective=float(np.exp(-(p * np.log(p)).sum())))


@torch.no_grad()
def assess(model, inputs, starts, futures, context, pairs, split, hidden, spec, displacement=False):
    logits, decoded = model(inputs)
    raw = decoded.numpy()
    start = starts.numpy()
    reconstructed = raw + start if displacement else raw
    prototypes = model.decode_ids(torch.arange(model.codes)).numpy()
    test = split["test"]
    error = (reconstructed - futures[:, -1]).astype(np.float64) ** 2
    # For each start measure all code candidates against actual future states,
    # plus the code chosen retrospectively by the encoder with access to t+K.
    candidates = prototypes[None, :, :] + start[test, None, :] if displacement else prototypes[None, :, :]
    distances = ((candidates[:, :, None, :] - futures[test, None, :, :]) ** 2).mean(-1)
    nearest = distances.min(-1)
    codes = logits.argmax(-1).numpy()[test]
    selected = distances[np.arange(len(codes)), codes]
    probe = spatial_probe(context, pairs, split, raw, spec["ridge_alphas"])
    return dict(
        reconstruction_mse=float(error[test].mean()), h_mse=float(error[test, :hidden].mean()),
        z_mse=float(error[test, hidden:].mean()), geometry=geometry(prototypes),
        occupancy=occupancy(codes, len(prototypes)), probe=probe,
        horizon=dict(mean_all_codes_nearest_mse=float(nearest.mean()),
                     selected_nearest_mse=float(selected.min(-1).mean()),
                     selected_nearest_offset_counts=np.bincount(selected.argmin(-1) + 1, minlength=futures.shape[1] + 1)[1:].tolist(),
                     per_pair_all_codes_nearest_mse=nearest.mean(1).tolist()),
    )


def figures(output, references):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, (name, r) in enumerate(references.items()):
        ax = axes[0, col]
        ax.bar(range(len(r["occupancy"]["counts"])), r["occupancy"]["counts"])
        ax.set(title=name, xlabel="Code", ylabel="Held-out count")
        ax = axes[1, col]
        im = ax.imshow(r["geometry"]["cosine"], vmin=-1, vmax=1, cmap="coolwarm")
        ax.set(xlabel="Code", ylabel="Code", title="Decoded cosine similarity")
    fig.colorbar(im, ax=axes[1].tolist(), shrink=.75)
    fig.subplots_adjust(wspace=.45, hspace=.55, right=.89)
    fig.savefig(output / "goal_geometry_utilization.png", dpi=150)
    plt.close(fig)


def mean_sd(values):
    return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}" if len(values) > 1 else f"{values[0]:.4f}"


def render(result):
    groups = result["models"]
    text = [
        "# Frozen Room 5x5 goal representations", "",
        f"{result['pair_count']} complete in-episode pairs at offset {result['horizon']}; "
        f"{result['split_counts']['test']} held-out pairs from {len(result['level_splits']['test'])} levels.",
        "All reconstruction errors use original RSSM feature units and the actual state four actions after the start.",
        "", "## Representation comparison", "",
        "Offline rows show mean ± sample SD across matched initialization seeds. The frozen row is one historical model.",
        "", "| Model | Future MSE | h MSE | z MSE | Hard active | Effective codes | Decoded effective rank | Mean decoded cosine | Nearest in-horizon MSE, all codes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, models in groups.items():
        columns = [[r[k] for r in models] for k in ("reconstruction_mse", "h_mse", "z_mse")]
        columns += [[r["occupancy"][k] for r in models] for k in ("hard_active", "hard_effective")]
        columns += [[r["geometry"][k] for r in models] for k in ("effective_rank", "mean_pairwise_cosine")]
        columns.append([r["horizon"]["mean_all_codes_nearest_mse"] for r in models])
        text.append(f"| {name} | " + " | ".join(mean_sd(c) for c in columns) + " |")
    base = result["common_features"]
    text += [
        "", "![Goal geometry and utilization for the frozen reference and first matched initialization](goal_geometry_utilization.png)", "",
        "For the displacement model, decoded cosine and rank measure residual code geometry. Its nearest-future and reconstruction scores add the start state back before comparison in RSSM space. Code selection uses the observed future for this offline diagnostic.",
        "", "## Common features and information loss", "",
        f"The fitting-set mean accounts for {base['mean_energy_fraction']:.1%} of uncentered fitting-target energy "
        f"(h {base['h_mean_energy_fraction']:.1%}, z {base['z_mean_energy_fraction']:.1%}). "
        f"On held-out pairs, a constant fitting mean gives MSE {base['fit_mean_test_mse']:.4f}; carrying the start forward gives {base['start_only_test_mse']:.4f}.",
        f"The frozen decoder's error is {groups['Frozen absolute'][0]['reconstruction_mse'] / max(base['fit_mean_test_mse'], 1e-12):.2f} times the constant-mean error. "
        "A low raw error can therefore coexist with little information about spatial changes.",
        "", "## Held-out spatial probes", "",
        "Ridge predicts observed displacement from the start. All rows include start coordinates and timestep. The displacement row adds only the decoded residual, so the start-state skip cannot by itself improve this probe. Raw RSSM rows use the visited future state as an information upper reference.",
        "", "| Probe features beyond controls | X RMSE | Y RMSE | X R² | Y R² |",
        "|---|---:|---:|---:|---:|",
    ]
    probes = {name: [r] for name, r in result["feature_probes"].items()}
    probes.update({name: [r["probe"] for r in group] for name, group in groups.items()})
    for name, rows in probes.items():
        cols = []
        for metric in ("rmse", "r2"):
            for axis in ("x", "y"):
                values = [r[axis][metric] for r in rows if r[axis][metric] is not None]
                cols.append(mean_sd(values) if values else "—")
        text.append(f"| {name} | " + " | ".join(cols) + " |")
    changes = result["paired_seed_changes"]
    text += [
        "", "## Initialization sensitivity and horizon", "",
        f"Paired MSE change (displacement minus absolute): {mean_sd([r['mse_delta'] for r in changes])}. "
        f"Paired effective-rank change: {mean_sd([r['rank_delta'] for r in changes])}. "
        f"Paired spatial RMSE change averaged over axes: {mean_sd([r['spatial_rmse_delta'] for r in changes])}.",
        "For each code, the horizon metric finds the closest recorded state at offsets 1–4 from the same start. Higher proximity is evidence about this recorded behavior distribution. The worker was frozen throughout, and the experiment does not measure policy controllability.",
        "", "## Recommended next controlled experiment", "",
    ]
    if np.mean([r["spatial_rmse_delta"] for r in changes]) < 0 and np.mean([r["rank_delta"] for r in changes]) > 0:
        text.append("First replicate the spatial and diversity gains on a second level-disjoint archive. Then review a Room 5x5 frozen-RSSM comparison that replaces only the goal representation and uses paired worker interventions to test controllability.")
    else:
        text.append("Keep the actor frozen. Replicate the matched absolute-versus-displacement test on a larger archive with more varied within-horizon movement before proposing an actor change.")
    text += ["", "Corridor R2 remains deferred. Per-seed metrics, decoded geometry and exact splits are in report.json. Reproduce with launch_command and a fresh output directory.", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ready-channel")
    args = parser.parse_args()
    spec = json.loads(CONFIG.read_text())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    record = provenance(args.base, spec)
    agent = load_frozen_agent(args.base / spec["checkpoint"], allow_hierarchical=True, load_critic=True)
    actor = agent.actor
    if agent.cfg.env.env_id != "MiniHack-Room-5x5-v0" or agent.update != spec["checkpoint_update"]:
        raise ValueError("Unexpected frozen checkpoint")
    if any(p.requires_grad for module in (agent.world, actor, agent.critic) for p in module.parameters()):
        raise ValueError("Checkpoint model is not frozen")
    reference = actor.goal_autoencoder
    with np.load(args.base / spec["dataset"], allow_pickle=False) as archive:
        context, features = archive["context"], archive["features"]
        archived_codes = archive["encoder_codes"]
    if not np.array_equal(np.unique(context[:, 0]), record["dataset_seeds"]) or len(record["dataset_seeds"]) != spec["expected_levels"]:
        raise ValueError("Archive levels differ from the provenance record")
    with torch.no_grad():
        replay_codes = reference.encoder(torch.from_numpy(features)).argmax(-1).numpy()
    if not np.array_equal(archived_codes, replay_codes):
        raise ValueError("Archive encoder codes do not match the frozen checkpoint")
    pairs = consecutive_pairs(context, actor.interval)
    split, groups = episode_split(context[pairs[:, 0], 0], spec["split_seed"])
    starts = torch.from_numpy(features[pairs[:, 0]])
    futures = features[pairs[:, 1:]]
    target = torch.from_numpy(futures[:, -1].copy())
    residual = target - starts
    hidden = agent.cfg.world.deter_size
    fit = target.numpy()[split["fit"]]
    constant = fit.mean(0)
    test = split["test"]
    fractions = [float((fit[:, sl].mean(0) ** 2).sum() / (fit[:, sl] ** 2).sum(1).mean())
                 for sl in (slice(None), slice(None, hidden), slice(hidden, None))]
    args.output.mkdir(exist_ok=True)
    dump(args.output / "provenance.json", record)
    dump(args.output / "started.json", dict(pairs=len(pairs), levels=len(record["dataset_seeds"]), horizon=actor.interval))
    print("Verified frozen checkpoint, archive encoder assignments, and level-disjoint pairs", flush=True)
    if args.ready_channel:
        subprocess.run(["tmux", "wait-for", "-S", args.ready_channel], check=True)
    shared = (starts, futures, context, pairs, split, hidden, spec)
    models = {
        "Frozen absolute": [assess(reference, target, *shared)],
        "Offline absolute": [], "Offline displacement": [],
    }
    changes = []
    for seed in spec["initialization_seeds"]:
        torch.manual_seed(seed)
        initial = GoalAutoencoder(target.shape[1], reference.codes, reference.encoder[0].out_features)
        for name, inputs in (("Offline absolute", target), ("Offline displacement", residual)):
            trained, fit_result = fit_model(copy.deepcopy(initial), inputs, split, spec)
            results = assess(trained, inputs, *shared, displacement=name == "Offline displacement")
            models[name].append(dict(results, seed=seed, fit=fit_result))
            torch.save(trained.state_dict(), args.output / f"{name.lower().replace(' ', '_')}_seed{seed}.pt")
        absolute, displacement = models["Offline absolute"][-1], models["Offline displacement"][-1]
        changes.append(dict(
            seed=seed, mse_delta=displacement["reconstruction_mse"] - absolute["reconstruction_mse"],
            rank_delta=displacement["geometry"]["effective_rank"] - absolute["geometry"]["effective_rank"],
            spatial_rmse_delta=float(np.mean([displacement["probe"][a]["rmse"] - absolute["probe"][a]["rmse"] for a in ("x", "y")])),
        ))
        print(f"Completed matched initialization {seed}", flush=True)
    result = dict(
        models=models, pair_count=len(pairs), horizon=actor.interval, paired_seed_changes=changes,
        level_splits={k: v.tolist() for k, v in groups.items()}, split_counts={k: int(v.sum()) for k, v in split.items()},
        common_features=dict(mean_energy_fraction=fractions[0], h_mean_energy_fraction=fractions[1],
                             z_mean_energy_fraction=fractions[2],
                             fit_mean_test_mse=float(((target.numpy()[test] - constant) ** 2).mean()),
                             start_only_test_mse=float((residual.numpy()[test] ** 2).mean())),
        feature_probes={name: spatial_probe(context, pairs, split, value, spec["ridge_alphas"])
                        for name, value in (("Controls only", None), ("Full RSSM", target.numpy()),
                                            ("Deterministic h", target.numpy()[:, :hidden]))},
    )
    for key in ("checkpoint", "dataset"):
        if digest(args.base / spec[key]) != spec[f"{key}_sha256"]:
            raise ValueError(f"Input changed during analysis: {key}")
    figures(args.output, {name: group[0] for name, group in models.items()})
    dump(args.output / "report.json", result)
    (args.output / "final_report.md").write_text(render(result))


if __name__ == "__main__":
    main()
