"""Aggregate completed checkpoint records and render the investigation."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY, TEAL, MIST, ORANGE, IVORY = "#13344C", "#499FA4", "#A0C3D9", "#FF9636", "#F2F3EE"
LAGS = (0, 1, 2, 4, 6, 8)
STYLES = [
    ("mode", "encoder", "Encoder embedding", TEAL, "-"),
    ("mode", "probabilities", "Posterior probabilities", NAVY, "--"),
    ("mode", "z", "Categorical argmax", ORANGE, "-."),
    ("sample", "z", "Categorical sample", NAVY, ":"),
    ("sample", "h", "Recurrent state (sampled history)", MIST, "--"),
]


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=320)
    plt.close(fig)


def aggregate(finals, nboot=2000):
    rng = np.random.default_rng(931447)
    result = {}
    for cond in ("mode", "sample"):
        probes = [r["conditions"][cond]["probes"] for r in finals]
        if not probes:
            continue
        for tag in sorted(set.intersection(*(set(probe) for probe in probes))):
            for family in ("linear", "rbf"):
                x = np.stack([p[tag][family]["accuracy_by_draw"] for p in probes])
                ns, nl, nd = x.shape
                s = rng.integers(ns, size=(nboot, ns, 1, 1))
                l = rng.integers(nl, size=(nboot, 1, nl, 1))
                d = rng.integers(nd, size=(nboot, 1, nl, nd))
                means = x.mean(axis=(1, 2))
                result[f"{cond}/{tag}/{family}"] = dict(mean=float(means.mean()), n_seeds=ns,
                    seed_sample_sd=float(means.std(ddof=1)) if ns > 1 else None,
                    crossed_seed_level_draw_ci95=np.quantile(x[s, l, d].mean((1, 2, 3)), [.025, .975]).tolist(),
                    seed_means=means.tolist())
    return result


def point(r, cond, feature, lag, family="linear"):
    return r["conditions"][cond]["probes"][f"{feature}/{lag}"][family]["accuracy"]["mean"]


def corrected(out, historical):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rows = list(historical.values())
    for key, label, color, style in [
        ("observation", "Encoder embedding", TEAL, "-"), ("h", "Recurrent state", NAVY, "--"),
        ("z", "Categorical argmax", ORANGE, "-."), ("hz", "Combined state (argmax history)", MIST, ":")]:
        curves = [[r["accuracy"] for r in row["retention"] if r["feature"] == key] for row in rows]
        ax.plot(LAGS, np.mean(curves, axis=0), label=label, color=color, ls=style, lw=2)
    ax.axhline(.5, color=NAVY, alpha=.3, lw=.8)
    ax.set(xlabel="Actions since cue", ylabel="Held-out ridge accuracy", ylim=(0, 1.04))
    ax.legend(frameon=False, loc="lower left")
    save(fig, out, "historical_memory_retention_corrected")


def figures(out, records, stats):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, cond in zip(axs, ("mode", "sample")):
        for feature, label, color, style in [
            ("encoder", "Encoder embedding", TEAL, "-"), ("probabilities", "Posterior probabilities", NAVY, "--"),
            ("z", "Categorical argmax" if cond == "mode" else "Categorical sample", ORANGE, "-."),
            ("h", "Recurrent state", NAVY, ":"), ("hz", "Combined state", MIST, "-")]:
            pairs = [(t, stats.get(f"{cond}/{feature}/{t}/linear")) for t in LAGS]
            pairs = [(t, r) for t, r in pairs if r is not None]
            if pairs:
                x, r = zip(*pairs)
                ax.plot(x, [v["mean"] for v in r], label=label, color=color, ls=style, lw=2)
                lo, hi = np.array([v["crossed_seed_level_draw_ci95"] for v in r]).T
                ax.fill_between(x, lo, hi, color=color, alpha=.12)
        ax.axhline(.5, color=NAVY, alpha=.3, lw=.8)
        ax.set(title="Argmax filtering" if cond == "mode" else "Posterior-sampled filtering",
               xlabel="Actions since cue", ylim=(0, 1.04))
        ax.legend(frameon=False, loc="lower left", fontsize=8)
    axs[0].set_ylabel("Held-out linear cue accuracy")
    save(fig, out, "final_representation_retention")

    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    for col, seed in enumerate((19, 23, 29)):
        series = sorted((r for r in records if r["seed"] == seed and not r["smoke"]), key=lambda r: r["update"])
        for row, lag in enumerate((0, 8)):
            ax = axs[row, col]
            for cond, feature, label, color, style in STYLES:
                ax.plot([r["update"] for r in series], [point(r, cond, feature, lag) for r in series],
                        label=label, ls=style, color=color, lw=1.5, marker=".", ms=3)
            ax.axhline(.5, color=NAVY, alpha=.3, lw=.8)
            ax.set(title=f"Seed {seed}, lag {lag}", ylim=(0, 1.04))
            if row == 1:
                ax.set_xlabel("Training update")
    axs[0, 0].set_ylabel("Held-out cue accuracy")
    axs[1, 0].set_ylabel("Held-out cue accuracy")
    axs[1, 0].legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, out, "checkpoint_cue_retention")


def number(x):
    return f"{x:.7g}"


def table_line(label, values):
    return f"| {label} | " + " | ".join(map(number, values)) + " |"


def render(out, completion):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "figure.facecolor": IVORY,
                         "axes.facecolor": IVORY, "text.color": NAVY, "axes.labelcolor": NAVY,
                         "axes.edgecolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                         "axes.spines.top": False, "axes.spines.right": False})
    manifest_path = out / "execution_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    paths = sorted((out / "checkpoints").glob("*.json"))
    records = [json.loads(p.read_text()) for p in paths]
    records = [r for r in records if r["source_sha256"] == manifest.get("source_sha256")]
    finals = sorted((r for r in records if r["final"] and not r["smoke"]), key=lambda r: r["seed"])
    stats = aggregate(finals)
    (out / "diagnostic_results.json").write_text(json.dumps({
        "checkpoint_records": [str(p) for p in paths], "final_aggregate": stats,
        "completed_checkpoints": len(records), "final_seeds": [r["seed"] for r in finals]}, indent=2) + "\n")
    if manifest.get("historical"):
        corrected(out, manifest["historical"])
    if finals:
        figures(out, records, stats)

    lines = ["# DreamerV3 T-maze representation bottleneck", "",
             f"Execution state **{completion['state']}**. Completed checkpoint records **{len(records)}/65**. "
             "All model calls used frozen weights on CPU.", ""]
    if not finals:
        lines += ["Final checkpoint measurements are unavailable in this attempt. See the completion manifest and worker log.", ""]
    else:
        lines += ["## Reset encoding and junction retention", "",
                  "| Seed | Raw | Preprocessed | Encoder | Raw logits | Before unimix | After unimix | Argmax z | Sampled z | Sampled h, lag 8 |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in finals:
            vals = [point(r, "mode", f, 0) for f in ("raw", "preprocessed", "encoder", "logits",
                    "probabilities_before_unimix", "probabilities", "z")]
            lines.append(table_line(r["seed"], vals + [point(r, "sample", "z", 0), point(r, "sample", "h", 8)]))
        lines += ["", "Entries are held-out linear accuracies. The aggregate JSON includes RBF results, "
                  "crossed training-seed, level and posterior-draw bootstrap intervals, and seed-to-seed SD. "
                  "Checkpoint JSON includes paired rank concordance, label-swap controls and distances.", "",
                  "| Seed | Posterior L2 | Mean factor JS (nats) | Exact argmax pair equality | Sampled z L2 | Cue-cell reconstruction | All-cell reconstruction |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in finals:
            mode, sample = (r["conditions"][c] for c in ("mode", "sample"))
            lines.append(table_line(r["seed"], [mode["separation"]["probabilities/0"]["l2"]["mean"],
                mode["latent"]["0"]["paired_mean_factor_js_nats"]["mean"],
                mode["separation"]["z/0"]["exact_equal_fraction"], sample["separation"]["z/0"]["l2"]["mean"],
                sample["reconstruction"]["0"]["cue_accuracy"]["mean"], sample["reconstruction"]["0"]["all_accuracy"]["mean"]]))
        lines += [""]
        for r in finals:
            m = r["conditions"]["mode"]["latent"]["0"]
            lines.append(f"- Seed {r['seed']} has reset entropy {number(m['entropy_per_factor_nats']['mean'])} nats/factor, "
                f"prior–posterior KL {number(m['kl_per_state_nats']['mean'])} nats/state, and {m['active_factors']} active factors. "
                f"The paired average single-draw Bayes accuracy upper bound at reset is "
                f"{number((1 + m['pair_product_tv_upper']['mean'])/2)}, allowing the decoder access to the level.")
        post = [point(r, "mode", "probabilities", 0) for r in finals]
        exact = [r["conditions"]["mode"]["separation"]["z/0"]["exact_equal_fraction"] for r in finals]
        if len(finals) == 3 and min(post) >= .75 and min(exact) == 1:
            conclusion = ("The earliest supported bottleneck is the hard categorical readout at reset. "
                          "The posterior distribution retains a decodable cue and the argmax states coincide exactly "
                          "for matched histories. Single-draw bounds quantify the cue signal available to sampled states.")
        elif len(finals) == 3 and max(post) <= .55 and min(point(r, "mode", "encoder", 0) for r in finals) >= .75:
            conclusion = ("The first attenuation in the linear probe occurs during posterior inference. "
                          "Nonlinear recovery and posterior divergence determine how strongly this can be interpreted.")
        else:
            conclusion = ("The available measurements leave a common earliest bottleneck inconclusive under the "
                          "prespecified decision rules. The seed-specific distances quantify the changes across stages.")
        lines += ["", conclusion, "",
                  "A chance probe score limits what that probe recovered. Nonzero JS establishes distribution sensitivity. "
                  "Cue-region reconstruction and open-loop checks are separate from decision accuracy.", "",
                  "## Checkpoints and controls", "",
                  "| Checkpoint | Encoder reset | Posterior reset | Argmax reset | Sample reset | Argmax h at junction | Sample h at junction |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in [r for r in records if r["smoke"] or r["final"]]:
            label = f"{'Smoke' if r['smoke'] else 'Baseline'} seed {r['seed']}, update {r['update']}"
            lines.append(table_line(label, [point(r, c, f, t) for c, f, t in (
                ("mode", "encoder", 0), ("mode", "probabilities", 0), ("mode", "z", 0),
                ("sample", "z", 0), ("mode", "h", 8), ("sample", "h", 8))]))
        lines += [""]
        for r in finals:
            periodic = sorted((p for p in records if not p["smoke"] and p["seed"] == r["seed"]), key=lambda p:p["update"])
            equal = [p["update"] for p in periodic if
                     p["conditions"]["mode"]["separation"]["z/0"]["exact_equal_fraction"] == 1]
            mode = r["conditions"]["mode"]
            mark = mode["controls"]["irrelevant"]["probabilities/0"]["l2"]["mean"]
            replay = r["replay"]
            imagined = point(r, "sample", "imagined_hz", 8)
            lines.append(f"- Seed {r['seed']} has complete reset argmax equality at updates {equal}. "
                f"The irrelevant marker produces reset posterior L2 {number(mark)}. "
                f"Open-loop sampled-reset combined-state accuracy at lag 8 is {number(imagined)}. "
                f"Replay audit covered {sum(replay['cue_counts'])} episodes and {replay['sampled_windows']} sampled windows. "
                f"Maximum manual-prefix versus replay h discrepancy was {number(replay['h_max_abs_manual_prefix_error'])}.")
        lines += ["", "Synthetic linear and XOR positives and identical-history and cue-neutral controls are checked "
                  "automatically. The smoke rows use the same trajectories and split as the final checkpoints. "
                  "Their differing runs and the 200-update checkpoint spacing limit comparisons of onset timing.", "",
                  "## Temporal and implementation audit", "",
                  "The pre-observation h, z and previous action are zero. The reset flag clears incoming state and action. "
                  "Reported reset h is the transition of these cleared inputs. Reset observations first enter the encoder "
                  "and posterior z. Recurrent h first receives that z after one action. The eight stem moves precede "
                  "the terminal branch action and signed reward at replay index 9.", "",
                  "Replay stores ten rows. The eight-row supervised window starts at 0, 1 or 2. Only start 0 directly "
                  "supervises reset reconstruction and only start 2 includes the branch reward. Starts 1 and 2 filter "
                  "the reset through detached burn-in. Reward loss therefore has no direct gradient path through that "
                  "episode's reset encoding. Whether this constraint caused the learned representation is unresolved.", "",
                  "| Path | Observation-conditioned state | Imagined successor |", "|---|---|---|",
                  "| Training collection and replay | One categorical draw | Prior sample in actor training |",
                  "| Default evaluation | Categorical argmax | Prior mode in deterministic imagination when used |",
                  "| Original history-swap diagnostic | Categorical argmax | Prior mode |",
                  "| This investigation | Argmax and 24 matched posterior draws | Fixed-action prior modes, seeded from both reset conditions |",
                  "", "The historical observation curve was the encoder embedding. The historical stochastic-state curve "
                  "was categorical argmax. The corrected figure uses these labels. Per-factor entropy, utilization and "
                  "paired JS are in checkpoint JSON. Open-loop probes use fixed actions and no later observations.", "",
                  "## Smallest follow-on experiment", "",
                  "Keep the checkpoints frozen. Select separated cue-conditioned reset hard states from posterior draws "
                  "using fit levels and test their factual, swapped and neutral histories on held-out levels under the "
                  "same stem actions. Compare recurrent separation, branch reward margin and decision accuracy. "
                  "This intervention tests whether stronger reset separation can propagate and affect choice.", "",
                  "## Uncertainty and scope", "",
                  "Only three training seeds and 16 paired test levels are available. The same maze geometry and historical "
                  "test levels are used throughout. Bootstrap intervals condition on the fitted probes. The RBF bandwidth "
                  "is based on fit data and regularization is selected on validation; test labels only score fixed predictions. "
                  "Label swaps preserve paired balance. Common random numbers preserve each posterior marginal. "
                  "The paired distances for sampled states depend on that coupling. The Bayes bound applies to a single "
                  "reset draw. Open-loop prior sampling and end-to-end policy effects remain untested by the new probes.", "",
                  "![Corrected historical figure](historical_memory_retention_corrected.png)", "",
                  "![Final cue retention](final_representation_retention.png)", "",
                  "![Checkpoint cue retention](checkpoint_cue_retention.png)", ""]
    lines += ["[Completion manifest](completion_manifest.json) · [Execution manifest](execution_manifest.json) · "
              "[Aggregate statistics](diagnostic_results.json)", ""]
    (out / "failure_mode_report.md").write_text("\n".join(lines))
