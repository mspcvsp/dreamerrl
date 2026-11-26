"""Render the final study's gate outcomes and descriptive support strata."""
from dreamerrl.eval.memory.presentation import MIST, NAVY, ORANGE, TEAL, plt, save_figure, style


OUTCOMES = {
    "insufficient_development_common_support": "Development did not supply enough independent exact pairs. The held-out seeds stayed closed.",
    "development_history_recovery_gate_failed": "Development support passed. Recurrent history did not reach the prespecified validation ordering threshold.",
    "insufficient_test_common_support": "A locked probe reached test. The held-out archive did not supply enough independent exact pairs for the primary inference.",
    "history_recovery_acceptance_not_met": "Independent test support passed. The recurrent probe did not meet the prespecified accuracy and interval criteria.",
    "bounded_history_recovery_supported": "The fixed recurrent probe met the prespecified ordering and interval criteria in the matched population.",
    "incomplete": "The worker stopped before completing the prescribed gates.",
}


def render(out, result):
    status = result["study_status"]
    cohorts = result["cohorts"]
    lines = ["# Final CAGE2 frozen recent-session study", "",
             f"**{status.replace('_', ' ').capitalize()}**", "", OUTCOMES[status], "",
             f"Replayed {result['episodes_replayed']} archived B-line episodes and {result['native_steps']} native steps. "
             f"Test cohort opened **{'yes' if result['test_opened'] else 'no'}**.", "",
             "## Independent exact-match support", "",
             "| Split | Eligible frames | Frames with an opposite-label match in another episode | Selected pairs | Required pairs |",
             "|---|---:|---:|---:|---:|"]
    for fold, row in cohorts.items():
        c = row["coverage"]
        lines.append(f"| {fold} | {c['eligible']['frames']} | {c['with_exact_opposite_episode']['frames']} | "
                     f"{c['qualifying_episode_disjoint_pairs']} | {result['independent_pairs'][fold + '_min']} |")
    lines += ["", "## Controls and readout comparison", "",
              "| Split | Current-input max score difference | Fixed-carry max score difference | Recurrent correct pairs (95% interval) | Raw-history correct pairs (95% interval) |",
              "|---|---:|---:|---|---|"]
    for fold, row in cohorts.items():
        controls = [row["controls"][k].get("score_max_abs") for k in ("current", "fixed_carry")]
        probe = []
        for kind in ("recurrent", "raw_history"):
            metrics = row.get("probes", {}).get(kind)
            probe.append("Unavailable" if metrics is None else
                         f"{metrics['correct']}/{metrics['pairs']} ({metrics['exact_ci95'][0]:.4f}–{metrics['exact_ci95'][1]:.4f})")
        lines.append(f"| {fold} | " + " | ".join("Unavailable" if x is None else f"{x:.6g}"
                                                for x in controls) + " | " + " | ".join(probe) + " |")
    lines += ["", "When scores were unavailable, JSON retains feature-level tie checks on any selected pairs. "
              "A support failure leaves historical recoverability unresolved. Stratified counts are descriptive. "
              "The exact intervals concern the episode-disjoint matched sample under the stated independence assumptions.", ""]
    for fold, row in cohorts.items():
        chosen = row["coverage"]["selected"]
        lines += [f"### {fold.capitalize()} selected coverage", "",
                  "| Elapsed time | Negative frames | Positive frames |", "|---:|---:|---:|"]
        for t, counts in sorted(chosen["by_elapsed_time"].items(), key=lambda kv: int(kv[0])):
            lines.append(f"| {t} | {counts['negative']} | {counts['positive']} |")
        lines += ["", "| Frames since last active session | Negative frames | Positive frames |", "|---|---:|---:|"]
        for age, counts in sorted(chosen["by_time_since_last_active"].items(),
                                  key=lambda kv: int(kv[0]) if kv[0] != "never" else 30):
            lines.append(f"| {age} | {counts['negative']} | {counts['positive']} |")
        lines += [""]
    lines += ["The target was recent hidden session history on Enterprise2, conditional on current inactivity. "
              "Privilege was restricted to offline eligibility and labels. Matching held all current Blue bits, "
              "elapsed time and the incoming action fixed. Any measured advantage concerns information recoverability "
              "in frozen states under this policy and opponent. Policy utilization remains outside the test.", "",
              "This was the final approved experiment. No follow-on investigation was launched. "
              "The source commit, worker status and commands are recorded in the execution and completion manifests.", "",
              "[Machine-readable results](results.json) · [Execution manifest](execution_manifest.json)", "",
              "![Qualifying match coverage](match_coverage.png)", ""]
    (out / "report.md").write_text("\n".join(lines))

    style()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    palette = {"fit": NAVY, "validation": TEAL, "test": ORANGE}
    for fold, row in cohorts.items():
        def summarize(key, indices):
            c = row["coverage"]["with_exact_opposite_episode"][key]
            return [sum(c.get(i, {}).values()) for i in indices]

        times = list(range(4, 30))
        axes[0].plot(times, summarize("by_elapsed_time", list(map(str, times))), color=palette[fold], label=fold,
                     ls={"fit": "-", "validation": "--", "test": ":"}[fold])
        ages = list(range(1, 30))
        axes[1].plot(range(1, 31), summarize("by_time_since_last_active", [*map(str, ages), "never"]),
                     color=palette[fold], ls={"fit": "-", "validation": "--", "test": ":"}[fold])
    axes[0].set(xlabel="Elapsed decisions", ylabel="Matchable frames")
    axes[1].set(xlabel="Frames since last active session", ylabel="Matchable frames",
                xticks=[1, 4, 10, 20, 30], xticklabels=["1", "4", "10", "20", "Never"])
    if cohorts:
        axes[0].legend(frameon=False)
    for ax in axes:
        ax.axhline(0, color=MIST, lw=.8)
    save_figure(fig, out / "match_coverage")

    test = cohorts.get("test", {}).get("probes")
    if test:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for i, (name, color) in enumerate((("recurrent", TEAL), ("raw_history", ORANGE))):
            row = test[name]
            ax.hlines(i, *row["exact_ci95"], color=color, lw=2)
            ax.scatter(row["correct_fraction"], i, color=NAVY, zorder=2)
        ax.set(xlim=(0, 1.02), yticks=[0, 1], yticklabels=["Recurrent h", "Raw history"], xlabel="Correct pair ordering")
        ax.axvline(.5, ls="--", lw=.8, color=NAVY)
        save_figure(fig, out / "readout_comparison")
