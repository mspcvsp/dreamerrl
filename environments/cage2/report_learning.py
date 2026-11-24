"""Render saved paired CAGE2 learning-curve scores; never selects training settings."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    scores = json.loads((run / "scorecard.json").read_text())
    complete = json.loads((run / "complete.json").read_text())
    baselines = complete["baseline_means"]
    with (run / "scorecard.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["update", "environment_steps", "mean_return", "paired_change_from_update1", "ci95_low", "ci95_high"]
        )
        for score in scores:
            writer.writerow(
                [
                    score["update"],
                    score["env_steps"],
                    score["mean_return"],
                    score["paired_mean_change"],
                    *score["paired_ci95"],
                ]
            )
    final = scores[-1]
    low, high = final["paired_ci95"]
    if low > 0:
        conclusion = "The predeclared final policy improved return over update 1 on these development seeds."
        recommendation = (
            "Replicate this unchanged setup with an independent training seed before increasing complexity."
        )
    elif high < 0:
        conclusion = "The predeclared final policy reduced return relative to update 1 on these development seeds."
        recommendation = (
            "Inspect imagined reward/critic estimates for frequent final-policy operations "
            "before another training budget."
        )
    else:
        conclusion = "The final paired interval includes zero; this run does not establish improved defensive return."
        recommendation = (
            "Audit imagined versus real returns for the final actor on these saved development "
            "episodes before more training."
        )
    lines = [
        "CAGE2 Blue versus supplied B-line, Scenario2, 30 steps, one training seed.",
        "Higher native return is better. Greedy actions and posterior modes at every checkpoint.",
        f"Sleeping baseline: {baselines['sleep']:.4f}; seeded random baseline: {baselines['random']:.4f}.",
        "",
        "Update | Environment steps | Mean return | Paired change | 95% paired interval",
    ]
    for row in scores:
        a, b = row["paired_ci95"]
        lines.append(
            f"{row['update']} | {row['env_steps']} | {row['mean_return']:.4f} | "
            f"{row['paired_mean_change']:+.4f} | [{a:+.4f}, {b:+.4f}]"
        )
    lines += [
        "",
        conclusion,
        recommendation,
        "Intervals resample the 20 matched development seeds. "
        "They do not quantify variation across independent training runs.",
        f"Best intermediate/update-1 result is exploratory: update {complete['exploratory_best']['update']}.",
        "",
        "Final operation counts: " + json.dumps(final["operation_counts"]),
        "Final target-host counts: " + json.dumps(final["target_host_counts"]),
        f"Final Restore frequency: {final['restore_count']}/600 ({final['restore_frequency']:.3%}).",
        "",
        "See evaluation_*.json and baseline_*.json for raw actions, per-step rewards, "
        "native reward breakdowns and per-episode returns.",
        "See scorecard.json for each checkpoint's per-seed differences, operation/host counts, "
        "and return standard deviation.",
    ]
    (run / "final_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
