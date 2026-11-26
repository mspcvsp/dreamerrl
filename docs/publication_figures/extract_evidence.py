"""Export only public aggregate figure values from the preserved local runs."""
import argparse
import hashlib
import json
from pathlib import Path


def collect(root):
    evidence = {}
    sources = {}

    def source(relative):
        path = root / "runs" / relative
        data = path.read_bytes()
        sources["runs/" + relative] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    matched = source("cage2_frozen_matched_20261126T133406Z/report.json")
    evidence["behavior"] = matched["by_opponent"]
    evidence["behavior_seed_means"] = {
        opponent: {kind: [cell["mean_return"] for cell in matched["cells"]
                          if cell["opponent"] == opponent and cell["checkpoint"].startswith(kind + "_")]
                   for kind in ("flat", "hierarchical")}
        for opponent in evidence["behavior"]}
    base = "minihack_tmaze_representation_diagnostics_20261126"
    e = source(base + "/diagnostic_results.json")["final_aggregate"]
    stages = ("raw", "preprocessed", "encoder", "logits", "probabilities_before_unimix", "probabilities", "z")
    evidence["reset"] = {s: e[f"mode/{s}/0/linear"]["seed_means"] for s in stages}
    evidence["reset"]["sample"] = e["sample/z/0/linear"]["seed_means"]
    evidence["sample_rbf"] = e["sample/z/0/rbf"]["mean"]
    evidence["bounds"], evidence["reconstruction"] = {}, {}
    for seed in (19, 23, 29):
        record = source(f"{base}/checkpoints/seed{seed}_4000.json")
        row = record["conditions"]["mode"]["latent"]["0"]["pair_product_tv_upper"]
        evidence["bounds"][seed] = (1 + row["mean"]) / 2
        recon = record["conditions"]["sample"]["reconstruction"]["0"]
        evidence["reconstruction"][seed] = {k: recon[k]["mean"] for k in ("cue_accuracy", "all_accuracy")}
        evidence["heldout_level_pairs"] = len(record["split"]["test"])
    intervention = source("frozen_memory_integration_20261126/tmaze/results.json")
    evidence["intervention"] = {
        r["seed"]: {
            k: {**{n: r["controls"][k][n]["mean"] for n in ("decision_accuracy", "right_shift", "reward_shift", "value_shift")},
                "junction_h_accuracy": r["controls"][k]["retention"]["8"]["linear"]["mean"],
                "h_l2_lag1": r["controls"][k]["retention"]["1"]["cue_h_l2"]["mean"],
                "h_l2_lag8": r["controls"][k]["retention"]["8"]["cue_h_l2"]["mean"]}
            for k in ("factual", "swapped", "neutral")}
        for r in intervention}
    pilot = source("frozen_memory_integration_20261126/cage2/results.json")
    evidence["pilot"] = {
        "behavioral_invariants": {k: v for k, v in pilot["behavioral_invariants"].items() if k != "module_hashes"},
        "test_episodes": len(pilot["split"]["test"]),
        "hidden_session_accuracy": {k: v["linear"]["balanced_accuracy"] for k, v in pilot["hidden_session_probes"].items()},
        "paired_history": pilot["hidden_session_paired_deltas"]["observation_time_h - observation_time"],
        "actor_history_sensitivity": pilot["history_readout_sensitivity"],
    }
    history = source("frozen_history_baselines_20261126/history/results.json")
    evidence["return_forecast"] = {k: history[k] for k in ("episode_counts", "history_steps", "time_lookup", "representations")}
    final = source("cage2_final_history_20261126/cage2-final/results.json")
    evidence["common_support"] = {k: final[k] for k in (
        "study_status", "test_opened", "episodes_replayed", "native_steps", "independent_pairs")}
    evidence["common_support"]["coverage"] = {
        k: {"eligible": v["coverage"]["eligible"]["frames"],
            "matchable": v["coverage"]["with_exact_opposite_episode"]["frames"],
            "pairs": v["coverage"]["qualifying_episode_disjoint_pairs"]}
        for k, v in final["cohorts"].items()}
    evidence["source_sha256"] = sources
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_checkout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(collect(args.main_checkout), indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
