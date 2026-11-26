"""Inject reset states selected on fit levels into frozen T-maze histories."""
from pathlib import Path
import json

from gymnasium.spaces import Box
import numpy as np
import torch
import torch.nn.functional as F

from dreamerrl.eval.memory.dreamer import CheckpointSpec, DreamerAdapter, categorical_state, file_sha256, verified_payload
from dreamerrl.eval.memory.interfaces import Episode, Labels, paired_episodes
from dreamerrl.eval.memory.presentation import MIST, NAVY, ORANGE, TEAL, plt, save_figure, style, write_json
from dreamerrl.eval.memory.statistics import binary_probe, bootstrap_mean, l2_distance, paired_accuracy
from dreamerrl.eval.memory.trajectory import filter_trajectory

CUE_CELL = 7


class TMazeArchive:
    def __init__(self, path):
        data = json.loads(path.read_text())
        self.rows = {}
        for condition in ("paired", "neutral", "identical", "irrelevant"):
            self.rows[condition] = {}
            for row in data[condition]:
                group, cue = row["seed"], row["cue"]
                trace = row["trace"]
                if [f["lag"] for f in trace] != list(range(9)):
                    raise ValueError("Unexpected T-maze lag or horizon")
                obs = torch.tensor([f["glyphs"] for f in trace]).flatten(1)
                actions = torch.tensor([f["action"] for f in trace])
                rewards = torch.tensor([f["reward"] for f in trace])
                if actions[:-1].any() or rewards[:-1].any() or rewards[-1] != (1 if actions[-1] == cue else -1):
                    raise ValueError("Unexpected stem action or branch reward")
                if [f["terminal"] for f in trace] != [False] * 8 + [True]:
                    raise ValueError("Unexpected episode boundary")
                self.rows[condition][group, cue] = Episode((group, cue), group, obs, actions, rewards, True, False)

    def collect(self, episode_ids, condition="paired"):
        return [self.rows[condition][key] for key in episode_ids]

    def labels(self, episode_id):
        return Labels(np.array(episode_id[1]), np.array(True))

    def model_input(self, observations):
        return observations.float()

    def matched_arrays(self, levels):
        ids = [(level, cue) for level in levels for cue in (0, 1)]
        paired, neutral, identical, irrelevant = [self.collect(ids, condition) for condition in self.rows]
        for i in range(0, len(paired), 2):
            a, b = paired[i:i+2]
            paired_episodes(a, b, 0)
            if not torch.equal((a.observations != b.observations).nonzero(), torch.tensor([[0, CUE_CELL]])):
                raise ValueError("The cue must change exactly one cell")
            if not torch.equal(neutral[i].observations, neutral[i+1].observations):
                raise ValueError("Neutral cue histories differ")
        for i, row in enumerate(paired):
            for other, expected in ((neutral[i], [[0, CUE_CELL]]), (irrelevant[i], [[0, 15]]), (identical[i], [])):
                paired_episodes(row, other, 0)
                if (row.observations != other.observations).nonzero().tolist() != expected:
                    raise ValueError("Archived intervention differs outside its designed input")
        return {name: self.model_input(torch.stack([e.observations for e in cases]))
                for name, cases in zip(self.rows, (paired, neutral, identical, irrelevant), strict=True)}, torch.stack(
                    [e.actions for e in paired])


def select_reset_states(q, neutral, draws, rng):
    """Choose hard states using fit posterior log likelihood ratios only."""
    q = q.double().reshape(-1, 2, *q.shape[-2:])
    # Sample from each fit posterior. Choose the draw with the largest signed
    # mean paired log-likelihood ratio for that class, with fixed tie order.
    log_ratio = (q[:, 1].log() - q[:, 0].log()).mean(0)
    states, score = [], []
    for cue in (0, 1):
        candidates = categorical_state(q[:, cue, None].expand(-1, draws, -1, -1), "sample", rng=rng)
        scores = (candidates * log_ratio).sum((-2, -1)).flatten()
        index = (scores if cue else -scores).argmax()
        states.append(candidates.flatten(0, 1)[index].float())
        score.append(float(scores[index]))
    neutral_z = categorical_state(neutral.mean(0), "argmax")
    return torch.stack(states), neutral_z, score


def condition_metrics(model, obs, actions, z, labels, mode_split, seed, prior):
    states, features = filter_trajectory(model, obs, actions, initial_z=z)
    last = states[-1]
    actor = model.readouts(last)
    options = [torch.full((len(obs),), i) for i in (0, 1)]
    margins = [model.reward(model.imagine_step(last, F.one_hot(action, 2).float())) for action in options]
    test = mode_split["test"]
    test_count = int(test.sum()) // 2
    probability = actor["action_probability"][:, 1].numpy()[test].reshape(test_count, 2)
    reward = (margins[1] - margins[0]).numpy()[test].reshape(test_count, 2)
    values = actor["value"].numpy()[test].reshape(test_count, 2)
    correct = (actor["action_logits"].argmax(-1).numpy()[test] == labels[test]).reshape(test_count, 2)
    summary = dict(decision_accuracy=bootstrap_mean(correct.mean(1), seed),
                   decision_accuracy_by_level=correct.mean(1).tolist(),
                   right_action_probability=probability.tolist(), reward_margin=reward.tolist(), critic_value=values.tolist(),
                   right_shift=bootstrap_mean(probability[:, 1] - probability[:, 0], seed),
                   reward_shift=bootstrap_mean(reward[:, 1] - reward[:, 0], seed),
                   value_shift=bootstrap_mean(values[:, 1] - values[:, 0], seed), retention={})
    for lag in range(features["h"].shape[1]):
        h = features["h"][:, lag].numpy()
        metrics = {}
        for nonlinear in (False, True):
            scores = binary_probe(h[:, None], labels, mode_split, nonlinear=nonlinear)["scores"]
            accuracy = paired_accuracy(scores[test].reshape(test_count, 2, 1), labels[test].reshape(test_count, 2))
            metrics["rbf" if nonlinear else "linear"] = bootstrap_mean(accuracy, seed)
        h_tensor = features["h"][test, lag]
        base_h = prior[test, lag]
        summary["retention"][str(lag)] = dict(metrics,
            cue_h_l2=bootstrap_mean(l2_distance(h_tensor[::2], h_tensor[1::2]).numpy(), seed),
            intact_h_l2=bootstrap_mean(l2_distance(h_tensor, base_h).reshape(-1, 2).mean(1).numpy(), seed))
    return summary, features["h"]


@torch.inference_mode()
def run(root, spec, out, should_stop):
    accepted = root / spec["accepted"]
    completion = json.loads((accepted / "completion_manifest.json").read_text())
    if completion["state"] != "successful":
        raise ValueError("Expected accepted completed T-maze diagnostics")
    results = []
    for seed in spec["seeds"]:
        if should_stop():
            return False
        source = accepted / "checkpoints" / f"seed{seed}_4000.json"
        record = json.loads(source.read_text())
        path = Path(record["path"]).resolve()
        if not path.is_relative_to(root / "runs"):
            raise ValueError("Checkpoint path is outside the main run archive")
        output = out / f"seed{seed}.json"
        if output.exists():
            previous = json.loads(output.read_text())
            if previous["checkpoint_sha256"] != record["sha256"] or file_sha256(path) != record["sha256"]:
                raise ValueError("Existing intervention record has different checkpoint provenance")
            results.append(previous)
            continue
        archived = path.parent / "heldout_trajectories.json"
        archive = TMazeArchive(archived)
        levels = sum([record["split"][x] for x in ("fit", "validation", "test")], [])
        observations, actions = archive.matched_arrays(levels)
        labels = np.tile([0, 1], len(levels))
        split = {k: np.repeat(np.isin(levels, v), 2) for k, v in record["split"].items()}
        payload = verified_payload(CheckpointSpec(path, record["sha256"], seed, 4000, 288000))
        model = DreamerAdapter.from_payload(payload, Box(0, 10000, (56,), np.float32), 2)
        del payload
        full = observations["paired"]
        zero, first = torch.zeros(len(full), 2), torch.ones(len(full), dtype=torch.bool)
        start = model.initial_state(len(full))
        _, factual = model.filter_step(start, full[:, 0], zero, first)
        _, neutral = model.filter_step(start, observations["neutral"][:, 0], zero, first)
        generator = torch.Generator().manual_seed(spec["rng_seed"] + seed)
        selected, neutral_z, scores = select_reset_states(factual["posterior"][split["fit"]],
            neutral["posterior"][split["fit"]], spec["draws_per_fit_posterior"], generator)
        _, intact = filter_trajectory(model, full, actions)
        control_cases = (
            ("intact", full, None),
            ("factual", full, selected[labels]),
            ("swapped", full, selected[1-labels]),
            ("neutral", observations["neutral"], neutral_z.expand(len(full), -1, -1)),
            ("identical", observations["identical"], None),
            ("irrelevant_marker", observations["irrelevant"], None),
        )
        cases = {}
        for condition, obs, z in control_cases:
            cases[condition], h = condition_metrics(model, obs, actions, z, labels, split, spec["rng_seed"], intact["h"])
            if condition == "identical" and not torch.equal(h, intact["h"]):
                raise ValueError("Identical-history run changed the recurrent states")
            if condition == "neutral" and not torch.equal(h[::2], h[1::2]):
                raise ValueError("Cue-neutral pairs diverged")
        delta = np.asarray(cases["factual"]["decision_accuracy_by_level"]) - cases["swapped"]["decision_accuracy_by_level"]
        value = dict(seed=seed, checkpoint=str(path), checkpoint_sha256=record["sha256"], accepted_record=str(source),
                     accepted_record_sha256=file_sha256(source), trajectory=str(archived), trajectory_sha256=file_sha256(archived),
                     split=record["split"], log_likelihood_ratio=scores,
                     forced_reset_state_indices=selected.argmax(-1).tolist(),
                     reset_factor_hamming=int(selected[0].argmax(-1).ne(selected[1].argmax(-1)).sum()),
                     factual_minus_swapped_accuracy=bootstrap_mean(delta, spec["rng_seed"]), controls=cases)
        write_json(output, value)
        results.append(value)
    write_json(out / "results.json", results)
    render(out, results)
    return True


def render(out, results):
    style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    conditions = (("factual", TEAL, "-"), ("swapped", ORANGE, "--"),
                  ("neutral", MIST, ":"), ("intact", NAVY, "-."))
    for name, color, line in conditions:
        data = np.array([[x["cue_h_l2"]["mean"] for x in r["controls"][name]["retention"].values()]
                         for r in results])
        ax.plot(range(data.shape[1]), data.mean(0), label=name.replace("_", " "), c=color, ls=line)
        ax.fill_between(range(data.shape[1]), data.min(0), data.max(0), color=color, alpha=.1)
    ax.set(xlabel="Actions since reset", ylabel="Paired recurrent-state L2")
    ax.legend(frameon=False)
    save_figure(fig, out / "retention")
    lines = ["# Frozen T-maze reset-state intervention", "",
             "Reset states were selected using only fit-level posterior draws. The same eight recorded stem actions were used throughout.",
             "", "| Seed | Reset factors changed | Factual accuracy | Swapped accuracy | Neutral accuracy | Factual − swapped (95% interval) |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        accuracy = [row["controls"][c]["decision_accuracy"]["mean"] for c in ("factual", "swapped", "neutral")]
        delta = row["factual_minus_swapped_accuracy"]
        lines.append(f"| {row['seed']} | {row['reset_factor_hamming']} | " + " | ".join(f"{a:.3f}" for a in accuracy)
                     + f" | {delta['mean']:.3f} {delta['ci95']} |")
    lines += ["", "The effect estimates concern the chosen artificial reset states. The candidates came from unusually favorable",
              "posterior draws and can be rare under ordinary inference. A branch accuracy difference would demonstrate",
              "sensitivity to the intervention. A null result would apply to this selection and these checkpoints.", "",
              "Recurrent separation, linear and RBF readout, actor probabilities, online critic value, prior-mode branch reward",
              "margin and identical-history and irrelevant-marker controls are available in the per-seed JSON.",
              "Intervals resample held-out level pairs within seed. Figure shading spans training-seed means.", ""]
    (out / "report.md").write_text("\n".join(lines))
