"""Frozen T-maze cue diagnostics for source revision 4746126."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_suffix(".json")
SPEC = json.loads(CONFIG.read_text())
SOURCE = ROOT / SPEC["source_worktree"]
if subprocess.check_output(["git", "-C", SOURCE, "rev-parse", "HEAD"], text=True).strip() != SPEC["source_revision"]:
    raise RuntimeError("Unexpected model source revision")
if subprocess.check_output(["git", "-C", SOURCE, "status", "--porcelain"], text=True).strip():
    raise RuntimeError("Model source worktree must be clean")
sys.path[:0] = [str(SOURCE), str(SOURCE / "environments/cage2")]

import numpy as np
import torch
import torch.nn.functional as F

from dreamerrl.eval.location_probe import load_frozen_agent
from dreamerrl.models.glyph_model import GlyphDecoder, glyph_ids
from dreamerrl.models.world_model import WorldModelState
from dreamerrl.replay_buffer.replay_buffer import ReplayBuffer
from dreamerrl.training.core.world_model_update import observe_batch
from latent_probe import episode_split
from experiment_supervisor import CHECKPOINT_EXIT, Job, supervise
from thermal_stop import REQUEST, STATUS, read_record

LAGS = SPEC["lags"]
CUE_CELL = 7
AUDIT_CHUNK = 1000


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False) + "\n")
    tmp.replace(path)


def trajectory_inputs(path):
    records = json.loads(path.read_text())
    scenarios = {}
    ids, labels = None, None
    for name in ("paired", "neutral", "identical", "irrelevant"):
        rows = records[name]
        keys = np.array([(r["seed"], r["cue"]) for r in rows])
        if ids is None:
            ids, labels = keys.T
        assert np.array_equal(keys, np.stack((ids, labels), axis=1))
        assert all([s["lag"] for s in r["trace"]] == list(range(9)) for r in rows)
        assert all([s["action"] for s in r["trace"][:-1]] == [0] * 8 for r in rows)
        assert all(not s["terminal"] and s["reward"] == 0 for r in rows for s in r["trace"][:-1])
        assert all(r["trace"][-1]["terminal"] and r["trace"][-1]["reward"] ==
                   (1 if r["trace"][-1]["action"] == r["cue"] else -1) for r in rows)
        scenarios[name] = torch.tensor([np.asarray([s["glyphs"] for s in r["trace"]]).reshape(9, 56).tolist()
                                        for r in rows], dtype=torch.float32)
    raw = scenarios["paired"]
    assert all(ids[::2] == ids[1::2]) and all(labels[::2] == 0) and all(labels[1::2] == 1)
    assert torch.equal(raw[::2, 1:], raw[1::2, 1:])
    assert torch.equal(raw[:, 0, CUE_CELL], torch.from_numpy(labels + 1).float())
    for name, cell in (("neutral", CUE_CELL), ("irrelevant", 15)):
        difference = torch.nonzero(raw != scenarios[name])
        assert len(difference) == len(raw) and torch.all(difference[:, 1] == 0) and torch.all(difference[:, 2] == cell)
    assert torch.equal(raw, scenarios["identical"])
    assert torch.equal(scenarios["neutral"][::2], scenarios["neutral"][1::2])
    return scenarios, ids, labels


def summary(values, bootstrap):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=values.mean().item(),
                ci95=np.quantile(values[bootstrap].mean(axis=1), [.025, .975]).tolist(),
                by_level=values.tolist())


def distances(x, bootstrap):
    x = x.astype(np.float64, copy=False)
    delta = x[1::2] - x[::2]
    magnitude = np.linalg.norm(delta, axis=-1)
    baseline = (np.linalg.norm(x[::2], axis=-1) + np.linalg.norm(x[1::2], axis=-1)) / 2
    rel = magnitude / np.maximum(baseline, 1e-30)
    return dict(l2=summary(magnitude.mean(axis=1), bootstrap),
                relative_l2=summary(rel.mean(axis=1), bootstrap),
                max_abs=float(np.abs(delta).max()),
                exact_equal_fraction=float(np.mean(magnitude == 0)))


def kernels(x, fit, nonlinear):
    # Scale globally to preserve weak absolute cue effects and avoid amplifying
    # individual nearly constant dimensions. Center and scale on fit only.
    x = x.astype(np.float64)
    x -= x[fit].mean(axis=0)
    scale = np.sqrt(np.mean(np.sum(x[fit] ** 2, axis=1)))
    x /= scale if scale else 1
    gram = x @ x[fit].T
    if not nonlinear:
        return [(1.0, gram)]
    d2 = np.maximum(np.sum(x * x, axis=1)[:, None] + np.sum(x[fit] ** 2, axis=1) - 2 * gram, 0)
    fit_d2 = d2[fit]
    # Exclude numerical zero distances when choosing the fit-only bandwidth.
    positive = fit_d2[fit_d2 > 1e-12]
    median = float(np.median(positive)) if positive.size else 1.
    return [(m, np.exp(-d2 / (2 * median * m))) for m in SPEC["rbf_bandwidth_multipliers"]]


def probe(x, split, bootstrap, null, nonlinear):
    # All stochastic draws for a level stay in one fold. Fit on one prespecified
    # draw per episode, score validation and test over the complete draw set.
    n, repeats, width = x.shape
    fit = np.flatnonzero(split["fit"]) * repeats
    val = np.flatnonzero(split["validation"])
    test = np.flatnonzero(split["test"])
    target = np.tile([-1., 1.], n // 2)
    best = None
    for bandwidth, kernel in kernels(x.reshape(-1, width), fit, nonlinear):
        train = kernel[fit]
        # Center the kernel on the fit set, including cross-kernel rows.
        kernel = kernel - kernel.mean(axis=1, keepdims=True) - train.mean(axis=0) + train.mean()
        for penalty in reversed(SPEC["ridge_penalties"]):
            inverse = np.linalg.solve(kernel[fit] + penalty * np.eye(len(fit)), np.eye(len(fit)))
            predict = (kernel @ (inverse @ target[fit // repeats])).reshape(n, repeats)
            score = np.mean((predict[val] >= 0) == (target[val, None] > 0))
            if best is None or score > best[0]:
                best = (score, bandwidth, penalty, predict)
    score, bandwidth, penalty, predictions = best
    test_scores = predictions[test].reshape(-1, 2, repeats)
    correct = (test_scores >= 0) == np.array([False, True])[None, :, None]
    accuracy_by_draw = correct.mean(axis=1)
    acc = accuracy_by_draw.mean(axis=1)
    swapped = ((test_scores >= 0) != np.array([False, True])[None, :, None]).mean(axis=(1, 2))
    margin = test_scores[:, 1] - test_scores[:, 0]
    concordance = ((margin > 0) + .5 * (margin == 0)).mean(axis=1)
    null_acc = np.where(null, swapped, acc).mean(axis=1)
    resampled_accuracy = None
    if repeats > 1:
        rng = np.random.default_rng(SPEC["sampling_seed"])
        resampled_draws = rng.integers(repeats, size=(len(bootstrap), len(acc), repeats))
        samples = accuracy_by_draw[bootstrap[:, :, None], resampled_draws]
        resampled_accuracy = np.quantile(samples.mean((1, 2)), [.025, .975]).tolist()
    return dict(accuracy=summary(acc, bootstrap), accuracy_by_draw=accuracy_by_draw.tolist(),
                level_and_draw_bootstrap_ci95=resampled_accuracy, paired_concordance=summary(concordance, bootstrap),
                paired_score_margin=summary(margin.mean(axis=1), bootstrap),
                validation_accuracy=float(score), penalty=penalty, bandwidth=bandwidth,
                shuffled_test_label_null=dict(repeats=len(null_acc), mean=float(null_acc.mean()),
                   interval95=np.quantile(null_acc, [.025, .975]).tolist(),
                   p_one_sided=float((1 + np.sum(null_acc >= acc.mean() - 1e-12)) / (1 + len(null_acc)))),
                test_episodes=len(test), draws_per_episode=repeats)


def put(features, name, lag, tensor, n, draws):
    features[name, lag] = tensor.reshape(n, draws, -1).cpu().numpy()


def logits_and_preprocessing(world, obs, h, embed, check):
    # Observe the inputs to Posterior.fc_logits independently of the public
    # logits field, which returns log(unimix probabilities).
    logits = world.posterior.fc_logits(F.silu(world.posterior.fc1(torch.cat((h, embed), -1))))
    prep = glyph_ids(obs, world.encoder.cells, world.encoder.vocab_size)
    if check:
        embedded_glyphs = world.encoder.embedding(prep).flatten(1)
        return logits, prep, embedded_glyphs
    return logits, prep, None


@torch.inference_mode()
def filter_prefix(agent, raw, draw_noise=None, *, detailed=False, imagined=False):
    world = agent.world
    assert isinstance(world.decoder, GlyphDecoder)
    n, steps, width = raw.shape
    draws = 1 if draw_noise is None else draw_noise.shape[2]
    world.posterior.deterministic_latent_for_tests = True
    state = world.init_state(n * draws)
    features, reconstruction, latent = {}, {}, {}
    for lag in range(steps):
        obs = raw[:, lag]
        embed = world.encoder(obs)
        expand = lambda x: x.repeat_interleave(draws, dim=0)
        action = torch.zeros(n * draws, 2)
        if lag:
            action[:, 0] = 1
        state = world.filter_step(state, expand(embed), action, torch.full((n * draws,), lag == 0))
        q = state.post_stats["probs"]
        if draw_noise is not None:
            # Gumbel-max gives exact posterior marginals. Sharing the noise
            # within an opposite-cue pair isolates sensitivity from RNG noise.
            indices = (q.log() + draw_noise[lag].reshape_as(q)).argmax(-1)
            state.z = F.one_hot(indices, q.shape[-1]).to(q.dtype)
        if lag not in LAGS:
            continue
        put(features, "h", lag, state.h, n, draws)
        put(features, "z", lag, state.z, n, draws)
        put(features, "hz", lag, torch.cat((state.h, state.z.flatten(1)), dim=1), n, draws)
        put(features, "probabilities", lag, q, n, draws)
        if draw_noise is None or detailed:
            logits, prep, lookup = logits_and_preprocessing(world, obs, state.h, expand(embed), check=True)
            for name, tensor in (("raw", obs), ("preprocessed", prep), ("glyph_lookup", lookup),
                                 ("encoder", embed)):
                if draw_noise is None:
                    put(features, name, lag, tensor, n, 1)
            for name, tensor in (("logits", logits), ("probabilities_before_unimix", logits.softmax(-1)),
                                 ("mixed_log_probabilities", state.post_stats["logits"])):
                put(features, name, lag, tensor, n, draws)
        if detailed or draw_noise is None:
            prior = world.prior(state.h)["probs"].double()
            q64 = q.double()
            posterior_kl = (q64 * (q64.log() - prior.log())).sum(-1)
            mean_q = q64.mean(0)
            conditional = -(q64 * q64.log()).sum(-1)
            information = -(mean_q * mean_q.log()).sum(-1) - conditional.mean(0)
            pair = q64.reshape(n, draws, *q.shape[1:])
            a, b = pair[::2], pair[1::2]
            middle = (a+b)/2
            js = ((a * (a.log() - middle.log())).sum(-1) + (b * (b.log() - middle.log())).sum(-1)) / 2
            kl_ab = (a * (a.log() - b.log())).sum((-2, -1))
            kl_ba = (b * (b.log() - a.log())).sum((-2, -1))
            # Product distribution bounds hold for a single draw at reset.
            # At later lags this is conditional on the two filtered histories.
            tv_upper = (torch.minimum(kl_ab, kl_ba).clamp_min(0) / 2).sqrt().clamp_max(1)
            latent[lag] = dict(entropy_per_factor_nats=conditional.reshape(n, draws, -1).mean((1, 2)).tolist(),
                kl_per_state_nats=posterior_kl.reshape(n, draws, -1).sum(-1).mean(1).tolist(),
                factor_information_nats=information.tolist(),
                active_factors=int((information > SPEC["active_latent_nats"]).sum()),
                marginal_perplexity=(-(mean_q * mean_q.log()).sum(-1)).exp().tolist(),
                occupied_categories_by_factor=state.z.sum(0).ne(0).sum(-1).tolist(),
                factor_entropy_nats=conditional.mean(0).tolist(),
                factor_pair_js_nats=js.mean((0, 1)).tolist(),
                factor_pair_js_by_level_nats=js.mean(1).tolist(),
                paired_mean_factor_js_nats=js.mean((1, 2)).tolist(),
                pair_product_kl_nats=((kl_ab + kl_ba) / 2).mean(1).tolist(),
                pair_product_tv_upper=tv_upper.mean(1).tolist())
        if detailed:
            # Evaluate every mode and the first four prespecified draws.
            count = min(draws, 4)
            selected = torch.arange(n * draws).reshape(n, draws)[:, :count].flatten()
            probs = world.decoder(state.h[selected], state.z[selected])
            target = obs.repeat_interleave(count, dim=0)
            predicted = world.decoder.predict(probs)
            match = predicted == target.long()
            loss = world.decoder.cell_loss(probs, target)
            reconstruction[lag] = dict(cue_accuracy=match[:, CUE_CELL].float().reshape(n, count).mean(1).tolist(),
                all_accuracy=match.float().reshape(n, -1).mean(1).tolist(),
                cue_nll=loss[:, CUE_CELL].reshape(n, count).mean(1).tolist(),
                all_nll=loss.reshape(n, -1).mean(1).tolist(),
                cue_predicted_ids=predicted[:, CUE_CELL].reshape(n, count).tolist())
        if lag == 0:
            assert torch.equal(state.h[::2 * draws], state.h[draws::2 * draws])
            if imagined:
                imagined_state = WorldModelState(state.h, state.z)
                imagined_features = {}
                for distance in range(1, 9):
                    act = F.one_hot(torch.zeros(n * draws, dtype=torch.long), 2).float()
                    imagined_state = world.imagine_step(imagined_state, agent.actor, action=act,
                                                       deterministic_imagination=True)
                    if distance in LAGS:
                        put(imagined_features, "imagined_h", distance, imagined_state.h, n, draws)
                        put(imagined_features, "imagined_z", distance, imagined_state.z, n, draws)
                        put(imagined_features, "imagined_hz", distance,
                            torch.cat((imagined_state.h, imagined_state.z.flatten(1)), -1), n, draws)
                        put(imagined_features, "imagined_prior", distance,
                            imagined_state.prior_stats["probs"], n, draws)
                features.update(imagined_features)
    return features, reconstruction, latent


@torch.inference_mode()
def replay_audit(agent, payload):
    replay = payload["replay"]
    episodes = replay["episodes"]
    lengths = dict(Counter(len(ep["obs"]) for ep in episodes))
    assert set(lengths) == {10}
    cues = np.zeros(2, dtype=int)
    by_cue_action = np.zeros((2, 2), dtype=int)
    for start in range(0, len(episodes), AUDIT_CHUNK):
        chunk = episodes[start:start + AUDIT_CHUNK]
        obs, action, reward, first, done, terminal = (
            torch.stack([e[key] for e in chunk]) for key in ("obs", "action", "reward", "is_first", "done", "is_terminal")
        )
        cue = obs[:, 0, CUE_CELL].long() - 1
        assert torch.all((cue == 0) | (cue == 1))
        assert torch.all(obs[:, 1:, CUE_CELL] == 0)
        assert torch.all(first[:, 0]) and not first[:, 1:].any()
        assert torch.all(done[:, -1]) and not done[:, :-1].any()
        assert torch.all(terminal[:, -1]) and not terminal[:, :-1].any()
        assert not reward[:, :-1].any()
        assert torch.equal(reward[:, -1], torch.where(action[:, -1] == cue, 1., -1.))
        cues += np.bincount(cue.numpy(), minlength=2)
        by_cue_action += np.bincount((2 * cue + action[:, -1]).numpy(), minlength=4).reshape(2, 2)
    seq, burn = replay["seq_len"], replay["burn_in"]
    windows = [dict(start=i, targets=list(range(i, i + seq)), burn_in=list(range(max(0, i-burn), i)))
               for i in range(11 - seq)]
    buffer = ReplayBuffer(agent.cfg.train, replay["obs_dim"], replay["action_dim"], torch.device("cpu"))
    buffer.load_state_dict(replay)
    batch = buffer.sample(64, seed=SPEC["sampling_seed"])
    world = agent.world
    world.posterior.deterministic_latent_for_tests = True
    posts = observe_batch(world, batch)
    errors = []
    for row in range(64):
        valid = batch["burn_in_valid"][row]
        obs = torch.cat((batch["burn_in_obs"][row, valid], batch["obs"][row]))
        actions = torch.cat((batch["burn_in_action"][row, valid], batch["action"][row]))
        assert obs[0, CUE_CELL] in (1, 2)
        state = world.init_state(1)
        for t, (glyphs, action) in enumerate(zip(obs, actions)):
            state = world.filter_step(state, world.encoder(glyphs[None]), F.one_hot(action.long()[None], 2).float(),
                                      torch.tensor([t == 0]))
            if t >= int(valid.sum()):
                expected = posts[t - int(valid.sum())]
                errors.append(float((state.h[0] - expected["h"][row]).abs().max()))
                assert torch.equal(state.z[0], expected["z"][row])
    tail = batch["reward"][:, -1].abs() == 1
    assert torch.all(batch["burn_in_obs"][tail, -2, CUE_CELL] != 0)
    return dict(episode_lengths=lengths, cue_counts=cues.tolist(), cue_by_terminal_action=by_cue_action.tolist(),
                window_starts_uniform=windows, terminal_target_sample_count=int(tail.sum()),
                sampled_windows=64, h_max_abs_manual_prefix_error=max(errors))


def audit_summaries(values, pair_mask, bootstrap, paired_keys=()):
    result = {}
    for lag, metrics in values.items():
        result[lag] = {}
        for name, value in metrics.items():
            array = np.asarray(value)
            if name == "factor_pair_js_by_level_nats":
                samples = array[pair_mask]
                result[lag][name] = samples.tolist()
                result[lag]["factor_pair_js_test_ci95_nats"] = np.quantile(samples[bootstrap].mean(1), [.025, .975], axis=0).T.tolist()
            elif name in paired_keys:
                result[lag][name] = summary(array[pair_mask], bootstrap)
            elif name in ("entropy_per_factor_nats", "kl_per_state_nats", "cue_accuracy", "all_accuracy",
                          "cue_nll", "all_nll"):
                result[lag][name] = summary(array.reshape(-1, 2)[pair_mask].mean(1), bootstrap)
            else:
                result[lag][name] = value
    return result


def state_digest(payload):
    digest = hashlib.sha256()
    for key, tensor in sorted(payload["world"].items()):
        digest.update(key.encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def analyze(path, seed, update, scenarios, ids, split, groups, *, detailed, smoke, final):
    fingerprint = sha(path)
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    cfg = payload["config"]
    assert payload["completed_updates"] == update and cfg.train.seed == seed and cfg.env.seed == seed
    assert cfg.env.env_id == "MiniHack-CuedTMaze-v0" and not cfg.ac.hierarchical and not cfg.env.finite_horizon
    assert not cfg.train.use_amp and not cfg.train.deterministic_env
    config_path = path.parent / "config.json"
    checkpoint_world = state_digest(payload)
    if final and not smoke:
        complete = json.loads((path.parent / "complete.json").read_text())
        assert complete["checkpoint_sha256"] == fingerprint
        assert complete["env_steps"] == payload["total_env_steps"] == 288000
        other = path.with_name("update_4000.pt")
        numbered = torch.load(other, map_location="cpu", mmap=True, weights_only=False)
        assert numbered["completed_updates"] == 4000 and numbered["total_env_steps"] == 288000
        assert state_digest(numbered) == checkpoint_world
        numbered_record = dict(path=str(other), sha256=sha(other), world_sha256=checkpoint_world)
        del numbered
    else:
        numbered_record = None
    print(f"CPU {path.parent.name}/{path.name} seed={seed} update={update}", flush=True)
    agent = load_frozen_agent(path)
    audit = replay_audit(agent, payload) if final and not smoke else None
    steps = payload["total_env_steps"]
    del payload
    n = len(ids)
    pair_mask = split["test"][::2]
    num_test = int(pair_mask.sum())
    rng = np.random.default_rng(SPEC["sampling_seed"])
    bootstrap = rng.integers(num_test, size=(SPEC["bootstrap_repeats"], num_test))
    null = rng.integers(2, size=(SPEC["shuffle_repeats"], num_test)).astype(bool)
    generator = torch.Generator().manual_seed(SPEC["sampling_seed"])
    uniform = torch.rand((9, n // 2, SPEC["sample_repeats"],
                          agent.world.latent.num_classes, agent.world.latent.stoch_size), generator=generator)
    noise = -(-uniform.clamp_min(1e-7).log()).log()
    noise = noise.repeat_interleave(2, dim=1)
    output = dict(path=str(path), sha256=fingerprint, world_sha256=checkpoint_world, config_path=str(config_path),
                  config_sha256=sha(config_path), seed=seed, update=update, smoke=smoke, final=final,
                  env_steps=steps, numbered_final=numbered_record, replay=audit, conditions={}, split=groups)
    for cond, randomness in (("mode", None), ("sample", noise)):
        features, recon, latent = filter_prefix(agent, scenarios["paired"], randomness,
                                               detailed=detailed, imagined=detailed)
        # Synthetic linear and XOR positive controls exercise both probe pipelines.
        if cond == "mode" and detailed:
            y = (np.arange(n) % 2) * 2 - 1
            nuisance = np.empty(n)
            for mask in split.values():
                nuisance[mask] = np.tile(np.repeat([-1., 1.], 2), mask.sum() // 4)
            features["synthetic_linear", 0] = np.stack((y, nuisance), axis=-1)[:, None, :]
            features["synthetic_xor", 0] = np.stack((y * nuisance, nuisance), axis=-1)[:, None, :]
        condition = output["conditions"][cond] = dict(probes={}, controls={}, separation={})
        mask = split["test"]
        for (name, lag), x in sorted(features.items()):
            tag = f"{name}/{lag}"
            condition["separation"][tag] = distances(x[mask], bootstrap)
            if not detailed and cond == "sample" and (lag not in (0, 8) or name not in ("h", "z", "hz", "probabilities")):
                continue
            condition["probes"][tag] = {family: probe(x, split, bootstrap, null, family == "rbf")
                                         for family in ("linear", "rbf")}
        if detailed and cond == "mode":
            for name, family in (("synthetic_linear", "linear"), ("synthetic_xor", "rbf")):
                assert condition["probes"][f"{name}/0"][family]["accuracy"]["mean"] >= .95
            for control in ("identical", "neutral", "irrelevant"):
                checked, _, _ = filter_prefix(agent, scenarios[control])
                control_results = condition["controls"][control] = {}
                for (name, lag), values in checked.items():
                    if control == "neutral":
                        x = values[mask]
                    else:
                        left = features[name, lag][mask][::2]
                        x = np.stack((left, values[mask][::2]), axis=1).reshape(-1, *left.shape[1:])
                    result = distances(x, bootstrap)
                    if control != "irrelevant":
                        assert result["max_abs"] == 0
                    control_results[f"{name}/{lag}"] = result
        condition["latent"] = audit_summaries(latent, pair_mask, bootstrap,
                                              ("pair_product_kl_nats", "pair_product_tv_upper", "paired_mean_factor_js_nats"))
        condition["reconstruction"] = audit_summaries(recon, pair_mask, bootstrap)
    assert sha(path) == fingerprint
    return output


def checkpoint_plan(baseline):
    for seed, worker in zip(SPEC["seeds"], sorted(baseline.glob("job_*")), strict=True):
        folder = worker / "results" / f"flat_seed{seed}"
        yield f"seed{seed}_4000", folder / "latest.pt", seed, 4000, True, False
    for run in SPEC["smoke_runs"]:
        folder = ROOT / run
        cfg = json.loads((folder / "config.json").read_text())
        evaluation = json.loads((folder / "evaluation.json").read_text())
        seed, update = cfg["train"]["seed"], evaluation["update"]
        yield f"smoke_seed{seed}_{update}", folder / "latest.pt", seed, update, True, True
    for update in range(0, 4000, 200):
        for index, seed in enumerate(SPEC["seeds"], start=1):
            folder = baseline / f"job_{index:03d}" / "results" / f"flat_seed{seed}"
            name = "initial.pt" if update == 0 else f"update_{update}.pt"
            yield f"seed{seed}_{update:04d}", folder / name, seed, update, False, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--worker-status", type=Path)
    args = parser.parse_args()
    if args.supervise:
        return supervise_run(args.output.resolve())
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SPEC["sampling_seed"])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = out / "checkpoints"
    records.mkdir(exist_ok=True)
    baseline = ROOT / SPEC["baseline_run"]
    trajectories = baseline / "job_001/results/flat_seed19/heldout_trajectories.json"
    scenarios, ids, _ = trajectory_inputs(trajectories)
    split, ids_by_split = episode_split(ids, SPEC["split_seed"])
    groups = {name: levels.tolist() for name, levels in ids_by_split.items()}
    assert sum(map(len, groups.values())) == len(set().union(*map(set, groups.values()))) == len(ids) // 2
    old_splits = []
    historical = {}
    for index, seed in enumerate(SPEC["seeds"], start=1):
        folder = baseline / f"job_{index:03d}/results/flat_seed{seed}"
        other_scenarios, other_ids, _ = trajectory_inputs(folder / "heldout_trajectories.json")
        assert np.array_equal(ids, other_ids)
        assert all(torch.equal(scenarios[name], other_scenarios[name]) for name in scenarios)
        old = json.loads((folder / "memory_report.json").read_text())
        old_splits.append(old["split"])
        historical[seed] = dict(path=str(folder / "memory_report.json"), checkpoint_sha256=old["checkpoint_sha256"],
                                natural_accuracy=old["natural_test_accuracy"], counterfactual=old["counterfactual"]["means"],
                                retention=old["retention"])
    assert all(groups == prior for prior in old_splits)
    report_source = Path(__file__).with_name("tmaze_bottleneck_report.py")
    launcher = Path(__file__).with_name("run_tmaze_bottleneck.sh")
    source_hashes = {str(p): sha(p) for p in (Path(__file__).resolve(), CONFIG, report_source, launcher)}
    manifest = dict(pid=os.getpid(), tmux_session=os.environ.get("DIAGNOSTIC_TMUX"), argv=sys.argv,
                    device="cpu", benchmark_training_revision=SPEC["source_revision"],
                    root_revision=subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip(),
                    source_sha256=source_hashes, lockfile=str(SOURCE / "uv.lock"),
                    lock_sha256=sha(SOURCE / "uv.lock"),
                    dependencies={p: importlib.metadata.version(p) for p in ("torch", "numpy", "gymnasium", "minihack", "nle", "matplotlib")},
                    python=sys.version, trajectories=str(trajectories), trajectories_sha256=sha(trajectories),
                    config=SPEC, split=groups, historical=historical, log=str(out / "console.log"),
                    shell_command=str(out / "launch_command"), worker_pid=str(out / "worker_pid"),
                    exit_status=str(out / "exit_status"), checkpoints=str(records), report=str(out / "failure_mode_report.md"))
    write(out / "execution_manifest.json", manifest)
    plan = list(checkpoint_plan(baseline))
    assert len(plan) == len({row[0] for row in plan}) == 65
    assert all(path.is_file() for _, path, *_ in plan)
    if channel := os.environ.get("DIAGNOSTIC_READY_CHANNEL"):
        subprocess.run(["tmux", "wait-for", "-S", channel], check=True)
    for key, path, seed, update, detailed, smoke in plan:
        request = read_record(args.worker_status / REQUEST) if args.worker_status else None
        if request is not None:
            write(args.worker_status / STATUS, dict(request_id=request["request_id"], reason=request["reason"],
                  checkpoint_status="saved", checkpoint=str(records)))
            return CHECKPOINT_EXIT
        destination = records / f"{key}.json"
        if destination.exists():
            previous = json.loads(destination.read_text())
            if previous["source_sha256"] == source_hashes and previous["sha256"] == sha(path):
                print(f"Reuse {key}", flush=True)
                continue
        record = analyze(path, seed, update, scenarios, ids, split, groups, detailed=detailed, smoke=smoke, final=detailed)
        record["source_sha256"] = source_hashes
        write(destination, record)
    manifest["results"] = [str(records / f"{key}.json") for key, *_ in plan]
    write(out / "execution_manifest.json", manifest)
    if args.worker_status:
        write(args.worker_status / "complete.json", dict(checkpoint=str(records)))
    return 0


def supervise_run(out):
    from tmaze_bottleneck_report import render
    attempt = out / ("supervisor_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    worker = attempt / "job_001" / "results" / "diagnostics"
    worker.mkdir(parents=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), str(out), "--worker-status", str(worker)]
    watchdog = [sys.executable, "-u", str(SOURCE / "environments/cage2/thermal_watchdog.py"),
                "--supervisor-control", str(attempt), "--log", str(attempt / "thermal.jsonl"),
                "--gpu", "0", "--warn-c", "80", "--stop-c", "87", "--sustain-seconds", "30"]
    jobs = [(1, Job(worker, command, CONFIG, "frozen_cpu_diagnostics", SPEC["sampling_seed"]))]
    write(out / "launch.json", dict(supervisor_pid=os.getpid(), attempt=str(attempt),
                                   tmux_session=os.environ.get("DIAGNOSTIC_TMUX"), command=command))
    status = 1
    completion = dict(state="incomplete", attempt=str(attempt), tmux_session=os.environ.get("DIAGNOSTIC_TMUX"))
    try:
        result, status = supervise(attempt, jobs, watchdog, concurrency=1, readiness_timeout=60,
                                   checkpoint_timeout=300, wall_seconds=3600)
        completion.update(supervisor=result, supervisor_exit_code=status)
        worker_state = result["jobs"][0]["state"]
        if status == 0 and worker_state == "completed":
            completion["state"] = "successful"
        elif worker_state in ("failed", "launch_failed"):
            completion["state"] = "failed"
    except Exception as exc:
        completion.update(state="failed", supervisor_error=repr(exc))
    try:
        render(out, completion)
    except Exception as exc:
        completion.update(state="failed", report_error=repr(exc))
        (out / "failure_mode_report.md").write_text("# T-maze representation diagnostics\n\n"
            f"Report generation failed. See completion_manifest.json and preserved checkpoint results. {exc!r}\n")
        status = 1
    completion.update(exit_code=status, report=str(out / "failure_mode_report.md"),
                      artifacts=[str(p) for p in sorted(out.iterdir())],
                      finished_utc=datetime.now(timezone.utc).isoformat())
    write(out / "completion_manifest.json", completion)
    return status


if __name__ == "__main__":
    sys.exit(main())
