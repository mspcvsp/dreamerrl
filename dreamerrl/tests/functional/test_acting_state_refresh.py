"""Refresh acting recurrence with current weights and the correct live episodes."""

import pytest
import torch
import torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import TimeLimit

from dreamerrl.tests.manual.utils import minihack_config
from dreamerrl.training.trainer import DreamerTrainer


def make_trainer(tmp_path, finite=False, context=-1):
    cfg = minihack_config(0, tmp_path)
    cfg.env.env_id = "popgym-RepeatPreviousEasy-v0"
    cfg.env.max_episode_steps = 6
    cfg.env.finite_horizon = finite
    cfg.train.collect_steps = 2
    cfg.train.acting_context = context
    trainer = DreamerTrainer(cfg)
    vector_env = trainer.env.venv
    assert isinstance(vector_env, SyncVectorEnv)
    first_env = vector_env.envs[0]
    assert isinstance(first_env, TimeLimit)
    first_env._max_episode_steps = 2
    trainer.world.posterior.deterministic_latent_for_tests = True
    trainer.world.prior.deterministic_latent_for_tests = True
    trainer.refresh_acting_state()
    return trainer


@torch.no_grad()
def expected_row(trainer, row, context=-1):
    data = trainer.replay.current_eps[row].data
    state = trainer.world.init_state(1)
    if not data["obs"]:
        return trainer.world.observe_step(
            state,
            trainer.env_state["state"][row : row + 1],
            torch.zeros(1, trainer.action_dim),
            is_first=torch.tensor([True]),
        )["post"]
    start = max(0, len(data["obs"]) - context) if context > 0 else 0
    for t in range(start, len(data["obs"])):
        action = F.one_hot(data["action"][t].long(), trainer.action_dim).float().unsqueeze(0)
        state = trainer.world.observe_step(
            state, data["obs"][t].unsqueeze(0), action, is_first=data["is_first"][t].reshape(1)
        )["post"]
    return state


def close(trainer):
    trainer.env.venv.close()
    trainer.tb.close()


@pytest.mark.parametrize("finite", [False, True])
def test_update_refreshes_asynchronous_rows_and_next_actor_input(tmp_path, monkeypatch, finite):
    trainer = make_trainer(tmp_path, finite)
    try:
        trainer.collect_env_steps()
        assert not trainer.replay.current_eps[0].data["obs"]
        assert len(trainer.replay.current_eps[1].data["obs"]) == 3
        stale = trainer.world_state.h.clone()
        trainer.update_world_model(trainer.replay.sample(2), 0)
        expected = [expected_row(trainer, row) for row in range(2)]
        for row, state in enumerate(expected):
            torch.testing.assert_close(trainer.world_state.h[row], state.h[0])
            torch.testing.assert_close(trainer.world_state.z[row], state.z[0])
        assert not torch.equal(stale, trainer.world_state.h)
        assert trainer.world_state.h.grad_fn is None and trainer.world_state.z.grad_fn is None
        assert trainer.acting_refresh_metrics["context_tokens"] == 4
        assert trainer.acting_refresh_metrics["computed_tokens"] == 4
        assert trainer.acting_refresh_metrics["truncated_rows"] == 0
        original = trainer.actor.act
        inputs = []

        def act(state, **kwargs):
            inputs.append(state.h.clone())
            return original(state, **kwargs)

        monkeypatch.setattr(trainer.actor, "act", act)
        trainer.collect_env_steps()
        torch.testing.assert_close(inputs[0], torch.cat([state.h for state in expected]))
    finally:
        close(trainer)


def test_bounded_refresh_matches_suffix_and_preserves_rng(tmp_path):
    trainer = make_trainer(tmp_path, context=2)
    try:
        trainer.collect_env_steps()
        metrics = trainer.refresh_acting_state()
        assert metrics["context_tokens"] == 3
        assert metrics["truncated_rows"] == 1
        for row in range(2):
            expected = expected_row(trainer, row, context=2)
            torch.testing.assert_close(trainer.world_state.h[row], expected.h[0])
            torch.testing.assert_close(trainer.world_state.z[row], expected.z[0])
        trainer.world.posterior.deterministic_latent_for_tests = False
        trainer.world.prior.deterministic_latent_for_tests = False
        rng, replay_rng = torch.get_rng_state().clone(), trainer.replay.rng.get_state().clone()
        trainer.refresh_acting_state()
        assert torch.equal(torch.get_rng_state(), rng)
        assert torch.equal(trainer.replay.rng.get_state(), replay_rng)
    finally:
        close(trainer)


@pytest.mark.parametrize("skip_step", [False, True])
def test_disabled_refresh_or_skipped_model_update_preserves_state(tmp_path, monkeypatch, skip_step):
    trainer = make_trainer(tmp_path, context=-1 if skip_step else 0)
    try:
        trainer.collect_env_steps()
        old_h, old_z = trainer.world_state.h.clone(), trainer.world_state.z.clone()
        if skip_step:
            monkeypatch.setattr(trainer.model_opt, "step", lambda *args, **kwargs: None)
        trainer.update_world_model(trainer.replay.sample(2), 0)
        torch.testing.assert_close(trainer.world_state.h, old_h, rtol=0, atol=0)
        torch.testing.assert_close(trainer.world_state.z, old_z, rtol=0, atol=0)
        assert trainer.acting_refresh_metrics == {}
    finally:
        close(trainer)


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_starts_fresh_context_and_preserves_saved_policy(tmp_path, legacy):
    trainer = make_trainer(tmp_path, context=2)
    resumed = None
    try:
        trainer.collect_env_steps()
        path = tmp_path / "acting.pt"
        trainer.save_checkpoint(path)
        if legacy:
            payload = torch.load(path, weights_only=False)
            del vars(payload["config"].train)["acting_context"]
            torch.save(payload, path)
        resumed = DreamerTrainer.from_checkpoint(path, cuda=False)
        assert resumed.cfg.train.acting_context == (0 if legacy else 2)
        assert resumed.replay.current_eps == []
        if not legacy:
            metrics = resumed.refresh_acting_state()
            assert metrics["context_tokens"] == 2  # one fresh reset observation per row
            assert metrics["truncated_rows"] == 0
    finally:
        close(trainer)
        if resumed is not None:
            close(resumed)


@pytest.mark.parametrize("long_history", [False, True])
def test_refresh_skips_heads_and_encodes_only_valid_history(tmp_path, monkeypatch, long_history):
    trainer = make_trainer(tmp_path)
    try:
        trainer.collect_env_steps()
        if long_history:
            data = trainer.replay.current_eps[1].data
            for key in ("obs", "action"):
                data[key] = [data[key][0].clone() for _ in range(270)]
            data["is_first"] = [torch.tensor(t == 0) for t in range(270)]
        expected = [expected_row(trainer, row) for row in range(2)]
        encoded, filtered = [], []
        encoder = trainer.world.encoder.forward
        rssm = trainer.world.rssm.forward

        def encode(obs):
            encoded.append(obs.shape[0])
            return encoder(obs)

        def transition(h, z, action):
            filtered.append(h.shape[0])
            return rssm(h, z, action)

        def unused(*args, **kwargs):
            raise AssertionError("Prediction/training heads must not run during refresh")

        monkeypatch.setattr(trainer.world.encoder, "forward", encode)
        monkeypatch.setattr(trainer.world.rssm, "forward", transition)
        for head in (
            trainer.world.prior,
            trainer.world.decoder,
            trainer.world.reward_heads,
            trainer.world.continue_head,
        ):
            monkeypatch.setattr(head, "forward", unused)
        metrics = trainer.refresh_acting_state()
        assert encoded == ([256, 15] if long_history else [4])
        assert filtered == ([2] + [1] * 269 if long_history else [2, 1, 1])
        assert metrics["computed_tokens"] == metrics["context_tokens"] == (271 if long_history else 4)
        for row, state in enumerate(expected):
            torch.testing.assert_close(trainer.world_state.h[row], state.h[0])
            torch.testing.assert_close(trainer.world_state.z[row], state.z[0])
    finally:
        close(trainer)


def test_filter_step_matches_full_observe_with_identical_sampling(world_model):
    state = world_model.init_state(2)
    obs = torch.randn(2, world_model.flat_obs_dim)
    action = F.one_hot(torch.tensor([0, 1]), world_model.net_cfg.action_dim).float()
    first = torch.tensor([True, False])
    rng = torch.get_rng_state()
    observed = world_model.observe_step(state, obs, action, is_first=first)["post"]
    torch.set_rng_state(rng)
    filtered = world_model.filter_step(state, world_model.encoder(obs), action, first)
    torch.testing.assert_close(filtered.h, observed.h)
    torch.testing.assert_close(filtered.z, observed.z)
