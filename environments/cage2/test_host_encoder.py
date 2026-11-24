"""Host encoder unit tests plus one bounded real-replay integration check."""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from dreamerrl.env.cage2.cage2_env_wrapper import Cage2BlueEnv
from dreamerrl.env.cage2.host_schema import visible_host_schema
from dreamerrl.models.host_obs_encoder import HostObsEncoder
from dreamerrl.models.mlp_obs_encoder import MLPObsEncoder
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config

HOSTS = (
    "Defender",
    "Enterprise0",
    "Enterprise1",
    "Enterprise2",
    "Op_Host0",
    "Op_Host1",
    "Op_Host2",
    "Op_Server0",
    "User0",
    "User1",
    "User2",
    "User3",
    "User4",
)
SUBNETS = (0,) * 4 + (1,) * 4 + (2,) * 5


def config(tmp_path, host=True):
    cfg = make_default_config()
    cfg.env.env_id, cfg.env.num_envs, cfg.env.max_episode_steps = "CAGE2-Blue-BLine-v0", 1, 30
    cfg.env.parallel = False
    cfg.world.obs_encoder = "cage2_host" if host else "flat"
    cfg.world.cage2_host_order = HOSTS if host else ()
    cfg.world.cage2_subnet_ids = SUBNETS if host else ()
    cfg.world.aux_objectives, cfg.world.disable_aux_losses = [], True
    cfg.train.cuda, cfg.train.use_amp, cfg.train.enable_fused_adamw_kernels = False, False, False
    cfg.train.freeze_actor_critic_steps, cfg.train.warmup_steps = 0, 0
    cfg.train.batch_size, cfg.train.seq_len, cfg.train.burn_in, cfg.train.collect_steps = 2, 4, 2, 30
    cfg.world.imagination_horizon = 3
    cfg.log.enable_wandb, cfg.log.tb_logdir = False, str(tmp_path)
    return cfg


def test_shapes_count_batching_and_gradients():
    encoder = HostObsEncoder(HOSTS, SUBNETS)
    assert sum(p.numel() for p in encoder.parameters()) == 56508
    assert sum(p.numel() for p in MLPObsEncoder(52, 200).parameters()) == 50800
    obs = torch.rand(2, 3, 52, requires_grad=True)
    batched = encoder(obs)
    assert batched.shape == (2, 3, 200) and encoder.output_size == 200
    torch.testing.assert_close(batched, encoder(obs.reshape(6, 52)).reshape(2, 3, 200))
    batched.square().mean().backward()
    assert obs.grad is not None and torch.isfinite(obs.grad).all() and obs.grad.abs().sum() > 0
    for parameter in encoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_complete_host_record_permutation_invariance():
    encoder = HostObsEncoder(HOSTS, SUBNETS)
    fields = torch.rand(3, 13, 4)
    mask = torch.ones(3, 13, dtype=torch.bool)
    mask[:, 2] = False
    permutation = torch.tensor([12, 3, 8, 1, 10, 0, 4, 6, 5, 9, 7, 2, 11])
    expected = encoder.encode_hosts(fields, encoder.host_ids, encoder.subnet_ids, mask)
    actual = encoder.encode_hosts(
        fields[:, permutation], encoder.host_ids[permutation], encoder.subnet_ids[permutation], mask[:, permutation]
    )
    torch.testing.assert_close(actual, expected)
    # Moving evidence to other identities is different from reordering records.
    assert not torch.allclose(encoder(fields.reshape(3, 52)), encoder(fields[:, permutation].reshape(3, 52)))


def test_masking_missing_values_padding_and_all_masked():
    encoder = HostObsEncoder(HOSTS, SUBNETS)
    fields = torch.rand(2, 13, 4)
    mask = torch.ones(2, 13, dtype=torch.bool)
    mask[:, 2] = False
    expected = encoder(fields.flatten(-2), mask)
    fields[:, 2] = float("nan")
    torch.testing.assert_close(encoder(fields.flatten(-2), mask), expected)
    padded = torch.cat([fields, torch.full((2, 2, 4), float("nan"))], 1)
    padded_mask = torch.cat([mask, torch.zeros(2, 2, dtype=torch.bool)], 1)
    ids = torch.cat([encoder.host_ids, torch.tensor([0, 0])])
    subnets = torch.cat([encoder.subnet_ids, torch.tensor([0, 0])])
    torch.testing.assert_close(encoder.encode_hosts(padded, ids, subnets, padded_mask), expected)
    all_masked = encoder(torch.full((2, 52), float("nan")), torch.zeros_like(mask))
    assert torch.isfinite(all_masked).all()
    valid_zeros = encoder(torch.zeros(2, 52))
    assert not torch.allclose(all_masked, valid_zeros)


def test_checkpoint_serialization_and_bad_schema(tmp_path):
    encoder = HostObsEncoder(HOSTS, SUBNETS)
    obs = torch.rand(2, 52)
    path = tmp_path / "encoder.pt"
    torch.save(encoder.state_dict(), path)
    restored = HostObsEncoder(HOSTS, SUBNETS)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored(obs), encoder(obs), atol=0, rtol=0)
    assert torch.equal(restored.subnet_ids, encoder.subnet_ids)
    with pytest.raises(ValueError):
        encoder(torch.rand(2, 51))
    with pytest.raises(ValueError):
        encoder(obs, torch.ones(2, 13))
    with pytest.raises(ValueError):
        HostObsEncoder(HOSTS, (0,) * 13)


def test_visible_schema_matches_actual_vector_and_resets(tmp_path):
    env = Cage2BlueEnv(config(tmp_path).env, torch.device("cpu"))
    try:
        for seed in (0, 1, 2):
            initial = env.reset(seed=seed)
            assert env.host_order == HOSTS and env.subnet_ids == SUBNETS
            assert visible_host_schema(env.wrapper) == (HOSTS, SUBNETS)
            assert initial["state"].shape == (1, 52)
    finally:
        env.close()


@pytest.mark.parametrize("cuda", [False, True])
def test_real_replay_update_checkpoint_and_encoder_mismatch_rejected(tmp_path, cuda):
    if cuda and not torch.cuda.is_available():
        pytest.skip("GPU unavailable")
    cfg = config(tmp_path / "initial")
    cfg.train.cuda = cuda
    trainer = DreamerTrainer(cfg)
    resumed = flat = None
    try:
        trainer.collect_env_steps()
        batch = trainer.replay.sample(2, seed=11)
        before = {k: p.detach().clone() for k, p in trainer.world.encoder.named_parameters()}
        model = trainer.update_world_model(batch, 0)
        actor, critic = trainer.update_actor_critic(batch, 0)
        assert torch.isfinite(torch.tensor([model.total_loss.item(), actor, critic])).all()
        assert any(not torch.equal(before[k], p) for k, p in trainer.world.encoder.named_parameters())
        path = tmp_path / "restart.pt"
        trainer.save_checkpoint(path)
        resumed = DreamerTrainer.from_checkpoint(path, cuda=cuda, log_dir=tmp_path / "reload")
        assert asdict(resumed.cfg.world) == asdict(trainer.cfg.world)
        for key, value in trainer.world.state_dict().items():
            torch.testing.assert_close(resumed.world.state_dict()[key], value, atol=0, rtol=0)
        assert len(resumed.replay.episodes) == len(trainer.replay.episodes)
        flat = DreamerTrainer(config(tmp_path / "flat", host=False))
        with pytest.raises(ValueError, match="configuration"):
            flat.load_checkpoint(path)
    finally:
        for item in (trainer, resumed, flat):
            if item is not None:
                item.env.venv.close()
                item.tb.close()


def test_original_update1_still_loads_without_host_weights(tmp_path):
    path = Path(__file__).resolve().parents[2] / "runs/cage2_blue_cycle_seed0_20261123/latest.pt"
    if not path.exists():
        pytest.skip("Historical update-1 artifact is available on Spark, not bundled with source")
    trainer = DreamerTrainer.from_checkpoint(path, cuda=False, log_dir=tmp_path)
    try:
        assert isinstance(trainer.world.encoder, MLPObsEncoder)
        assert trainer.cfg.world.obs_encoder == "flat"
        assert trainer.completed_updates == 1 and trainer.total_env_steps == 510
    finally:
        trainer.env.venv.close()
        trainer.tb.close()
