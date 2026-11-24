"""One real replay batch per encoder on GB10; bounded integration test, not a learning run."""

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import torch
from host_experiment_support import initialize_paired, load_config
from learner_cycle import optimizer_steps

from dreamerrl.training.trainer import DreamerTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GB10"
    torch.set_num_threads(2)
    results = []
    with tempfile.TemporaryDirectory(prefix="dreamerrl-host-integration-") as temp:
        for arm in ("flat", "cage2_host"):
            trainer = DreamerTrainer(load_config(arm, 0, Path(temp) / arm))
            try:
                hashes = initialize_paired(trainer)
                counts = {
                    name: sum(p.numel() for p in module.parameters())
                    for name, module in [
                        ("encoder", trainer.world.encoder),
                        ("world", trainer.world),
                        ("actor", trainer.actor),
                        ("critic", trainer.critic),
                    ]
                }
                started = time.perf_counter()
                for _ in range(16):
                    trainer.collect_env_steps()
                collection_seconds = time.perf_counter() - started
                assert trainer.total_env_steps == 480 and len(trainer.replay.episodes) == 16
                batch = trainer.replay.sample(16, seed=0)
                before = {name: p.detach().clone() for name, p in trainer.world.encoder.named_parameters()}
                torch.cuda.synchronize()
                started = time.perf_counter()
                model = trainer.update_world_model(batch, 0)
                actor, critic = trainer.update_actor_critic(batch, 0)
                torch.cuda.synchronize()
                update_seconds = time.perf_counter() - started
                loss = dict(world=float(model.total_loss), actor=actor, critic=critic)
                assert all(math.isfinite(value) for value in loss.values())
                delta = sum(
                    float((value.detach() - before[name]).abs().sum())
                    for name, value in trainer.world.encoder.named_parameters()
                )
                assert delta > 0 and optimizer_steps(trainer) == {"model_opt": [1], "actor_opt": [1], "critic_opt": [1]}
                gradients = {
                    name: float(p.grad.abs().sum())
                    for name, p in trainer.world.encoder.named_parameters()
                    if p.grad is not None
                }
                assert all(math.isfinite(value) and value > 0 for value in gradients.values())
                results.append(
                    dict(
                        arm=arm,
                        parameter_counts=counts,
                        losses=loss,
                        optimizer_steps=optimizer_steps(trainer),
                        encoder_delta_l1=delta,
                        encoder_gradient_l1=gradients,
                        common_initialization_hashes=hashes,
                        collection_seconds=collection_seconds,
                        one_batch_seconds=update_seconds,
                        real_transitions=480,
                        batch_size=16,
                        seq_len=16,
                        burn_in=8,
                        imagination_horizon=15,
                        device=str(trainer.device),
                    )
                )
            finally:
                trainer.env.venv.close()
                trainer.tb.close()
    assert results[0]["common_initialization_hashes"] == results[1]["common_initialization_hashes"]
    report = dict(scope="Two one-batch integration tests, no formal experiment or historical weights", results=results)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
