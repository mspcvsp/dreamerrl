# DreamerRL

DreamerRL is a PyTorch implementation of DreamerV3 for studying learned world models. It contains frozen-agent research on simulated network defense in TTCP CAGE Challenge 2 and on memory in a controlled MiniHack T-maze.

A compact test showed why aggregate reconstruction can be misleading. In the three studied T-maze checkpoints the cue was recoverable from the encoder and soft posterior. Conversion to one hard categorical state nearly erased it. CAGE2 did not reproduce that finding for its selected pilot target, and the final matched-history test stopped for insufficient common support.

## Start with the evidence

```bash
python3 docs/publication_figures/read_saved_results.py
```

This CPU demonstration reads the tracked aggregate snapshot and prints cue-decoding accuracy for three stages. It requires Python only, with no packages, model weights, simulator, or training. Open [the saved values](docs/publication_figures/evidence.json) and compare a stage with the two-choice chance result. The [presenter guide](docs/presentation/presenter_guide.md) includes a short audience exercise.

| Explore | Start here |
|---|---|
| What the project found | [Campaign closeout](docs/closeout/report.md) and [evidence-to-claim matrix](docs/closeout/claims.md) |
| World-model architecture | [Architecture and fidelity](docs/dreamerv3_fidelity.md), [published DreamerV3 paper](https://www.nature.com/articles/s41586-025-08744-2) |
| CAGE2 Blue simulated defense | [Environment and setup](environments/cage2/README.md), [upstream challenge](https://github.com/cage-challenge/cage-challenge-2) |
| Controlled MiniHack memory task | [Legacy profile](docs/uv_environment.md#explicit-legacy-minihacknle-profile) and [frozen investigation](docs/frozen_memory_findings.md) |
| Diagnostics and example configurations | [Shared framework](docs/frozen_memory_diagnostics.md), [CAGE2 pilot settings](environments/cage2/cage2_memory_pilot.json), [archive forecast settings](environments/cage2/archived_history_baselines.json) |
| Publication materials | [Article draft](docs/linkedin/article.md), [Scout slides](docs/presentation/slides.md), [build and QA](docs/presentation/README.md) |

## How it works

An encoder extracts features from the current observation. The recurrent state carries context across actions. A categorical latent represents the current state given the observation. The learned model predicts from those states and can generate imagined trajectories for actor and critic learning. The frozen diagnostic framework tests what is encoded, what survives compression and recurrence, and how the acting policy responds to controlled state changes. The [runtime metadata contract](docs/runtime_env_info.md) explains how model dimensions are derived from the environment.

CAGE2 is the primary simulation. Its historical comparison included flat and hierarchical policies. Differences were small and changed direction across opponents in the local 30-step evaluation. MiniHack is retained for reproducing the controlled memory findings and running regression tests. It uses a separate environment profile.

## How the project was developed

The project used Microsoft Copilot and OpenAI Codex extensively across implementation, debugging, tests, diagnostics, analysis and communication. The human technical director set requirements, chose experiments, approved workloads and reviewed the evidence. The author uploaded text to Microsoft Copilot manually outside VS Code. GitHub hosts the code and version history. See the [consistency review](docs/presentation/narrative_consistency.md) for the scope of the tool attribution.

## Install the research software

Use Git, Bash, uv and the managed Python version described in the [locked environment guide](docs/uv_environment.md). Follow the platform-specific guidance there, including the native-wheel preparation on Spark.

```bash
git submodule update --init
bash --noprofile --norc scripts/setup_uv.sh
uv sync --locked
uv run --locked python environments/cage2/learning_curve.py --help
```

The last command lists options. The launchers for full training are separate from the demonstration. Checkpoint and archive-based reproduction requires the original files and profile described in the [artifact inventory](docs/closeout/artifacts.md). Those files live in ignored local storage. Use provenance hashes when obtaining them from the maintainer.

| Workload | Typical prerequisites |
|---|---|
| Read the aggregate snapshot | Python 3 on a CPU. No research environment setup. |
| Build the article graphics and slide deck | CPU, the locked Matplotlib environment, and system Pandoc, XeLaTeX/Beamer, Poppler, and optional LibreOffice. [Build guide](docs/presentation/README.md) |
| Reproduce frozen diagnostics | Compatible Python profile, original checkpoints and archives, and the relevant source revision. The completed campaign used a Spark host and CPU diagnostics. |
| Full RL training | Separate workload. The historical pinned training configuration and native CUDA requirements are in the [CAGE2 reproduction record](environments/cage2/reproducibility/README.md). |

## Limits and license

These results concern frozen checkpoints and specific simulator protocols. The T-maze categorical bottleneck is scoped to the tested runs. The CAGE2 matched-history gate left recurrent recoverability unresolved. Deferred architecture and learning-objective comparisons are future research.

Project code is [MIT licensed](LICENSE). CybORG is pinned as a third-party submodule with its own license. DreamerV3 and MiniHack are cited upstream projects. See the [release review](docs/presentation/release_readiness.md) for portability and publication boundaries.
