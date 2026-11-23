"""Compact Rich rendering for Dreamer training and held-out evaluation."""

from rich.console import Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


def number(value, precision=3):
    return "—" if value is None else f"{value:,.{precision}f}"


def render_training(state):
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column()
    table.add_column(justify="right")
    table.add_column()
    table.add_column(justify="right")
    fields = [
        ("Environment steps", "env_steps", 0),
        ("Episodes", "episodes", 0),
        ("Replay frames", "replay_size", 0),
        ("Sample length", "sequence_length", 0),
        ("Recent train return", "recent_return_mean", 3),
        ("World loss", "world_loss", 3),
        ("Actor loss", "actor_loss", 3),
        ("Critic loss", "critic_loss", 3),
        ("Dynamics KL", "kl_dyn", 3),
        ("Representation KL", "kl_rep", 3),
        ("Env steps/s", "env_steps_per_second", 1),
        ("Update seconds", "update_seconds", 2),
        ("Training ETA (s)", "eta_training_seconds", 0),
        ("Replay tokens/env step", "model_tokens_per_env_step", 2),
    ]
    rows = [(name, number(state.get(key), precision)) for name, key, precision in fields]
    cadence = state.get("cadence")
    if cadence:
        rows.append(("WM updates/env step", number(cadence.get("world_model_updates_per_env_transition"), 4)))
        rows.append(("Imagined steps/env step", number(cadence.get("imagined_transitions_per_env_transition"), 1)))
    if "learning_rate" in state:
        rows.append(("Learning rate", f"{state['learning_rate']:.2e}"))
    if state.get("cuda_memory_mb") is not None:
        rows.append(("CUDA allocated MiB", number(state["cuda_memory_mb"], 0)))
    for i in range(0, len(rows), 2):
        right = rows[i + 1] if i + 1 < len(rows) else ("", "")
        table.add_row(*rows[i], *right)
    return Panel(table, title="Dreamer training")


def render_evaluation(evaluation):
    if not evaluation:
        return Panel("Waiting for fixed-seed evaluation", title="Validation returns")
    table = Table.grid(expand=True)
    table.add_column()
    table.add_column(justify="right")
    for name, key in [("Random", "random_mean"), ("Untrained", "untrained_mean"), ("Trained", "trained_mean")]:
        table.add_row(name, number(evaluation.get(key)))
    for label, key in [("Δ vs random", "vs_random"), ("Δ vs untrained", "vs_untrained")]:
        result = evaluation.get(key)
        if result:
            ci = "n<2" if result["ci95_low"] is None else f"[{result['ci95_low']:.3f}, {result['ci95_high']:.3f}]"
            table.add_row(label, f"{result['mean_difference']:+.3f}  95% CI {ci}")
    return Panel(
        Group(table, Text("Validation diagnostics; confirm across seeds on separate final test levels.", style="dim")),
        title=f"Validation • update {evaluation.get('update', 0)}",
    )


def render_dashboard(state, evaluation, phase, run_dir, device):
    update, target = state.get("update", 0), state.get("target_updates", 1)
    progress = Group(
        Text(f"{phase} • {device} • update {update}/{target}", style="bold cyan"),
        ProgressBar(total=max(target, 1), completed=update),
        Text(str(run_dir), style="dim"),
    )
    return Group(Panel(progress, title="MiniHack offline run"), render_training(state), render_evaluation(evaluation))
