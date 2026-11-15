"""Rich Live dashboard with automatic plain-output fallback under redirection."""

from typing import Any

from rich.console import Console
from rich.live import Live

from dreamerrl.utils.trainer_renderers import render_dashboard


class RLDashboard:
    def __init__(self, run_dir, device, mode="auto", console=None):
        self.console = console or Console()
        self.run_dir, self.device = run_dir, str(device)
        self.state: dict[str, Any] = {}
        self.evaluation: dict[str, Any] | None = None
        self.phase = "starting"
        self.enabled = mode != "off" and self.console.is_terminal
        self.live = (
            Live(get_renderable=self.render, console=self.console, refresh_per_second=2, screen=False)
            if self.enabled
            else None
        )

    def render(self):
        return render_dashboard(self.state, self.evaluation, self.phase, self.run_dir, self.device)

    def __enter__(self):
        if self.live:
            self.live.start()
        return self

    def update(self, state=None, evaluation=None, phase=None):
        if state is not None:
            self.state = state
        if evaluation is not None:
            self.evaluation = evaluation
        if phase is not None:
            self.phase = phase
        if self.live:
            self.live.refresh()

    def __exit__(self, *args):
        if self.live:
            self.live.stop()
