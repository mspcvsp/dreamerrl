"""Shared diagnostic visualization settings and export."""
from pathlib import Path

from dreamerrl.eval.memory.reporting import write_json  # noqa: F401 (legacy import)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, MIST, ORANGE, IVORY = "#13344C", "#499FA4", "#A0C3D9", "#FF9636", "#F2F3EE"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "figure.facecolor": IVORY,
                         "axes.facecolor": IVORY, "text.color": NAVY, "axes.labelcolor": NAVY,
                         "axes.edgecolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                         "axes.spines.top": False, "axes.spines.right": False})


def save_figure(fig, path: Path):
    fig.tight_layout()
    for ext in (".svg", ".png"):
        fig.savefig(path.with_suffix(ext), dpi=320)
    plt.close(fig)
