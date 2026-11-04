"""Consistent, accessible matplotlib styling for all figures (light theme, fixed series order)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; figures are always saved, never only shown
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order - colour follows the entity (method), never its rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
METHOD_ORDER = ["deterministic", "mc_dropout", "tta", "mc_tta", "ensemble"]
METHOD_LABEL = {"deterministic": "Deterministic", "mc_dropout": "MC Dropout", "tta": "Paired TTA",
                "mc_tta": "MC Dropout + TTA", "ensemble": "Deep ensemble"}
METHOD_COLOR = {m: SERIES[i] for i, m in enumerate(METHOD_ORDER)}
LABEL_COLOR = {0: SERIES[0], 1: SERIES[1]}  # no cancer / cancer
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
NEUTRAL = "#8a8985"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": TEXT,
        "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def save(fig, path):
    fig.savefig(path)   # always save before closing; never rely on show()
    plt.close(fig)
    return path
