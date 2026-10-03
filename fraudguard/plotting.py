"""One consistent, colour-blind-safe chart style for the notebooks and app."""

import matplotlib.pyplot as plt

# Fixed colours: "normal" is always blue and "fraud" always orange, in every chart.
NORMAL_COLOR = "#2a78d6"
FRAUD_COLOR = "#eb6834"
# One fixed colour per model (same order everywhere), never re-assigned.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE_COLOR = "#fcfcfb"


def apply_style() -> None:
    """Thin marks, quiet axes and a light grid so the data stands out."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE_COLOR,
            "axes.facecolor": SURFACE_COLOR,
            "savefig.facecolor": SURFACE_COLOR,
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "savefig.bbox": "tight",
            "font.size": 10,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": MUTED_COLOR,
            "axes.edgecolor": GRID_COLOR,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED_COLOR,
            "ytick.color": MUTED_COLOR,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )
