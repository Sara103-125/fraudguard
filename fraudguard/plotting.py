"""One consistent, colour-blind-safe chart style for the notebooks and app."""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

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


def plot_confusion_matrix(ax, y_true, y_pred, title: str = "") -> None:
    """2x2 confusion matrix. Colour = % of the actual class, text = count and %."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    blues = LinearSegmentedColormap.from_list("blues", ["#eef4fc", "#0d366b"])
    ax.imshow(row_pct, cmap=blues, vmin=0, vmax=1)

    names = [["True negative", "False alarm"], ["Missed fraud", "Caught fraud"]]
    for i in range(2):
        for j in range(2):
            color = "white" if row_pct[i, j] > 0.5 else TEXT_COLOR
            ax.text(j, i, f"{names[i][j]}\n{cm[i, j]:,}\n({row_pct[i, j]:.1%})",
                    ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actual normal", "Actual fraud"])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)


def plot_pr_curves(ax, y_true, scores_by_model: dict) -> None:
    """Precision-recall curve for each model, with PR-AUC in the legend."""
    for color, (name, scores) in zip(MODEL_COLORS, scores_by_model.items()):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, color=color, label=f"{name} (PR-AUC {ap:.3f})")
    fraud_rate = sum(y_true) / len(y_true)
    ax.axhline(fraud_rate, color=MUTED_COLOR, linestyle="--", linewidth=1,
               label=f"Random guessing (PR-AUC {fraud_rate:.3f})")
    ax.set_xlabel("Recall (share of frauds caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="center")  # the middle of a PR plot is usually empty
