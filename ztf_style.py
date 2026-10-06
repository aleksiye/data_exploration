"""Shared plotting style for the ZTF Bright Transient Survey notebooks.

Import it at the top of a notebook:

    import ztf_style
    ztf_style.use()

Keeping colours and chart settings in one place means every figure in the
series looks the same, and a class (e.g. "SN Ia") always has the same colour.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Chart chrome -----------------------------------------------------------
SURFACE = "#fcfcfb"      # figure / axes background
INK = "#0b0b0b"          # titles
INK_2 = "#52514e"        # axis labels, tick labels, annotations
MUTED = "#898781"        # de-emphasised marks ("everything that isn't the story")
GRID = "#e1e0d9"         # hairline gridlines
AXIS = "#c3c2b7"         # axis spines

# --- Categorical colours ----------------------------------------------------
# Three slots from a colour-blind-validated palette. Three is the maximum for
# scatter plots, where every colour must be distinguishable from every other.
CLASS3_COLORS = {
    "SN Ia": "#2a78d6",      # blue
    "Other SN": "#eb6834",   # orange
    "Non-SN": "#1baf7a",     # aqua
}
CLASS3_ORDER = list(CLASS3_COLORS)

IA = CLASS3_COLORS["SN Ia"]
NOT_IA = CLASS3_COLORS["Other SN"]
IA_COLORS = {True: IA, False: NOT_IA}

# Full eight-colour order, used as matplotlib's default colour cycle (for line
# and bar charts where only neighbouring series need to be distinguishable).
CYCLE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# --- Colormaps ---------------------------------------------------------------
# Sequential (magnitude / density): one hue, light -> dark.
BLUES = LinearSegmentedColormap.from_list(
    "ztf_blues",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
# Diverging (sign, e.g. correlation): blue <-> red with a neutral grey middle.
DIVERGING = LinearSegmentedColormap.from_list(
    "ztf_diverging",
    ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a9a8", "#e34948", "#9e2a29"],
)

FIG_DIR = Path(__file__).parent / "figures"


def use():
    """Apply the series-wide matplotlib/seaborn style."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.figsize": (8, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "lines.linewidth": 2,
        "patch.linewidth": 0,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "image.cmap": "ztf_blues",
    })
    for cmap in (BLUES, DIVERGING):
        if cmap.name not in mpl.colormaps:
            mpl.colormaps.register(cmap)


def save(fig, name):
    """Save a figure as PNG into figures/ (for slides and reports)."""
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path
