"""
plot_taguchi_path.py
--------------------
Visualizes the best path found by ACO Row 8 (selected Taguchi config)
on a single chosen instance. One image per run.

Usage (run from anywhere inside the project):
    python graph_program/plot_taguchi_path.py cpp_5x5_chunk
    python graph_program/plot_taguchi_path.py cpp_10x10_line
    python graph_program/plot_taguchi_path.py cpp_20x20_sparse

Pick the instance name exactly as it appears in the instances/ folder
(without the .txt extension).

Output: graph_program/plots/path_{instance}_row8.png

Design notes
------------
Light "field telemetry" theme, built for documentation:
  * Field cells are rounded tiles with small gutters (modern map look).
  * Green tiles  = covered ground; darker green = revisited (x2 / x3+).
  * Amber tiles  = ground the robot missed; slate tiles = obstacles.
  * The route is drawn like a navigation app: a white casing line
    underneath, then a colour-gradient line on top (cyan -> indigo)
    encoding traversal order, with chevrons showing direction.
  * A thin progression bar under the map maps colour -> mission time.
  * All text (telemetry, ACO parameters, legend) lives in a side panel,
    so nothing ever covers the field.
"""

import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# --- Paths ---
# Script lives at graph_program/ inside the project root, so CPP_BASE is one level up.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_BASE   = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")

# --- Config ---
ROW_NUM = 8

# Instance names match the filenames in instances/ (without .txt).
VALID_INSTANCES = [
    "cpp_5x5_chunk",   "cpp_5x5_line",   "cpp_5x5_sparse",
    "cpp_10x10_chunk", "cpp_10x10_line", "cpp_10x10_sparse",
    "cpp_20x20_chunk", "cpp_20x20_line", "cpp_20x20_sparse",
]

# --- Theme (single source of truth for every colour in the figure) ---
THEME = {
    "fig_bg":      "#F7F8FA",   # overall page
    "map_bg":      "#E4E8EC",   # shows through tile gutters
    "ink":         "#1B2733",   # main text
    "ink_soft":    "#5B6B7B",   # secondary text
    "card_bg":     "#FFFFFF",
    "card_edge":   "#D8DEE5",
    "covered_1":   "#C9E7CF",   # visited once
    "covered_2":   "#96D1A4",   # visited twice
    "covered_3":   "#5FB375",   # visited 3+ times
    "uncovered":   "#FFE08A",   # missed ground - pops immediately
    "obstacle":    "#46555F",
    "path_casing": "#FFFFFF",
    "start":       "#00A651",
    "end":         "#E53935",
    "accent":      "#2962FF",
}

# Traversal-order gradient: early -> late
PATH_CMAP = LinearSegmentedColormap.from_list(
    "mission", ["#00BCD4", "#2979FF", "#5E35B1"]
)


# --- Data loading ---

def load_grid(grid_path: str):
    """Parse instance .txt file into a 2D list (0=free, 1=obstacle)."""
    grid = []
    with open(grid_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Size"):
                continue
            row = [int(x) for x in line.split()]
            if row:
                grid.append(row)
    return grid


def load_best_solution(json_path: str, row_num: int) -> dict:
    """Extract best path, fitness breakdown and ACO params for one L18 row."""
    with open(json_path) as f:
        data = json.load(f)
    row = data["rows"][row_num - 1]
    return {
        "path":      [tuple(p) for p in row["best_path"]],
        "fitness":   row["best_fitness"],
        "distance":  row["best_distance"],
        "revisits":  row["best_revisits"],
        "unvisited": row["best_unvisited"],
        "params":    row.get("params", {}),
        "mean":      row.get("mean_fitness"),
        "std":       row.get("std_dev"),
        "sn":        row.get("sn_ratio"),
        "reps":      data.get("reps_per_row"),
    }


# --- Drawing: the field map ---

def draw_field(ax, grid, sol):
    """Draw obstacle/coverage tiles and overlay the gradient route."""
    nrows, ncols = len(grid), len(grid[0])
    path = sol["path"]

    visit_count = Counter(path)
    n_free = sum(1 for r in range(nrows) for c in range(ncols) if grid[r][c] == 0)
    coverage_pct = 100.0 * len(visit_count) / n_free if n_free > 0 else 0.0

    # -- tiles (rounded, slightly shrunk -> gutters show map_bg) --
    for r in range(nrows):
        for c in range(ncols):
            if grid[r][c] == 1:
                color = THEME["obstacle"]
            else:
                v = visit_count.get((r, c), 0)
                if v == 0:
                    color = THEME["uncovered"]
                elif v == 1:
                    color = THEME["covered_1"]
                elif v == 2:
                    color = THEME["covered_2"]
                else:
                    color = THEME["covered_3"]
            tile = mpatches.FancyBboxPatch(
                (c - 0.5, nrows - 1 - r - 0.5), 1, 1,
                boxstyle="round,pad=-0.045,rounding_size=0.10",
                facecolor=color, edgecolor="none", zorder=1,
            )
            ax.add_patch(tile)

    # -- route: white casing underneath, gradient line on top --
    xs = np.array([p[1] for p in path], dtype=float)
    ys = np.array([nrows - 1 - p[0] for p in path], dtype=float)
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n_seg = len(segs)

    lw = max(1.6, 5.0 - nrows * 0.16)           # thinner on big grids
    casing = LineCollection(
        segs, colors=THEME["path_casing"], linewidths=lw + 2.2,
        capstyle="round", joinstyle="round", zorder=3, alpha=0.9,
    )
    ax.add_collection(casing)
    route = LineCollection(
        segs, cmap=PATH_CMAP, linewidths=lw,
        capstyle="round", joinstyle="round", zorder=4,
    )
    route.set_array(np.linspace(0.0, 1.0, n_seg))
    ax.add_collection(route)

    # -- cell size in points (drives marker/chevron sizing on any grid) --
    fig = ax.figure
    pos = ax.get_position()
    cell_pt = pos.width * fig.get_figwidth() * 72.0 / ncols

    # -- direction chevrons: rotated triangles at segment midpoints --
    step = max(3, n_seg // 18)
    chev_ms = min(12.0, max(6.0, 0.26 * cell_pt))   # diameter in points
    for i in range(step // 2, n_seg, step):
        (x0, y0), (x1, y1) = segs[i]
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 and dy == 0:
            continue
        t = i / max(n_seg - 1, 1)
        angle = np.degrees(np.arctan2(dy, dx)) - 90  # triangle points "up" at 0°
        ax.plot((x0 + x1) / 2, (y0 + y1) / 2,
                marker=(3, 0, angle), markersize=chev_ms,
                color=PATH_CMAP(t), markeredgecolor="white",
                markeredgewidth=0.7, linestyle="none", zorder=5)

    # -- start / end markers (white halo ring so they read on any tile) --
    d_pt = min(22.0, max(13.0, 0.38 * cell_pt))     # marker diameter in points
    ms = d_pt ** 2
    ax.scatter([xs[0]], [ys[0]], s=ms * 1.9, color="white", zorder=6)
    ax.scatter([xs[0]], [ys[0]], s=ms, color=THEME["start"],
               marker="o", edgecolors="white", linewidths=1.4, zorder=7)
    ax.scatter([xs[-1]], [ys[-1]], s=ms * 2.1, color="white", zorder=6)
    ax.scatter([xs[-1]], [ys[-1]], s=ms * 1.5, color=THEME["end"],
               marker="*", edgecolors="white", linewidths=1.0, zorder=7)

    # -- axes cosmetics --
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(-0.5, nrows - 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor(THEME["map_bg"])
    tick_every = 1 if nrows <= 10 else 2
    ax.set_xticks(range(0, ncols, tick_every))
    # y labels show matrix row index (row 0 at the top, matching the grid file)
    ax.set_yticks(range(nrows - 1, -1, -tick_every))
    ax.set_yticklabels(range(0, nrows, tick_every), fontsize=7,
                       color=THEME["ink_soft"])
    ax.tick_params(length=0, labelsize=7, colors=THEME["ink_soft"])
    for s in ax.spines.values():
        s.set_visible(False)

    return coverage_pct


def draw_progress_bar(ax):
    """Thin colourbar mapping route colour -> traversal order."""
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, aspect="auto", cmap=PATH_CMAP)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-0.012, 0.5, "START", transform=ax.transAxes, ha="right",
            va="center", fontsize=7.5, fontweight="bold", color=THEME["start"])
    ax.text(1.012, 0.5, "END", transform=ax.transAxes, ha="left",
            va="center", fontsize=7.5, fontweight="bold", color=THEME["end"])
    ax.set_title("route progression", fontsize=7, color=THEME["ink_soft"],
                 pad=2, loc="center")


# --- Drawing: the side panel ---

class _Panel:
    """Stacks rounded cards top-down in the panel axis.

    All vertical sizes are multiplied by a scale factor so the absolute
    spacing (in inches) stays constant no matter how tall the figure is —
    cards stay anchored to the top and can never overflow the bottom.
    """

    ROW   = 0.042   # key-value line spacing (at reference height)
    HDR   = 0.082   # card header zone
    PAD   = 0.014   # inner bottom padding
    GAP   = 0.022   # gap between cards
    REF_H = 4.9     # scale = REF_H / fig height -> spacing constant in inches
                    # (fig height is never < 5.6, so total stack always fits)

    def __init__(self, ax):
        self.ax = ax
        self.s  = self.REF_H / ax.figure.get_figheight()
        self.y  = 0.995
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    def card(self, title, n_rows, extra=0.0):
        """Open a card sized for n_rows kv-lines (+extra); returns cursor y."""
        s = self.s
        height = (self.HDR + n_rows * self.ROW + self.PAD + extra) * s
        self.ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, self.y - height), 0.96, height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            transform=self.ax.transAxes, facecolor=THEME["card_bg"],
            edgecolor=THEME["card_edge"], linewidth=0.9, zorder=1,
        ))
        self.ax.text(0.08, self.y - 0.038 * s, title,
                     transform=self.ax.transAxes, fontsize=7.5,
                     fontweight="bold", color=THEME["ink_soft"], zorder=2)
        cursor = self.y - self.HDR * s
        self.y -= height + self.GAP * s
        return cursor

    def kv(self, y, label, value, value_color=None, bold=False):
        """One 'label ...... value' line; returns y for the next line."""
        self.ax.text(0.08, y, label, transform=self.ax.transAxes, fontsize=8.5,
                     color=THEME["ink_soft"], family="monospace", zorder=2)
        self.ax.text(0.92, y, value, transform=self.ax.transAxes, fontsize=8.5,
                     color=value_color or THEME["ink"], family="monospace",
                     ha="right", fontweight="bold" if bold else "normal",
                     zorder=2)
        return y - self.ROW * self.s


def draw_panel(ax, sol, coverage_pct):
    """Telemetry / parameters / legend panel on the right side."""
    panel = _Panel(ax)
    s = panel.s
    n_steps = len(sol["path"]) - 1
    warn = "#E65100"

    # ---- card 1: mission telemetry (incl. coverage bar) ----
    y = panel.card("MISSION TELEMETRY", 6, extra=0.040)
    y = panel.kv(y, "fitness",   f"{sol['fitness']:,.2f}", THEME["accent"], True)
    y = panel.kv(y, "distance",  f"{sol['distance']:,.2f}")
    y = panel.kv(y, "steps",     f"{n_steps}")
    y = panel.kv(y, "revisits",  f"{sol['revisits']}",
                 THEME["start"] if sol["revisits"] == 0 else warn)
    y = panel.kv(y, "unvisited", f"{sol['unvisited']}",
                 THEME["start"] if sol["unvisited"] == 0 else warn)
    y = panel.kv(y, "coverage",  f"{coverage_pct:.1f}%",
                 THEME["start"] if coverage_pct >= 100 else warn, True)
    bar_w, bar_x, bar_h = 0.84, 0.08, 0.016 * s
    ax.add_patch(mpatches.FancyBboxPatch(
        (bar_x, y), bar_w, bar_h,
        boxstyle="round,pad=0.001,rounding_size=0.008",
        transform=ax.transAxes, facecolor="#E8ECF0", edgecolor="none", zorder=2))
    ax.add_patch(mpatches.FancyBboxPatch(
        (bar_x, y), bar_w * min(coverage_pct, 100) / 100, bar_h,
        boxstyle="round,pad=0.001,rounding_size=0.008",
        transform=ax.transAxes,
        facecolor=THEME["start"] if coverage_pct >= 100 else "#FB8C00",
        edgecolor="none", zorder=3))

    # ---- card 2: ACO configuration (Taguchi row) ----
    p = sol["params"]
    labels = {"n_ants": "ants", "alpha": "alpha α", "beta": "beta β",
              "rho": "rho ρ", "q0": "q₀"}
    keys = [k for k in ["n_ants", "alpha", "beta", "rho", "q0"] if k in p]
    y = panel.card(f"ACO CONFIG · L18 ROW {ROW_NUM}", len(keys))
    for key in keys:
        y = panel.kv(y, labels[key], f"{p[key]:g}")

    # ---- card 3: row statistics across reps ----
    if sol["mean"] is not None:
        y = panel.card(f"ROW STATS · {sol['reps']} REPS", 3)
        y = panel.kv(y, "mean", f"{sol['mean']:,.2f}")
        y = panel.kv(y, "std",  f"{sol['std']:,.2f}")
        y = panel.kv(y, "S/N",  f"{sol['sn']:,.2f} dB")

    # ---- card 4: legend (2 columns x 3 rows) ----
    legend_items = [
        (THEME["covered_1"], "covered"),
        (THEME["covered_2"], "covered ×2"),
        (THEME["covered_3"], "covered ×3+"),
        (THEME["uncovered"], "missed"),
        (THEME["obstacle"],  "obstacle"),
    ]
    row_h = 0.036 * s
    y = panel.card("FIELD LEGEND", 0, extra=3 * 0.036 - 0.014)
    for i, (color, label) in enumerate(legend_items):
        col_i, row_i = i % 2, i // 2
        x0 = 0.08 + col_i * 0.45
        yy = y - row_i * row_h
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, yy - 0.005 * s), 0.045, 0.020 * s,
            boxstyle="round,pad=0.001,rounding_size=0.006",
            transform=ax.transAxes, facecolor=color, edgecolor="#C6CDD4",
            linewidth=0.5, zorder=2))
        ax.text(x0 + 0.065, yy, label, transform=ax.transAxes,
                fontsize=7.3, color=THEME["ink"], va="bottom", zorder=2)


# --- Entry point ---

def main():
    parser = argparse.ArgumentParser(
        description="Visualize ACO best path for Row 8 (Taguchi result)."
    )
    parser.add_argument(
        "instance",
        choices=VALID_INSTANCES,
        metavar="INSTANCE",
        help=(
            "Instance to visualize — use the exact filename from instances/ without .txt. "
            f"Choices: {', '.join(VALID_INSTANCES)}"
        ),
    )
    args = parser.parse_args()

    json_path = os.path.join(CPP_BASE, "results", f"taguchi_{args.instance}.json")
    grid_path = os.path.join(CPP_BASE, "instances", f"{args.instance}.txt")

    for path, label in [(json_path, "JSON"), (grid_path, "Grid")]:
        if not os.path.isfile(path):
            print(f"Error: {label} not found at:\n  {path}")
            return

    print(f"Loading JSON: {json_path}")
    sol = load_best_solution(json_path, ROW_NUM)

    print(f"Loading grid: {grid_path}")
    grid = load_grid(grid_path)

    nrows, ncols = len(grid), len(grid[0])

    # --- figure layout: [ map | panel ] with a thin progress bar under map ---
    map_w   = max(4.6, ncols * 0.42)
    panel_w = 2.7
    fig_w   = map_w + panel_w + 0.8
    fig_h   = max(5.6, nrows * 0.42 + 1.9)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(THEME["fig_bg"])
    gs = fig.add_gridspec(
        2, 2, width_ratios=[map_w, panel_w], height_ratios=[1, 0.030],
        left=0.06, right=0.985, top=0.875, bottom=0.075,
        wspace=0.16, hspace=0.16,
    )
    ax_map   = fig.add_subplot(gs[0, 0])
    ax_bar   = fig.add_subplot(gs[1, 0])
    ax_panel = fig.add_subplot(gs[:, 1])

    # --- header ---
    display = args.instance.removeprefix("cpp_")
    size, layout = display.split("_", 1)
    fig.text(0.06, 0.955, "ACO Coverage Path", fontsize=15,
             fontweight="bold", color=THEME["ink"])
    fig.text(0.06, 0.915,
             f"Taguchi L18 · Row {ROW_NUM} best tour · "
             f"instance {size} {layout.capitalize()}",
             fontsize=9.5, color=THEME["ink_soft"])

    coverage_pct = draw_field(ax_map, grid, sol)
    draw_progress_bar(ax_bar)
    draw_panel(ax_panel, sol, coverage_pct)

    out_path = os.path.join(PLOTS_DIR, f"path_{args.instance}_row{ROW_NUM}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
