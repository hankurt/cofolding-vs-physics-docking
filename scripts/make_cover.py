"""Render the README cover / GitHub social-preview images.

    python scripts/make_cover.py [outdir]

Numbers are the three-level exposure split from docs/RESULTS.md
(`cofoldbench report --split exposure`). Default outdir is docs/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = str(Path(sys.argv[1] if len(sys.argv) > 1
               else Path(__file__).resolve().parent.parent / "docs")) + "/"

BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#262d36"
COFOLD = "#ff7b72"
COFOLD_P = "#ffa198"
COFOLD2 = "#d29922"
DOCK = "#3fb950"
DOCK2 = "#58a6ff"

levels = ["famous\npre-cut-off", "obscure\npre-cut-off", "after\ncut-off"]
x = np.arange(3)

# label, values, colour, marker, linewidth, linestyle
series = [
    ("Boltz-2 (co-folding)", [95, 71, 27], COFOLD, "o", 2.8, "-"),
    ("Boltz-2 pocket (co-folding)", [95, 81, 27], COFOLD_P, "o", 2.0, (0, (5, 2))),
    ("Chai-1 (co-folding)", [89, 48, 27], COFOLD2, "o", 2.2, "-"),
    ("GNINA (docking)", [74, 71, 73], DOCK, "s", 2.8, "-"),
    ("AutoDock Vina (docking)", [63, 62, 47], DOCK2, "s", 2.2, "-"),
]


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------- Cover 1: the headline gradient (wide social / GitHub social preview) ----------
fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor(BG)
style(ax)

for label, ys, c, m, lw, ls in series:
    ax.plot(x, ys, color=c, lw=lw, ls=ls, marker=m, ms=10, mec=BG, mew=1.5,
            label=label, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(levels, color=FG, fontsize=12)
ax.set_xlim(-0.35, 2.75)
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_ylabel("binding pose correct  (within 2 Å)", color=MUTED, fontsize=12,
              labelpad=10)

# Boltz-2, Boltz-2 pocket and Chai-1 all land on 27%; fan the labels out.
label_y = {
    "Boltz-2 (co-folding)": 36,
    "Boltz-2 pocket (co-folding)": 26,
    "Chai-1 (co-folding)": 16,
}
for label, ys, c, m, lw, ls in series:
    ax.annotate(label, (2, label_y.get(label, ys[2])), xytext=(12, 0),
                textcoords="offset points", color=c, fontsize=11.5,
                va="center")

# Title sits high enough to leave a clear gap above the subtitle, which stays
# just above the axes.
ax.set_title("AI co-folding vs physics-based docking",
             color=FG, fontsize=22, fontweight="bold", loc="left", pad=34)
ax.text(0, 1.035, "Docking is flat. Co-folding falls off a cliff on structures it has not seen.",
        transform=ax.transAxes, color=MUTED, fontsize=13)
fig.text(0.062, 0.025, "59 targets  ·  5 methods  ·  1437 poses  ·  one RTX 4070 Ti",
         color=MUTED, fontsize=11)
fig.tight_layout(rect=[0.02, 0.05, 1, 0.93])
fig.savefig(OUT + "cover.png", facecolor=BG)
plt.close(fig)

# ---------- Cover 2: grouped bars, same data, GitHub social-preview ratio 1280x640 ----------
fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor(BG)
style(ax)
w = 0.17
for i, (label, ys, c, m, lw, ls) in enumerate(series):
    pos = x + (i - 2) * w
    bars = ax.bar(pos, ys, width=w * 0.88, color=c, label=label, zorder=3)
    for b, v in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%", ha="center",
                color=c, fontsize=10.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(levels, color=FG, fontsize=12)
ax.set_ylim(0, 110)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_ylabel("binding pose correct  (within 2 Å)", color=MUTED, fontsize=12,
              labelpad=10)
leg = ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=11,
                labelcolor=FG)
ax.set_title("Where co-folding models actually win",
             color=FG, fontsize=22, fontweight="bold", loc="left", pad=28)
ax.text(0, 1.045,
        "Three-level exposure split: famous, obscure-but-seen, unseen. The overall average hides all of it.",
        transform=ax.transAxes, color=MUTED, fontsize=13)
fig.tight_layout(rect=[0.02, 0.02, 1, 0.94])
fig.savefig(OUT + "cover_bars.png", facecolor=BG)
plt.close(fig)

# ---------- Cover 3: square, minimal, for LinkedIn feed ----------
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
fig.patch.set_facecolor(BG)
style(ax)
ax.plot(x, [95, 71, 27], color=COFOLD, lw=4, marker="o", ms=14, mec=BG, mew=2,
        zorder=3)
ax.plot(x, [74, 71, 73], color=DOCK, lw=4, marker="s", ms=13, mec=BG, mew=2,
        zorder=3)
ax.text(-0.18, 32, "Boltz-2\nco-folding", color=COFOLD, fontsize=13,
        fontweight="bold", ha="left", va="center")
ax.annotate("GNINA\ndocking", (2, 73), xytext=(-6, 26),
            textcoords="offset points", color=DOCK, fontsize=13,
            fontweight="bold", ha="right")
ax.set_xticks(x)
ax.set_xticklabels(["famous", "obscure\n(still seen)", "unseen"], color=FG,
                   fontsize=13)
ax.set_xlim(-0.3, 2.3)
ax.set_ylim(0, 108)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_ylabel("binding pose correct", color=MUTED, fontsize=13, labelpad=10)
ax.set_title("Does the model know it,\nor did it memorise it?", color=FG,
             fontsize=24, fontweight="bold", loc="left", pad=26)
fig.text(0.1, 0.03, "cofolding-vs-physics-docking  ·  59 targets, 1437 poses",
         color=MUTED, fontsize=11)
fig.tight_layout(rect=[0.03, 0.06, 1, 0.93])
fig.savefig(OUT + "cover_square.png", facecolor=BG)
plt.close(fig)

print("done")
