"""Per-method summary table and figures from ``results.csv``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUCCESS_RMSD = 2.0

# Fixed colour per method so a method keeps its colour in every figure,
# whichever subset of methods a run contains (categorical palette, fixed order).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
KNOWN_METHODS = ["boltz2", "boltz2_pocket", "chai1", "af3", "haddock3", "gnina", "vina", "autodock_gpu"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def method_colors(methods: list[str]) -> dict[str, str]:
    order = [m for m in KNOWN_METHODS if m in methods] + sorted(m for m in methods if m not in KNOWN_METHODS)
    if len(order) > len(PALETTE):
        raise ValueError(f"{len(order)} methods; plot at most {len(PALETTE)} at a time")
    slots = {m: KNOWN_METHODS.index(m) for m in order if m in KNOWN_METHODS}
    free = [i for i in range(len(PALETTE)) if i not in slots.values()]
    for m in order:
        if m not in slots:
            slots[m] = free.pop(0)
    return {m: PALETTE[slots[m]] for m in order}


def _success(df: pd.DataFrame) -> pd.Series:
    return df["ligand_rmsd"].fillna(np.inf) <= SUCCESS_RMSD


# Training-data cut-offs (PDB release dates) reported by the model authors.
# A target released after a model's cut-off cannot have been memorised by it.
TRAINING_CUTOFFS = {"chai1": "2021-01-12", "af3": "2021-09-30", "boltz2": "2023-06-01"}
DEFAULT_CUTOFF = max(TRAINING_CUTOFFS.values())


def time_split(release_dates: dict[str, str | None], cutoff: str = DEFAULT_CUTOFF) -> dict[str, str]:
    """target_id -> 'before cutoff' / 'after cutoff' (targets with no date are left out)."""
    out = {}
    for tid, date in release_dates.items():
        if date:
            out[tid] = "after cutoff" if str(date)[:10] > cutoff else "before cutoff"
    return out


def exposure_split(targets, cutoff: str = DEFAULT_CUTOFF) -> dict[str, str]:
    """target_id -> 'famous (pre-cutoff)' / 'obscure (pre-cutoff)' / 'after cutoff'.

    Being released before a model's cut-off is not one condition but two. The
    classic drug complexes are in the training data many times over - every
    benchmark, every tutorial, redundant entries - while an ordinary structure
    from the same years is in it once. Splitting the pre-cut-off half that way
    separates *heavily represented* from merely *seen*, which a two-way time
    split confounds. Targets are marked famous by ``classic`` in their note.
    """
    out = {}
    for t in targets:
        date = (t.release_date or "")[:10]
        if not date:
            continue
        if date > cutoff:
            out[t.target_id] = "after cutoff"
        elif "classic" in (t.note or "").lower():
            out[t.target_id] = "famous (pre-cutoff)"
        else:
            out[t.target_id] = "obscure (pre-cutoff)"
    return out


SPLIT_ORDER = ["famous (pre-cutoff)", "obscure (pre-cutoff)", "before cutoff", "after cutoff"]


def _summarize_one(res: pd.DataFrame, top_k: int, denominator: set[str]) -> list[dict]:
    """One row per method, scored over ``denominator``: every target attempted.

    A method that produced no pose for a target (a co-folding model that ran out
    of memory, a docking run that crashed) must count that target as a failure,
    not drop it. Scoring each method only over the targets it managed would
    reward failing on the hard ones.
    """
    n = len(denominator)
    rows = []
    for method, grp in res.groupby("method"):
        grp = grp[grp["target_id"].isin(denominator)]
        top1 = grp[grp["rank"] == 1].set_index("target_id")
        topk = grp[grp["rank"] <= top_k].groupby("target_id")
        attempted = grp["target_id"].nunique()
        rows.append({
            "method": method,
            "n_targets": n,
            "n_missing": n - attempted,
            "top1_success": top1["success"].sum() / n,
            "top1_success_valid": (top1["success"] & top1["valid"]).sum() / n,
            f"top{top_k}_success": topk["success"].any().sum() / n,
            "top1_median_rmsd": top1["ligand_rmsd"].median(),
            "top1_recovery_all": top1["recovery_all"].mean(),
            "top1_recovery_polar": top1["recovery_polar"].mean(),
            "top1_pocket_ca_rmsd": top1["pocket_ca_rmsd"].median(),
            "n_errors": int((~grp["status"].isin(["ok", "ligand_graph_mismatch"])).sum()),
        })
    return rows


def missing_poses(results: pd.DataFrame, targets: list[str] | None = None) -> dict[str, list[str]]:
    """method -> targets with no pose at all (silently skipped or crashed)."""
    all_targets = set(targets) if targets else set(results["target_id"].dropna().astype(str))
    out = {}
    for method, grp in results.groupby("method"):
        gone = sorted(all_targets - set(grp["target_id"].astype(str)))
        if gone:
            out[str(method)] = gone
    return out


def summarize(results: pd.DataFrame, top_k: int = 5, subsets: dict[str, str] | None = None,
              targets: list[str] | None = None) -> pd.DataFrame:
    """One row per method (and per subset). Failed or unreadable poses count as failures.

    ``subsets`` maps target_id to a label, e.g. from :func:`time_split`; the
    table then has an ``all`` row per method plus one row per label.
    ``targets`` is the full set attempted (from ``targets.csv``); without it the
    union of target ids present in ``results`` is used. Every method is scored
    over the same denominator, so a target a method never produced counts as a
    failure for it and shows up in ``n_missing``.
    """
    res = results.copy()
    res["success"] = _success(res)
    res["valid"] = (res["graph_ok"] == True) & (res["n_clashes"].fillna(1) == 0)  # noqa: E712
    res["target_id"] = res["target_id"].astype(str)
    all_targets = set(targets) if targets else set(res["target_id"].dropna())
    rows = [{"subset": "all", **r} for r in _summarize_one(res, top_k, all_targets)]
    order = ["all"]
    if subsets:
        norm = {str(k): v for k, v in subsets.items()}
        norm.update({str(k).upper(): v for k, v in subsets.items()})
        labels = set(norm.values())
        for label in ([l for l in SPLIT_ORDER if l in labels]
                      + sorted(labels - set(SPLIT_ORDER))):
            order.append(label)
            sub = {t for t in all_targets if norm.get(t, norm.get(t.upper())) == label}
            rows += [{"subset": label, **r} for r in _summarize_one(res, top_k, sub)]
    df = pd.DataFrame(rows)
    df["_o"] = df["subset"].map(order.index)
    df = df.sort_values(["_o", "top1_success"], ascending=[True, False]).drop(columns="_o")
    cols = ["method", "subset"] + [c for c in df.columns if c not in ("method", "subset")]
    return df[cols].reset_index(drop=True)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(results: pd.DataFrame, summary: pd.DataFrame, outdir: str | Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if "subset" in summary:
        summary = summary[summary["subset"] == "all"]
    colors = method_colors(list(summary["method"]))
    methods = list(summary["method"])
    written = []

    # 1. Top-1 success rate per method (one series, methods on the x axis).
    fig, ax = plt.subplots(figsize=(1.2 * len(methods) + 2, 4.0))
    x = np.arange(len(methods))
    vals = summary["top1_success"].to_numpy() * 100
    valid = summary["top1_success_valid"].to_numpy() * 100
    ax.bar(x, vals, width=0.6, color=[colors[m] for m in methods], edgecolor="white", linewidth=2)
    ax.scatter(x, valid, marker="_", s=600, color=TEXT, linewidths=2, zorder=3,
               label="also physically valid (no clashes, intact ligand)")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 2, f"{v:.0f}%", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 110)
    ax.set_ylabel(f"Top-1 poses within {SUCCESS_RMSD:g} Å (%)", color=TEXT)
    ax.set_title("Pose accuracy by method", color=TEXT, loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    _style(ax)
    fig.tight_layout()
    written.append(outdir / "top1_success.png")
    fig.savefig(written[-1], dpi=200)
    plt.close(fig)

    # 2. Cumulative distribution of top-1 ligand RMSD.
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    top1 = results[results["rank"] == 1]
    for m in methods:
        r = top1.loc[top1["method"] == m, "ligand_rmsd"].to_numpy(dtype=float)
        # Missing and failed poses both stay in the denominator.
        n = int(summary.loc[summary["method"] == m, "n_targets"].iloc[0])
        if n == 0:
            continue
        r = np.sort(r[np.isfinite(r)])
        frac = np.arange(1, len(r) + 1) / n * 100
        xs = np.concatenate([[0], r, [10]])
        ys = np.concatenate([[0], frac, [frac[-1] if len(frac) else 0]])
        ax.step(xs, ys, where="post", color=colors[m], linewidth=2, label=m)
    ax.axvline(SUCCESS_RMSD, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Top-1 ligand RMSD, pocket-aligned (Å)", color=TEXT)
    ax.set_ylabel("Targets (%)", color=TEXT)
    ax.set_title("Cumulative RMSD", color=TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    written.append(outdir / "rmsd_cdf.png")
    fig.savefig(written[-1], dpi=200)
    plt.close(fig)

    # 3. Interaction recovery vs RMSD for top-1 poses.
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for m in methods:
        sel = top1[top1["method"] == m]
        ax.scatter(sel["ligand_rmsd"], sel["recovery_all"] * 100, s=40, color=colors[m],
                   edgecolors="white", linewidths=1.5, label=m)
    ax.axvline(SUCCESS_RMSD, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("Top-1 ligand RMSD (Å)", color=TEXT)
    ax.set_ylabel("Crystal interactions recovered (%)", color=TEXT)
    ax.set_ylim(0, 105)
    ax.set_title("Right place vs right contacts", color=TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    written.append(outdir / "rmsd_vs_recovery.png")
    fig.savefig(written[-1], dpi=200)
    plt.close(fig)
    return written


def to_markdown(summary: pd.DataFrame) -> str:
    cols = list(summary.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for rec in summary.to_dict("records"):
        cells = []
        for c in cols:
            v = rec[c]
            if isinstance(v, float):
                if np.isnan(v):
                    cells.append("-")
                else:
                    cells.append(f"{v * 100:.0f}%" if "success" in c or "recovery" in c else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
