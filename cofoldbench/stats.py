"""Confidence intervals and paired tests for the success rates.

A benchmark on a few dozen targets produces differences that look large and are
not. Three things are needed to read the table honestly:

* a **confidence interval** on every success rate (Wilson score interval, which
  behaves sensibly at 0% and 100% and on small n, unlike the normal
  approximation);
* a **paired test** when two methods are compared on the same targets
  (exact McNemar: only the targets where they disagree carry information);
* an **unpaired interval** on the before/after cut-off difference within one
  method, since those are different targets (Newcombe's method).

No SciPy: the exact binomial and the closed-form intervals are all that is
needed, so these run wherever the rest of the package does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _binom_tail(b: int, n: int) -> float:
    """P(X <= b) for X ~ Binomial(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(b + 1)) / (2 ** n)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    ``b`` = targets the first method got right and the second wrong, ``c`` the
    reverse. Targets both got right (or both wrong) carry no information about
    which is better, which is the point of pairing.
    """
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * _binom_tail(min(b, c), n))


def diff_ci(k1: int, n1: int, k2: int, n2: int, z: float = Z95) -> tuple[float, float]:
    """Newcombe interval for p1 - p2 from two independent samples."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


@dataclass
class Rates:
    """Per-method top-1 successes over a fixed set of targets."""
    success: dict[str, set[str]]   # method -> targets it got right
    targets: set[str]

    def k(self, method: str) -> int:
        return len(self.success.get(method, set()) & self.targets)

    @property
    def n(self) -> int:
        return len(self.targets)


def top1_success(results: pd.DataFrame, success_rmsd: float = 2.0) -> dict[str, set[str]]:
    """method -> set of targets whose top-1 pose is within ``success_rmsd``.

    A target with no pose for a method is absent, i.e. a failure.
    """
    top1 = results[results["rank"] == 1]
    out: dict[str, set[str]] = {}
    for method, grp in top1.groupby("method"):
        ok = grp["ligand_rmsd"].astype(float) <= success_rmsd
        out[str(method)] = set(grp.loc[ok.fillna(False), "target_id"].astype(str))
    return out


def rate_table(results: pd.DataFrame, targets: set[str], success_rmsd: float = 2.0) -> pd.DataFrame:
    """Success rate with a 95% Wilson interval per method."""
    rates = Rates(top1_success(results, success_rmsd), targets)
    rows = []
    for method in sorted(rates.success):
        k = rates.k(method)
        lo, hi = wilson(k, rates.n)
        rows.append({"method": method, "n": rates.n, "successes": k,
                     "rate": k / rates.n if rates.n else float("nan"),
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).sort_values("rate", ascending=False).reset_index(drop=True)


def pairwise(results: pd.DataFrame, targets: set[str], success_rmsd: float = 2.0) -> pd.DataFrame:
    """Exact McNemar for every pair of methods on the same targets."""
    succ = top1_success(results, success_rmsd)
    methods = sorted(succ)
    rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            sa, sb = succ[a] & targets, succ[b] & targets
            only_a = len(sa - sb)
            only_b = len(sb - sa)
            rows.append({"method_a": a, "method_b": b, "a_only": only_a, "b_only": only_b,
                         "both": len(sa & sb), "neither": len(targets - sa - sb),
                         "p_mcnemar": mcnemar_exact(only_a, only_b)})
    return pd.DataFrame(rows).sort_values("p_mcnemar").reset_index(drop=True)


def subset_difference(results: pd.DataFrame, subsets: dict[str, set[str]],
                      first: str, second: str, success_rmsd: float = 2.0) -> pd.DataFrame:
    """Per method: success on ``first`` vs ``second`` subset, with an interval.

    The two subsets are different targets, so this is unpaired: the interval is
    wide, and that is the honest reading of a few dozen targets.
    """
    succ = top1_success(results, success_rmsd)
    a, b = subsets[first], subsets[second]
    rows = []
    for method in sorted(succ):
        ka, kb = len(succ[method] & a), len(succ[method] & b)
        na, nb = len(a), len(b)
        lo, hi = diff_ci(ka, na, kb, nb)
        rows.append({
            "method": method,
            f"{first}_rate": ka / na if na else float("nan"), f"{first}_n": na,
            f"{second}_rate": kb / nb if nb else float("nan"), f"{second}_n": nb,
            "difference": (ka / na - kb / nb) if na and nb else float("nan"),
            "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows).sort_values("difference", ascending=False).reset_index(drop=True)


def format_table(df: pd.DataFrame, percent: tuple[str, ...] = ()) -> str:
    out = df.copy()
    for col in out.columns:
        if col in percent or col.endswith("_rate") or col in ("rate", "difference", "ci_lo", "ci_hi"):
            out[col] = out[col].map(lambda v: "-" if pd.isna(v) else f"{v * 100:+.0f}%"
                                    if col in ("difference", "ci_lo", "ci_hi") and "rate" not in col
                                    else f"{v * 100:.0f}%")
        elif col == "p_mcnemar":
            out[col] = out[col].map(lambda v: f"{v:.4f}")
    return out.to_string(index=False)


def per_target(results: pd.DataFrame, targets: set[str], subsets: dict[str, str] | None = None,
               notes: dict[str, str] | None = None, success_rmsd: float = 2.0) -> pd.DataFrame:
    """Top-1 ligand RMSD per target and method, with the pocket Ca RMSD.

    This is what shows whether a method's failures cluster somewhere: in the
    post-cut-off targets, in the awkward ligands, or in the targets whose
    pocket the model predicted badly (``pocket_<method>`` columns), which would
    make it a protein-prediction failure rather than a ligand-placement one.
    ``.`` means no pose at all for that target.
    """
    top1 = results[results["rank"] == 1].copy()
    top1["target_id"] = top1["target_id"].astype(str)
    rows = []
    for tid in sorted(targets):
        row = {"target_id": tid}
        if subsets:
            row["subset"] = "after" if subsets.get(tid, "").startswith("after") else "before"
        sel = top1[top1["target_id"] == tid]
        for method in sorted(top1["method"].astype(str).unique()):
            m = sel[sel["method"].astype(str) == method]
            if m.empty:
                row[method] = float("nan")
            else:
                row[method] = float(m.iloc[0]["ligand_rmsd"]) if pd.notna(m.iloc[0]["ligand_rmsd"]) \
                    else float("inf")
            if method.startswith("boltz2") or method.startswith("chai"):
                row[f"pocket_{method}"] = (float(m.iloc[0]["pocket_ca_rmsd"])
                                           if not m.empty and pd.notna(m.iloc[0]["pocket_ca_rmsd"])
                                           else float("nan"))
        if notes and tid in notes:
            row["ligand"] = notes[tid]
        rows.append(row)
    return pd.DataFrame(rows)


def format_per_target(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind == "f":
            out[col] = out[col].map(lambda v: "." if pd.isna(v) else
                                    ("broken" if math.isinf(v) else f"{v:.2f}"))
        else:
            out[col] = out[col].fillna("")
    return out.to_string(index=False)
