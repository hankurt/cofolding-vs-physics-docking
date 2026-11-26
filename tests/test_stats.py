"""Checks on the confidence intervals and paired tests."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from cofoldbench import stats


def test_wilson_known_values():
    # Textbook Wilson interval, and sane behaviour at the boundaries.
    lo, hi = stats.wilson(4, 16)
    assert (lo, hi) == pytest.approx((0.10182, 0.49502), abs=1e-4)
    assert stats.wilson(0, 16)[0] == 0.0          # never negative
    assert stats.wilson(16, 16)[1] == 1.0         # never above 1
    assert stats.wilson(0, 16)[1] > 0             # 0/16 is not "certainly 0%"
    assert stats.wilson(16, 16)[0] < 1
    assert all(math.isnan(v) for v in stats.wilson(0, 0))
    # Wider with less data.
    narrow = stats.wilson(50, 100)
    wide = stats.wilson(5, 10)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_mcnemar_exact():
    # All discordant pairs favour one method: 2 * (1/2)^7.
    assert stats.mcnemar_exact(7, 0) == pytest.approx(2 / 128)
    assert stats.mcnemar_exact(0, 7) == pytest.approx(2 / 128)
    assert stats.mcnemar_exact(0, 0) == 1.0       # no information
    assert stats.mcnemar_exact(5, 4) == 1.0       # as even as it gets
    assert stats.mcnemar_exact(10, 2) < 0.05
    assert stats.mcnemar_exact(6, 4) > 0.05       # a 60/40 split of 10 proves little


def test_diff_ci_excludes_zero_only_for_a_real_difference():
    lo, hi = stats.diff_ci(22, 23, 4, 16)         # 96% vs 25%
    assert lo > 0 and (lo, hi) == pytest.approx((0.410, 0.859), abs=1e-3)
    lo, hi = stats.diff_ci(17, 23, 11, 16)        # 74% vs 69%: no real difference
    assert lo < 0 < hi


def _results(success: dict[str, list[str]], all_targets: list[str]) -> pd.DataFrame:
    """Top-1 rows: RMSD 1.0 for a success, 5.0 for a miss, absent for no pose."""
    rows = []
    for method, good in success.items():
        for t in all_targets:
            if t == "NOPOSE" and method == "chai1":
                continue                            # ran out of memory
            rows.append({"target_id": t, "method": method, "rank": 1,
                         "ligand_rmsd": 1.0 if t in good else 5.0})
    return pd.DataFrame(rows)


def test_top1_success_and_missing_pose():
    res = _results({"boltz2": ["A", "B"], "chai1": ["A"]}, ["A", "B", "NOPOSE"])
    succ = stats.top1_success(res)
    assert succ["boltz2"] == {"A", "B"}
    assert succ["chai1"] == {"A"}
    table = stats.rate_table(res, {"A", "B", "NOPOSE"}).set_index("method")
    # chai1 has no pose for NOPOSE: counted in n, not in successes.
    assert table.loc["chai1", "n"] == 3 and table.loc["chai1", "successes"] == 1


def test_pairwise_and_subset_difference():
    targets = [f"T{i}" for i in range(16)]
    # gnina right on 11, boltz2 on 4 of those same targets, nested so b_only = 7.
    res = _results({"gnina": targets[:11], "boltz2": targets[:4]}, targets)
    pw = stats.pairwise(res, set(targets)).set_index(["method_a", "method_b"])
    row = pw.loc[("boltz2", "gnina")]
    assert row["a_only"] == 0 and row["b_only"] == 7
    assert row["p_mcnemar"] == pytest.approx(2 / 128)   # significant despite n=16

    before, after = set(targets[:8]), set(targets[8:])
    diff = stats.subset_difference(res, {"before cutoff": before, "after cutoff": after},
                                   "before cutoff", "after cutoff").set_index("method")
    # boltz2: 4/8 before, 0/8 after.
    assert diff.loc["boltz2", "before cutoff_rate"] == 0.5
    assert diff.loc["boltz2", "after cutoff_rate"] == 0.0


def test_nan_rmsd_counts_as_failure():
    """A broken ligand graph gives NaN RMSD; that is a miss, not a skip."""
    res = pd.DataFrame([{"target_id": "A", "method": "m", "rank": 1, "ligand_rmsd": float("nan")}])
    assert stats.top1_success(res)["m"] == set()


def test_exposure_split_separates_famous_from_obscure():
    """Pre-cut-off is two conditions: heavily represented, and merely seen."""
    from dataclasses import dataclass
    from cofoldbench import report

    @dataclass
    class T:
        target_id: str
        release_date: str | None
        note: str | None

    split = report.exposure_split([
        T("1IEP", "2001-04-18", "ABL kinase; imatinib; classic"),
        T("4UNR", "2015-06-17", "thymidylate kinase; benzoquinoline; pre-cutoff"),
        T("9T1T", "2026-11-04", "BTK kinase; pre-cutoff pair"),   # date wins over the note
        T("NODATE", None, "classic"),
    ])
    assert split == {"1IEP": "famous (pre-cutoff)",
                     "4UNR": "obscure (pre-cutoff)",
                     "9T1T": "after cutoff"}

    # Subsets come out in exposure order, not alphabetically.
    rows = []
    for tid in ("1IEP", "4UNR", "9T1T"):
        rows.append(dict(target_id=tid, method="boltz2", rank=1, ligand_rmsd=1.0,
                         graph_ok=True, n_clashes=0, status="ok", recovery_all=0.8,
                         recovery_polar=0.8, pocket_ca_rmsd=0.3))
    summary = report.summarize(pd.DataFrame(rows), subsets=split,
                               targets=["1IEP", "4UNR", "9T1T"])
    assert list(summary["subset"]) == ["all", "famous (pre-cutoff)",
                                       "obscure (pre-cutoff)", "after cutoff"]
