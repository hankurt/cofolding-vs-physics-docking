"""Find candidate complexes released after a date (the time-split test set).

Queries the RCSB search API for X-ray entries with one protein entity and at
least one small molecule, then the RCSB GraphQL API for their ligands, and
keeps entries with exactly one drug-like ligand (no cofactors, no additives),
one ligand per protein and one protein per ligand so the set stays diverse.

The output is a candidates CSV to review by eye, then pass to
``cofoldbench fetch``, which does the structural checks.

A release date after the training cut-off does not by itself mean the
protein or ligand is new to the models: a close homologue with a similar
ligand may be in the training set. Review the hits (and see the README).
"""

from __future__ import annotations

import json
import random
import urllib.request

import pandas as pd

from .fetch import COFACTORS, USER_AGENT
from .structure import ADDITIVES

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"

GRAPHQL = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_accession_info { initial_release_date }
    rcsb_entry_info { resolution_combined }
    polymer_entities {
      entity_poly { rcsb_sample_sequence_length }
      rcsb_polymer_entity { pdbx_description }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids }
    }
    nonpolymer_entities {
      nonpolymer_comp { chem_comp { id name formula_weight type } }
    }
  }
}
"""


def _post(url: str, payload: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        body = fh.read()
    return json.loads(body) if body else {}


def _terminal(attribute: str, operator: str, value) -> dict:
    return {"type": "terminal", "service": "text",
            "parameters": {"attribute": attribute, "operator": operator, "value": value}}


def search_query(after: str, max_resolution: float, rows: int, before: str | None = None,
                 start: int = 0) -> dict:
    """Entries released after ``after`` and, with ``before``, strictly before it."""
    nodes = [
        _terminal("rcsb_accession_info.initial_release_date", "greater", after),
        _terminal("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
        _terminal("rcsb_entry_info.resolution_combined", "less_or_equal", max_resolution),
        _terminal("rcsb_entry_info.polymer_entity_count_protein", "equals", 1),
        _terminal("rcsb_entry_info.polymer_entity_count_nucleic_acid", "equals", 0),
        _terminal("rcsb_entry_info.nonpolymer_entity_count", "greater_or_equal", 1),
    ]
    if before:
        nodes.append(_terminal("rcsb_accession_info.initial_release_date", "less", before))
    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "desc"}],
        },
    }


def year_windows(after: str, before: str) -> list[tuple[str, str]]:
    """Split a date range into one window per calendar year.

    Results are sorted newest first and the API returns only the first page, so
    a single query over a wide range gives only its most recent end: asking for
    everything since 2015 returns last month's entries. One query per year
    spreads the sample over the whole range without deep pagination.
    """
    y0, y1 = int(after[:4]), int(before[:4])
    out = []
    for y in range(y0, y1 + 1):
        lo = after if y == y0 else f"{y}-01-01"
        hi = before if y == y1 else f"{y + 1}-01-01"
        if lo < hi:
            out.append((lo, hi))
    return out


# Ligand names that mark a crystallisation additive or detergent whatever its
# CCD code: the code list cannot keep up with every oligo-ether variant.
ADDITIVE_WORDS = ("glycol", "polyethylene", "peg", "detergent", "glucoside",
                  "maltoside", "glucopyranos", "maltopyranos", "pyranoside",
                  "thioglucos", "triton", "tween", "chaps", "buffer", "hepes",
                  "tris(hydroxymethyl)", "cryoprotect", "lauryl", "octyl beta",
                  "nonyl", "decyl", "dodecyl")


def pick_ligand(entry: dict, min_mw: float, max_mw: float) -> dict | None:
    """The single drug-like ligand of an entry, or None."""
    drug_like, cofactor = [], False
    for ent in entry.get("nonpolymer_entities") or []:
        cc = ((ent.get("nonpolymer_comp") or {}).get("chem_comp")) or {}
        cid = (cc.get("id") or "").upper()
        if not cid or cid in ADDITIVES:
            continue
        if cid in COFACTORS:
            cofactor = True
            continue
        name = (cc.get("name") or "").lower()
        if any(w in name for w in ADDITIVE_WORDS):
            continue
        mw = cc.get("formula_weight") or 0
        if min_mw <= mw <= max_mw and "PEPTIDE" not in (cc.get("type") or "").upper():
            drug_like.append(cc)
    if cofactor or len(drug_like) != 1:
        return None
    return drug_like[0]


def filter_entries(entries: list[dict], min_mw: float = 250, max_mw: float = 700,
                   min_length: int = 80, max_length: int = 700,
                   label: str = "recent") -> pd.DataFrame:
    rows, seen_protein, seen_ligand = [], set(), set()
    for e in entries:
        if not e:
            continue
        poly = e.get("polymer_entities") or []
        if len(poly) != 1:
            continue
        length = ((poly[0].get("entity_poly") or {}).get("rcsb_sample_sequence_length")) or 0
        if not min_length <= length <= max_length:
            continue
        lig = pick_ligand(e, min_mw, max_mw)
        if lig is None:
            continue
        protein = ((poly[0].get("rcsb_polymer_entity") or {}).get("pdbx_description") or "").strip()
        key = protein.upper()
        if key in seen_protein or lig["id"] in seen_ligand:
            continue
        seen_protein.add(key)
        seen_ligand.add(lig["id"])
        chains = (poly[0].get("rcsb_polymer_entity_container_identifiers") or {}).get("auth_asym_ids") or []
        res = (e.get("rcsb_entry_info") or {}).get("resolution_combined") or [None]
        rows.append({
            "pdb_id": e["rcsb_id"],
            "chain": sorted(chains)[0] if chains else "",
            "ligand_resname": lig["id"],
            "protein": protein,
            "ligand_name": (lig.get("name") or "").lower(),
            "release_date": ((e.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10],
            "resolution": res[0],
            "length": length,
            "set": label,
        })
    return pd.DataFrame(rows)


def search(after: str, n: int = 20, max_resolution: float = 2.0, pool: int = 1000,
           before: str | None = None, seed: int = 0, **filters) -> pd.DataFrame:
    """Candidate entries released in a date range.

    Without ``before`` this is the newest ``pool`` entries after ``after``: what
    you want for a time-split set, where the newest structures are the point.

    With ``before`` the range is sampled year by year. Results come back newest
    first, so taking each year's first page would return only its last days
    (every hit in late December). Instead each year is sampled from a random
    offset into its own hit list, which spreads the sample over the whole range.
    ``seed`` makes that reproducible.
    """
    windows = year_windows(after, before) if before else [(after, None)]
    rows = max(50, pool // len(windows))
    rng = random.Random(seed)
    ids: list[str] = []
    for lo, hi in windows:
        start = 0
        if hi:
            head = _post(SEARCH_URL, search_query(lo, max_resolution, 1, hi))
            total = int(head.get("total_count") or 0)
            if total > rows:
                # Deep paging is capped by the service, so stay well inside it.
                start = rng.randrange(0, min(total - rows, 9000) + 1)
        hits = _post(SEARCH_URL, search_query(lo, max_resolution, rows, hi, start))
        ids += [h["identifier"] for h in hits.get("result_set", [])]
    entries = []
    for i in range(0, len(ids), 100):
        data = _post(GRAPHQL_URL, {"query": GRAPHQL, "variables": {"ids": ids[i:i + 100]}})
        entries += (data.get("data") or {}).get("entries") or []
    df = filter_entries(entries, label="pre-cutoff" if before else "recent", **filters)
    if before and len(df) > n:
        df = df.sort_values("release_date")
        step = len(df) / n
        df = df.iloc[[int(i * step) for i in range(n)]]
    return df.head(n).reset_index(drop=True)
