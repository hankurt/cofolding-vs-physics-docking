"""Score every pose in a manifest against its reference structure."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import contacts
from .ligand import centroid_distance, symmetric_rmsd, validity
from .structure import load_pose, write_complex_pdb
from .superpose import superpose_on_pocket, transform_complex
from .targets import Target

RESULT_COLUMNS = [
    "target_id", "method", "rank", "score", "score_name", "status",
    "ligand_rmsd", "centroid_dist", "pocket_ca_rmsd", "n_pocket_ca",
    "min_dist", "n_clashes", "graph_ok",
    "recovery_all", "recovery_hydrophobic", "recovery_polar", "recovery_ionic",
    "fp_tanimoto", "n_ref_interactions", "aligned_pdb",
]


def evaluate_pose(target: Target, row: dict, aligned_dir: Path | None = None,
                  pocket_cutoff: float = 10.0) -> dict:
    ref = target.reference
    out = {k: row.get(k) for k in ("target_id", "method", "rank", "score", "score_name")}
    out["target_id"] = target.target_id
    mdl = load_pose(row["structure"], row.get("ligand") or None, int(row.get("pose_index") or 0))
    if mdl.ligand is None:
        return {**out, "status": "no_ligand"}

    sp = superpose_on_pocket(ref, mdl, pocket_cutoff, ref_chains=target.chains)
    aligned = transform_complex(mdl, sp)

    val = validity(aligned, ref.ligand)
    ref_fp = contacts.fingerprint(ref)
    pose_fp = contacts.fingerprint(aligned)
    out.update(
        status="ok" if val.graph_matches_reference else "ligand_graph_mismatch",
        ligand_rmsd=symmetric_rmsd(aligned.ligand, ref.ligand),
        centroid_dist=centroid_distance(aligned.ligand, ref.ligand),
        pocket_ca_rmsd=sp.pocket_ca_rmsd,
        n_pocket_ca=sp.n_pocket_ca,
        min_dist=val.min_distance,
        n_clashes=val.n_clashes,
        graph_ok=val.graph_matches_reference,
        recovery_all=contacts.recovery(ref_fp, pose_fp),
        recovery_hydrophobic=contacts.recovery(ref_fp, pose_fp, "hydrophobic"),
        recovery_polar=contacts.recovery(ref_fp, pose_fp, "polar"),
        recovery_ionic=contacts.recovery(ref_fp, pose_fp, "ionic"),
        fp_tanimoto=contacts.tanimoto(ref_fp, pose_fp),
        n_ref_interactions=len(ref_fp),
    )
    if aligned_dir is not None:
        dest = aligned_dir / target.target_id / f"{row['method']}_rank{int(row['rank'])}.pdb"
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_complex_pdb(aligned, dest, ligand_resname=target.ligand_resname[:3])
        out["aligned_pdb"] = str(dest)
    return out


def evaluate(targets: list[Target], manifest: pd.DataFrame, aligned_dir: str | Path | None = None,
             max_rank: int | None = None, pocket_cutoff: float = 10.0, verbose: bool = False) -> pd.DataFrame:
    by_id = {t.target_id.lower(): t for t in targets}
    aligned_dir = Path(aligned_dir) if aligned_dir else None
    rows = []
    for row in manifest.to_dict("records"):
        if max_rank and int(row["rank"]) > max_rank:
            continue
        target = by_id.get(str(row["target_id"]).lower())
        if target is None:
            rows.append({**row, "status": "unknown_target"})
            continue
        try:
            rows.append(evaluate_pose(target, row, aligned_dir, pocket_cutoff))
        except Exception as exc:  # one broken pose must not stop the benchmark
            if verbose:
                traceback.print_exc()
            rows.append({**{k: row.get(k) for k in ("method", "rank", "score", "score_name")},
                         "target_id": target.target_id, "status": f"error: {exc}"})
    df = pd.DataFrame(rows)
    for col in RESULT_COLUMNS:
        if col not in df:
            df[col] = np.nan
    return df[RESULT_COLUMNS]
