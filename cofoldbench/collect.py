"""Gather poses from each engine's native output into one manifest CSV.

Manifest columns: target_id, method, rank, structure, ligand, pose_index,
score, score_name. ``rank`` is the engine's own ranking (1 = the pose the
method would pick); ``ligand`` is set when the ligand lives in a separate
file from the receptor (GNINA SDF, Vina PDBQT).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .structure import read_pdbqt_poses, read_sdf_poses

COLUMNS = ["target_id", "method", "rank", "structure", "ligand", "pose_index", "score", "score_name"]


def _row(target, method, rank, structure, ligand=None, pose_index=0, score=np.nan, score_name=""):
    return dict(target_id=target, method=method, rank=int(rank), structure=str(structure),
                ligand=str(ligand) if ligand else "", pose_index=int(pose_index),
                score=score, score_name=score_name)


def boltz(results_dir: str | Path, method: str = "boltz2") -> list[dict]:
    """Boltz-1/2 output: ``*/predictions/<name>/<name>_model_<k>.cif``.

    Models are already sorted by confidence (model_0 is the top pick).
    """
    rows = []
    for model in sorted(Path(results_dir).glob("**/predictions/*/*_model_*.*")):
        if model.suffix not in (".cif", ".pdb"):
            continue
        name = model.parent.name
        k = int(re.search(r"_model_(\d+)", model.name).group(1))
        conf = model.parent / f"confidence_{name}_model_{k}.json"
        score = json.loads(conf.read_text()).get("confidence_score", np.nan) if conf.exists() else np.nan
        rows.append(_row(name, method, k + 1, model, score=score, score_name="confidence_score"))
    return rows


def chai(results_dir: str | Path, method: str = "chai1") -> list[dict]:
    """Chai-1 output, one directory per target: ``<target>/pred.model_idx_<k>.cif``.

    Ranked by ``aggregate_score`` from the matching ``scores.model_idx_<k>.npz``.
    """
    rows = []
    for tdir in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        models = []
        for model in tdir.glob("pred.model_idx_*.cif"):
            k = int(re.search(r"model_idx_(\d+)", model.name).group(1))
            npz = tdir / f"scores.model_idx_{k}.npz"
            score = float(np.asarray(np.load(npz)["aggregate_score"]).ravel()[0]) if npz.exists() else np.nan
            models.append((score, model))
        models.sort(key=lambda sm: -sm[0] if not np.isnan(sm[0]) else np.inf)
        for rank, (score, model) in enumerate(models, 1):
            rows.append(_row(tdir.name, method, rank, model, score=score, score_name="aggregate_score"))
    return rows


def af3(results_dir: str | Path, method: str = "af3") -> list[dict]:
    """AlphaFold3 output: ``<job>/seed-<s>_sample-<k>/model.cif``.

    Ranked by ``ranking_score`` from each sample's ``summary_confidences.json``.
    AF3 lower-cases job directory names; evaluation matches target ids
    case-insensitively.
    """
    rows = []
    for jdir in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        models = []
        for sdir in jdir.glob("seed-*_sample-*"):
            model = sdir / "model.cif"
            summ = sdir / "summary_confidences.json"
            if not model.exists():
                continue
            score = json.loads(summ.read_text()).get("ranking_score", np.nan) if summ.exists() else np.nan
            models.append((score, model))
        models.sort(key=lambda sm: -sm[0] if not np.isnan(sm[0]) else np.inf)
        for rank, (score, model) in enumerate(models, 1):
            rows.append(_row(jdir.name, method, rank, model, score=score, score_name="ranking_score"))
    return rows


def gnina(target: str, receptor: str | Path, poses: str | Path, method: str = "gnina",
          score_name: str = "CNNscore") -> list[dict]:
    """GNINA SDF output; poses are kept in file order (GNINA's own ranking)."""
    rows = []
    for k, (_, scores) in enumerate(read_sdf_poses(poses)):
        rows.append(_row(target, method, k + 1, receptor, poses, k,
                         score=scores.get(score_name, np.nan), score_name=score_name))
    return rows


def vina(target: str, receptor: str | Path, poses: str | Path, method: str = "vina") -> list[dict]:
    """Vina / AutoDock-GPU PDBQT output; poses in file order."""
    rows = []
    for k, (_, scores) in enumerate(read_pdbqt_poses(poses)):
        rows.append(_row(target, method, k + 1, receptor, poses, k,
                         score=scores.get("vina_affinity", np.nan), score_name="vina_affinity"))
    return rows


def haddock3(target: str, run_dir: str | Path, method: str = "haddock3") -> list[dict]:
    """HADDOCK3 run: poses and ranks from the last ``caprieval`` step's ``capri_ss.tsv``."""
    run_dir = Path(run_dir)
    steps = sorted(run_dir.glob("*_caprieval"), key=lambda p: int(p.name.split("_")[0]))
    if not steps:
        raise FileNotFoundError(f"no caprieval step in {run_dir}")
    table = pd.read_csv(steps[-1] / "capri_ss.tsv", sep="\t")
    rows = []
    for rec in table.sort_values("caprieval_rank").to_dict("records"):
        model = (steps[-1] / rec["model"]).resolve()
        if not model.exists() and Path(str(model) + ".gz").exists():
            model = Path(str(model) + ".gz")
        rows.append(_row(target, method, rec["caprieval_rank"], model,
                         score=rec.get("score", np.nan), score_name="haddock_score"))
    return rows


def single(target: str, structure: str | Path, method: str, ligand: str | Path | None = None,
           rank: int = 1) -> list[dict]:
    """Any single pose (complex file, or receptor + ligand file)."""
    return [_row(target, method, rank, structure, ligand)]


def write_manifest(rows: list[dict], path: str | Path, append: bool = False) -> pd.DataFrame:
    path = Path(path)
    df = pd.DataFrame(rows, columns=COLUMNS)
    if append and path.exists():
        old = pd.read_csv(path, dtype={"ligand": str}, keep_default_na=False)
        keys = set(zip(df.target_id, df.method))
        old = old[[(t, m) not in keys for t, m in zip(old.target_id, old.method)]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    return df
