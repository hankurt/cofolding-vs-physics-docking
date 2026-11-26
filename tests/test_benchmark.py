"""End-to-end checks on the AmpC / H7M example (data/structures/6dpt_A.pdb)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from cofoldbench import collect, contacts, inputs, report
from cofoldbench.evaluate import evaluate
from cofoldbench.ligand import symmetric_rmsd
from cofoldbench.structure import (Complex, Residue, ligand_coords, read_complex, read_pdbqt_poses,
                                   set_ligand_coords, write_complex_pdb)
from cofoldbench.targets import load_targets

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "data" / "targets.csv"


@pytest.fixture(scope="module")
def target():
    # By id, not by row: data/targets.csv holds the whole benchmark set.
    return next(t for t in load_targets(TARGETS) if t.target_id == "6DPT")


def single_target_csv(tmp_path, target, smiles=None) -> Path:
    """A targets.csv holding just this target, for tests that score over it.

    data/targets.csv is the whole benchmark set, so a test must not depend on
    its row order or its length (the report's denominator is every target in it).
    """
    import csv as _csv
    from cofoldbench.fetch import TARGET_COLUMNS
    row = {
        "target_id": target.target_id,
        "structure": str(target.structure),          # absolute: resolved as-is
        "chain": " ".join(target.chains),
        "ligand_resname": target.ligand_resname,
        "smiles": smiles or target.smiles,
        "pocket_residues": " ".join(map(str, target.pocket_residues)),
        "sequence": target.sequence or "",
        "release_date": target.release_date or "",
        "resolution": "", "note": "",
    }
    path = tmp_path / "one_target.csv"
    with path.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=TARGET_COLUMNS)
        w.writeheader()
        w.writerow(row)
    return path


def moved(cx: Complex, rot: np.ndarray, shift: np.ndarray, offset: int = 0,
          chain: str | None = None, ligand_shift: np.ndarray | None = None) -> Complex:
    """Rigidly move a complex, renumber it, and optionally displace the ligand."""
    res = [Residue(chain or r.chain, r.number + offset, r.icode, r.name, r.atom_names, r.elements,
                   r.coords @ rot.T + shift) for r in cx.residues]
    lig = ligand_coords(cx.ligand)
    if ligand_shift is not None:
        lig = lig + ligand_shift
    return Complex(res, set_ligand_coords(cx.ligand, lig @ rot.T + shift))


def run(target, tmp_path, cx: Complex, method="test", ligand_file=None) -> dict:
    pdb = tmp_path / f"{method}.pdb"
    write_complex_pdb(cx, pdb, ligand_resname="LIG")
    row = collect.single(target.target_id, pdb, method, ligand_file)
    df = evaluate([target], pd.DataFrame(row), aligned_dir=tmp_path / "aligned")
    return df.iloc[0].to_dict()


def test_reference_loads(target):
    ref = target.reference
    assert ref.ligand.GetNumAtoms() == 25
    assert len(ref.residues) == 358


def test_self_is_perfect(target, tmp_path):
    r = run(target, tmp_path, target.reference)
    assert r["status"] == "ok"
    assert r["ligand_rmsd"] == pytest.approx(0, abs=1e-3)
    assert r["pocket_ca_rmsd"] == pytest.approx(0, abs=1e-3)
    assert r["recovery_all"] == pytest.approx(1.0)
    assert r["n_clashes"] == 0


def test_frame_and_numbering_do_not_matter(target, tmp_path):
    """A co-folding model sits in its own frame, numbered from 1, on chain B."""
    rot = Rotation.from_euler("xyz", [40, -75, 120], degrees=True).as_matrix()
    cx = moved(target.reference, rot, np.array([31.0, -12.0, 7.5]), offset=-3, chain="B")
    r = run(target, tmp_path, cx)
    assert r["ligand_rmsd"] == pytest.approx(0, abs=1e-2)
    assert r["recovery_all"] == pytest.approx(1.0)


def test_ligand_displacement_is_measured(target, tmp_path):
    cx = moved(target.reference, np.eye(3), np.zeros(3), ligand_shift=np.array([1.5, 0, 0]))
    r = run(target, tmp_path, cx)
    assert r["ligand_rmsd"] == pytest.approx(1.5, abs=1e-2)
    assert r["centroid_dist"] == pytest.approx(1.5, abs=1e-2)


def test_far_pose_loses_interactions(target, tmp_path):
    cx = moved(target.reference, np.eye(3), np.zeros(3), ligand_shift=np.array([0, 0, 25.0]))
    r = run(target, tmp_path, cx)
    assert r["ligand_rmsd"] > 20
    assert r["recovery_all"] < 0.2


def test_atom_order_does_not_matter(target):
    lig = target.reference.ligand
    perm = list(reversed(range(lig.GetNumAtoms())))
    from rdkit import Chem
    shuffled = Chem.RenumberAtoms(lig, perm)
    assert symmetric_rmsd(shuffled, lig) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("distance", [6.0, 40.0])
def test_broken_ligand_is_flagged(target, tmp_path, distance):
    """A torn-off atom must not match as a subgraph of the reference."""
    lig = ligand_coords(target.reference.ligand).copy()
    lig[0] += np.array([distance, 0, 0])
    cx = Complex(target.reference.residues, set_ligand_coords(target.reference.ligand, lig))
    r = run(target, tmp_path, cx)
    assert r["status"] == "ligand_graph_mismatch"
    assert np.isnan(r["ligand_rmsd"])


def test_separate_ligand_file_pdbqt(target, tmp_path):
    """Docking output: receptor PDB + ligand poses in PDBQT, as Vina writes them."""
    lig = target.reference.ligand
    pos = ligand_coords(lig)
    ad = {"C": "C", "N": "NA", "O": "OA", "S": "SA", "Cl": "Cl"}
    lines = []
    for model, shift in enumerate([0.0, 2.0], 1):
        lines += [f"MODEL {model}", f"REMARK VINA RESULT:    -{9 - model}.1      0.000      0.000"]
        for i, (a, xyz) in enumerate(zip(lig.GetAtoms(), pos + [shift, 0, 0]), 1):
            lines.append(f"ATOM  {i:5d}  {a.GetSymbol():<3} LIG A   1    "
                         f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00     0.000 {ad[a.GetSymbol()]:<2}")
        lines.append("ENDMDL")
    pdbqt = tmp_path / "out.pdbqt"
    pdbqt.write_text("\n".join(lines) + "\n")
    assert [s["vina_affinity"] for _, s in read_pdbqt_poses(pdbqt)] == [-8.1, -7.1]

    receptor = tmp_path / "rec.pdb"
    write_complex_pdb(Complex(target.reference.residues, None), receptor)
    rows = collect.vina(target.target_id, receptor, pdbqt)
    df = evaluate([target], pd.DataFrame(rows))
    assert list(df["rank"]) == [1, 2]
    assert df["ligand_rmsd"].tolist() == pytest.approx([0.0, 2.0], abs=1e-2)


def test_boltz_collector_and_case_insensitive_ids(target, tmp_path):
    pred = tmp_path / "boltz_results_x" / "predictions" / "6dpt"
    pred.mkdir(parents=True)
    rot = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    for k, shift in enumerate([0.0, 3.0]):
        cx = moved(target.reference, rot, np.array([5.0, 5, 5]), offset=-3,
                   ligand_shift=np.array([shift, 0, 0]))
        write_complex_pdb(cx, pred / f"6dpt_model_{k}.pdb")
        (pred / f"confidence_6dpt_model_{k}.json").write_text(json.dumps({"confidence_score": 0.9 - k / 10}))
    rows = collect.boltz(tmp_path)
    assert [r["rank"] for r in rows] == [1, 2]
    df = evaluate([target], pd.DataFrame(rows))
    assert (df["target_id"] == "6DPT").all()
    assert df["ligand_rmsd"].tolist() == pytest.approx([0.0, 3.0], abs=1e-2)

    summary = report.summarize(df)
    assert summary.iloc[0]["top1_success"] == 1.0
    assert summary.iloc[0]["top1_success_valid"] == 1.0


def test_aligned_pdb_is_readable(target, tmp_path):
    rot = Rotation.from_euler("y", 60, degrees=True).as_matrix()
    r = run(target, tmp_path, moved(target.reference, rot, np.array([1.0, 2, 3]), offset=10))
    back = read_complex(r["aligned_pdb"])
    assert back.residues[0].number == target.reference.residues[0].number
    assert contacts.recovery(contacts.fingerprint(target.reference), contacts.fingerprint(back)) == 1.0


def test_inputs(target, tmp_path):
    seqs, index = inputs.sequences_and_index(target)
    assert seqs["A"][index[("A", 64)] - 1] == "S"   # AmpC catalytic Ser64
    yaml = inputs.boltz_yaml(target, pocket=True)
    assert "[A, 61]" in yaml and "binder: L" in yaml
    job = json.loads(inputs.af3_json(target))
    assert job["sequences"][1]["ligand"]["smiles"] == target.smiles
    assert inputs.write_inputs([target], tmp_path) == []
    assert (tmp_path / "chai" / "6DPT.fasta").read_text().startswith(">protein|name=6DPT_A")


def test_sequence_override_shifts_pocket_index(target):
    from dataclasses import replace
    t = replace(target, sequence="MKT" + target.chain_sequences()["A"])
    _, index = inputs.sequences_and_index(t)
    assert index[("A", 64)] == 64


def test_cli_end_to_end(target, tmp_path):
    """collect -> evaluate -> report through the command line."""
    from cofoldbench.cli import main
    rot = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    for method, shift in [("boltz2", 0.8), ("gnina", 1.6), ("vina", 4.0)]:
        pdb = tmp_path / f"{method}.pdb"
        write_complex_pdb(moved(target.reference, rot, np.ones(3), ligand_shift=np.array([shift, 0, 0])), pdb)
        main(["collect", "single", "--target", "6DPT", "--structure", str(pdb), "--method", method,
              "--manifest", str(tmp_path / "manifest.csv")])
    csv = single_target_csv(tmp_path, target)
    main(["evaluate", "--targets", str(csv), "--manifest", str(tmp_path / "manifest.csv"),
          "--out", str(tmp_path / "results.csv"), "--aligned-dir", str(tmp_path / "aligned")])
    main(["report", "--results", str(tmp_path / "results.csv"), "--outdir", str(tmp_path / "report"),
          "--targets", str(csv)])
    summary = pd.read_csv(tmp_path / "report" / "summary.csv")
    # One row per method and subset now, so index by both.
    summary = summary.set_index(["method", "subset"])
    assert summary.loc[("boltz2", "all"), "top1_success"] == 1.0
    assert summary.loc[("vina", "all"), "top1_success"] == 0.0
    assert summary.loc[("vina", "all"), "n_missing"] == 0
    for name in ("top1_success.png", "rmsd_cdf.png", "rmsd_vs_recovery.png", "summary.md"):
        assert (tmp_path / "report" / name).exists()
