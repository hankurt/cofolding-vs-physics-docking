"""Docking inputs for every target: receptor, starting ligand, search box.

Per target, ``inputs/dock/<target>/`` gets

* ``receptor.pdb``  protein heavy atoms of the reference (redocking into the
  holo receptor: the best case for docking; cross-docking is on the roadmap);
* ``ligand.sdf``    a 3D conformer generated **from the SMILES** with RDKit
  (with hydrogens), never the crystal pose, so docking starts as blind to the
  answer as the co-folding models are;
* ``box.txt``       Vina/GNINA box built on the **same pocket residues** that
  condition Boltz-2, so both families get the same binding-site hint.

The files are tool-neutral: ``scripts/run_docking.sh`` adds hydrogens and
writes PDBQT for Vina in the docking environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .structure import Complex, write_complex_pdb
from .targets import Target

BOX_PADDING = 6.0   # A added around the pocket residues' side chains
BOX_MIN, BOX_MAX = 18.0, 30.0


def ligand_3d(smiles: str, seed: int = 42) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"could not embed {smiles}")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    return mol


def pocket_box(t: Target) -> tuple[np.ndarray, np.ndarray]:
    """Box centre and size from the pocket residues' side-chain atoms."""
    pocket = set(t.pocket_residues)
    backbone = {"N", "C", "O", "OXT"}
    xyz = [c for r in t.reference.residues
           if r.chain == t.chains[0] and r.number in pocket
           for name, c in zip(r.atom_names, r.coords) if name not in backbone]
    if not xyz:
        raise ValueError(f"{t.target_id}: no pocket residues - fill the pocket_residues column")
    xyz = np.array(xyz)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    center = (lo + hi) / 2
    size = np.clip(hi - lo + BOX_PADDING, BOX_MIN, BOX_MAX)
    return center, size


def box_text(center: np.ndarray, size: np.ndarray) -> str:
    keys = ("x", "y", "z")
    lines = [f"center_{k} = {v:.3f}" for k, v in zip(keys, center)]
    lines += [f"size_{k} = {v:.1f}" for k, v in zip(keys, size)]
    return "\n".join(lines) + "\n"


def write_dock_inputs(targets: list[Target], outdir: str | Path, seed: int = 42) -> list[Path]:
    outdir = Path(outdir)
    written = []
    for t in targets:
        d = outdir / t.target_id
        d.mkdir(parents=True, exist_ok=True)
        write_complex_pdb(Complex(t.reference.residues, None), d / "receptor.pdb")
        mol = ligand_3d(t.smiles, seed)
        mol.SetProp("_Name", f"{t.target_id}_{t.ligand_resname}")
        with Chem.SDWriter(str(d / "ligand.sdf")) as w:
            w.write(mol)
        center, size = pocket_box(t)
        (d / "box.txt").write_text(box_text(center, size))
        written.append(d)
    return written


def write_pdbqt(d: str | Path, ph: float = 7.4) -> list[Path]:
    """Hydrogens and PDBQT files for one ``inputs/dock/<target>`` directory.

    Needs the ``dock`` extra (``pip install -e ".[dock]"``: Open Babel + Meeko).
    Writes ``receptor_h.pdb`` (all hydrogens at ``ph``, for GNINA),
    ``receptor.pdbqt`` (polar hydrogens, rigid, for Vina) and ``ligand.pdbqt``
    (Meeko torsion tree, for Vina).
    """
    from openbabel import pybel

    pybel.ob.obErrorLog.SetOutputLevel(0)
    d = Path(d)
    rec = next(pybel.readfile("pdb", str(d / "receptor.pdb")))
    rec.OBMol.AddHydrogens(False, True, ph)
    rec.write("pdb", str(d / "receptor_h.pdb"), overwrite=True)
    rec = next(pybel.readfile("pdb", str(d / "receptor.pdb")))
    rec.OBMol.AddHydrogens(True, True, ph)
    rec.write("pdbqt", str(d / "receptor.pdbqt"), overwrite=True, opt={"r": None})

    mol = Chem.SDMolSupplier(str(d / "ligand.sdf"), removeHs=False)[0]
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    setups = MoleculePreparation().prepare(mol)
    text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(f"meeko could not write {d / 'ligand.pdbqt'}: {err}")
    (d / "ligand.pdbqt").write_text(text)
    return [d / "receptor_h.pdb", d / "receptor.pdbqt", d / "ligand.pdbqt"]
