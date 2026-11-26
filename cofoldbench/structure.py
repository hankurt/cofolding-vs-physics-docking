"""Read protein-ligand complexes from any engine's output into one representation.

Co-folding models write mmCIF, HADDOCK3 writes PDB complexes, GNINA writes SDF
poses and Vina writes PDBQT poses. Everything is reduced to a list of protein
residues plus one ligand as an RDKit molecule (heavy atoms, one conformer).
"""

from __future__ import annotations

import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

WATERS = {"HOH", "WAT", "DOD", "H2O"}
# Common crystallisation additives and ions that are never the ligand of interest.
ADDITIVES = {
    # ions and small cryo/buffer molecules
    "SO4", "PO4", "GOL", "EDO", "ACT", "DMS", "CL", "NA", "K", "MG",
    "CA", "ZN", "MN", "FE", "NI", "CU", "CO", "CD", "IOD", "BR", "FMT", "TRS",
    "MES", "EPE", "BTB", "BME", "DTT", "CIT", "FLC", "TLA", "MLI", "ACY",
    "NO3", "CO3", "BCT", "SCN", "AZI", "NH4", "MPD", "IPA", "MOH", "EOH",
    "SIN", "PYR", "OXL", "TAR", "LAC", "URE", "IMD", "BEZ",
    # polyethylene glycols and oligo-ethers: cryoprotectants that fill surface
    # grooves and are never the ligand of interest, but are large enough to pass
    # a molecular-weight filter (PE8 is 428 Da)
    "PEG", "PGE", "PG0", "PG4", "PG5", "PG6", "P6G", "P33", "1PE", "2PE",
    "3PE", "PE3", "PE4", "PE5", "PE8", "7PE", "XPE", "12P", "15P", "33O",
    "DIO", "DOX", "ETE", "TOE", "SBT",
}

# AutoDock atom types -> element
AD_TYPES = {
    "C": "C", "A": "C", "N": "N", "NA": "N", "NS": "N", "OA": "O", "OS": "O",
    "S": "S", "SA": "S", "P": "P", "F": "F", "Cl": "Cl", "CL": "Cl", "Br": "Br",
    "BR": "Br", "I": "I", "H": "H", "HD": "H", "HS": "H", "Mg": "Mg", "Zn": "Zn",
    "Fe": "Fe", "Mn": "Mn", "Ca": "Ca", "SI": "Si", "Si": "Si", "B": "B",
    # Meeko opens macrocycles for docking and types the two carbons it breaks
    # the ring between as CG<n>; they are ordinary carbons.
    "CG": "C",
}
# Pseudo-atoms that are not part of the molecule and must be dropped when a
# docked pose is read back: Meeko's macrocycle "glue" (G<n>) and the dummy
# water of hydrated docking (W).
AD_PSEUDO = {"G", "W"}
AD_DIGITS = str.maketrans("", "", "0123456789")


def ad_element(adtype: str) -> str | None:
    """Element for an AutoDock atom type, or None for a pseudo-atom to skip.

    Types may carry an index (``CG0``, ``G1``), so the digits are stripped
    before lookup.
    """
    base = adtype.translate(AD_DIGITS)
    if base.upper() in AD_PSEUDO:
        return None
    if base in AD_TYPES:
        return AD_TYPES[base]
    if base.upper() in AD_TYPES:
        return AD_TYPES[base.upper()]
    el = base.capitalize()
    try:                      # GetAtomicNumber raises for a name it does not know
        if el.isalpha() and Chem.GetPeriodicTable().GetAtomicNumber(el) > 0:
            return el
    except RuntimeError:
        pass
    raise ValueError(f"unknown AutoDock atom type {adtype!r}")


@dataclass
class Residue:
    chain: str
    number: int
    icode: str
    name: str
    atom_names: list[str]
    elements: list[str]
    coords: np.ndarray  # (n_atoms, 3)

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.number, self.icode)

    def atom(self, name: str) -> np.ndarray | None:
        try:
            return self.coords[self.atom_names.index(name)]
        except ValueError:
            return None


@dataclass
class Complex:
    residues: list[Residue]
    ligand: Chem.Mol | None

    def chains(self) -> dict[str, list[Residue]]:
        out: dict[str, list[Residue]] = {}
        for r in self.residues:
            out.setdefault(r.chain, []).append(r)
        return out


def is_amino_acid(name: str) -> bool:
    info = gemmi.find_tabulated_residue(name)
    return bool(info and info.is_amino_acid())


def one_letter(name: str) -> str:
    info = gemmi.find_tabulated_residue(name)
    if info and info.is_amino_acid():
        code = info.one_letter_code.upper()
        return code if code.isalpha() else "X"
    return "X"


def ligand_from_atoms(elements: list[str], coords: np.ndarray, names: list[str] | None = None) -> Chem.Mol:
    """Build a heavy-atom molecule and perceive connectivity from distances.

    Bond orders are not needed: pose comparison works on the heavy-atom graph.
    """
    rw = Chem.RWMol()
    keep = [i for i, e in enumerate(elements) if e.upper() not in ("H", "D")]
    for i in keep:
        el = elements[i].capitalize()
        atom = Chem.Atom(el)
        atom.SetNoImplicit(True)
        if names:
            atom.SetProp("atom_name", names[i])
        rw.AddAtom(atom)
    conf = Chem.Conformer(len(keep))
    for j, i in enumerate(keep):
        conf.SetAtomPosition(j, Point3D(*map(float, coords[i])))
    rw.AddConformer(conf, assignId=True)
    mol = rw.GetMol()
    if mol.GetNumAtoms() > 1:
        rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False)
    return mol


def _read_gemmi(path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(path), merge_chain_parts=True)
    st.remove_hydrogens()
    st.setup_entities()
    return st


def _pick_ligand(candidates: list[tuple[str, gemmi.Residue]], resname: str | None):
    if resname:
        hits = [(c, r) for c, r in candidates if r.name == resname]
        if not hits:
            return None
        return hits[0]
    hits = [(c, r) for c, r in candidates if r.name not in ADDITIVES]
    if not hits:
        return None
    return max(hits, key=lambda cr: len(cr[1]))


def read_complex(path: str | Path, ligand_resname: str | None = None,
                 chains: list[str] | None = None) -> Complex:
    """Read a PDB/mmCIF complex (optionally gzipped).

    The ligand is the residue named ``ligand_resname``; if that is not given the
    largest non-polymer, non-water, non-additive residue is used. Co-folding
    tools rename ligands (e.g. ``LIG``), so leave it empty for their output.
    """
    st = _read_gemmi(Path(path))
    model = st[0]
    residues: list[Residue] = []
    het: list[tuple[str, gemmi.Residue]] = []
    for ch in model:
        for res in ch:
            if res.name in WATERS:
                continue
            # Modified amino acids (MSE, PTR ...) count as protein when they have a CA.
            if (is_amino_acid(res.name) and res.name != ligand_resname
                    and res.find_atom("CA", "*") is not None):
                if chains and ch.name not in chains:
                    continue
                residues.append(Residue(
                    chain=ch.name, number=res.seqid.num, icode=res.seqid.icode.strip(),
                    name=res.name,
                    atom_names=[a.name for a in res],
                    elements=[a.element.name for a in res],
                    coords=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res], dtype=float),
                ))
            else:
                het.append((ch.name, res))
    ligand = None
    pick = _pick_ligand(het, ligand_resname)
    if pick is not None:
        _, res = pick
        ligand = ligand_from_atoms(
            [a.element.name for a in res],
            np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res]),
            [a.name for a in res],
        )
    return Complex(residues=residues, ligand=ligand)


def _open_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return path.read_text()


def read_pdbqt_poses(path: str | Path) -> list[tuple[Chem.Mol, dict[str, float]]]:
    """Parse every MODEL of a Vina/AutoDock PDBQT file.

    Returns (molecule, scores) per pose; scores holds ``vina_affinity`` when a
    ``REMARK VINA RESULT`` line is present.
    """
    poses = []
    elements: list[str] = []
    coords: list[list[float]] = []
    names: list[str] = []
    scores: dict[str, float] = {}

    def flush():
        if elements:
            poses.append((ligand_from_atoms(elements, np.array(coords), names), dict(scores)))

    for line in _open_text(Path(path)).splitlines():
        if line.startswith("MODEL"):
            flush()
            elements, coords, names, scores = [], [], [], {}
        elif line.startswith("REMARK VINA RESULT"):
            scores["vina_affinity"] = float(line.split()[3])
        elif line.startswith(("ATOM", "HETATM")):
            adtype = line[77:79].strip() or line[76:79].strip()
            el = ad_element(adtype)
            if el is None:          # macrocycle glue / dummy water: not an atom
                continue
            elements.append(el)
            names.append(line[12:16].strip())
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    flush()
    return poses


@contextmanager
def quiet_rdkit():
    """Silence RDKit warnings while parsing poses.

    Docking output triggers one harmless warning per pose (GNINA marks its SDF
    as 2D while writing 3D coordinates), which would bury the real output on a
    full target set. Errors are still shown.
    """
    RDLogger.DisableLog("rdApp.warning")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.warning")


def read_sdf_poses(path: str | Path) -> list[tuple[Chem.Mol, dict[str, float]]]:
    """Parse every molecule of an SDF (GNINA, or any engine converted to SDF).

    Numeric SD properties (``CNNscore``, ``minimizedAffinity`` ...) are returned
    as the scores dict.
    """
    path = Path(path)
    poses = []
    with quiet_rdkit():
        if path.suffix == ".gz":
            supplier = Chem.ForwardSDMolSupplier(gzip.open(path), removeHs=False, sanitize=False)
        else:
            supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        for mol in supplier:
            if mol is None:
                continue
            scores = {}
            for key, val in mol.GetPropsAsDict().items():
                try:
                    scores[key] = float(val)
                except (TypeError, ValueError):
                    pass
            pos = mol.GetConformer().GetPositions()
            elements = [a.GetSymbol() for a in mol.GetAtoms()]
            poses.append((ligand_from_atoms(elements, pos), scores))
    return poses


def read_ligand_pose(path: str | Path, index: int = 0) -> Chem.Mol:
    """Read one ligand pose from SDF, PDBQT, PDB or mmCIF."""
    path = Path(path)
    suffixes = "".join(path.suffixes[-2:]).lower()
    if ".sdf" in suffixes or ".mol" in suffixes:
        return read_sdf_poses(path)[index][0]
    if ".pdbqt" in suffixes:
        return read_pdbqt_poses(path)[index][0]
    cx = read_complex(path)
    if cx.ligand is None:
        raise ValueError(f"no ligand found in {path}")
    return cx.ligand


def load_pose(structure: str | Path, ligand: str | Path | None = None, pose_index: int = 0,
              ligand_resname: str | None = None) -> Complex:
    """Load a pose: either a full complex, or a receptor plus a separate ligand file."""
    cx = read_complex(structure, ligand_resname=ligand_resname)
    if ligand:
        cx.ligand = read_ligand_pose(ligand, pose_index)
    return cx


def ligand_coords(mol: Chem.Mol) -> np.ndarray:
    return mol.GetConformer().GetPositions()


def set_ligand_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*map(float, xyz)))
    return out


def write_complex_pdb(cx: Complex, path: str | Path, ligand_resname: str = "LIG",
                      ligand_chain: str = "L") -> None:
    """Write a complex as PDB (protein ATOM records + ligand HETATM records)."""
    lines = []
    serial = 1
    for r in cx.residues:
        for name, el, xyz in zip(r.atom_names, r.elements, r.coords):
            pad = f" {name:<3}" if len(name) < 4 else name
            lines.append(
                f"ATOM  {serial:5d} {pad:<4} {r.name:>3} {r.chain[:1]}{r.number:4d}{r.icode[:1] or ' '}   "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {el:>2}"
            )
            serial += 1
    lines.append("TER")
    if cx.ligand is not None:
        pos = ligand_coords(cx.ligand)
        counts: dict[str, int] = {}
        for atom, xyz in zip(cx.ligand.GetAtoms(), pos):
            el = atom.GetSymbol()
            if atom.HasProp("atom_name"):
                name = atom.GetProp("atom_name")
            else:
                counts[el] = counts.get(el, 0) + 1
                name = f"{el}{counts[el]}"
            pad = f" {name:<3}" if len(name) < 4 and len(el) == 1 else f"{name:<4}"
            lines.append(
                f"HETATM{serial:5d} {pad[:4]} {ligand_resname:>3} {ligand_chain}   1    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {el.upper():>2}"
            )
            serial += 1
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")
