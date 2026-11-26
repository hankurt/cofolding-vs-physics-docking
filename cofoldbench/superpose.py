"""Map a model onto the reference and superpose it on the binding pocket.

Co-folding models number residues from 1 along the input sequence and sit in
an arbitrary frame, and cross-docking receptors can be homology models with a
shifted numbering. So every model is (1) mapped to the reference numbering by
sequence alignment and (2) superposed on the reference pocket C-alpha atoms
before any ligand comparison. Ligand RMSD is therefore pocket-aligned, the
standard for co-folding benchmarks (PoseBusters, PLINDER).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import gemmi
import numpy as np

from .structure import Complex, Residue, ligand_coords


@dataclass
class Superposition:
    rotation: np.ndarray       # (3, 3), applied as coords @ rotation.T + translation
    translation: np.ndarray    # (3,)
    pocket_ca_rmsd: float      # after fitting
    n_pocket_ca: int
    mapping: dict              # model residue key -> reference residue key

    def apply(self, coords: np.ndarray) -> np.ndarray:
        return coords @ self.rotation.T + self.translation


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation that best map ``mobile`` onto ``target``."""
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    h = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, tc - mc @ rot.T


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def align_residues(ref: list[Residue], mdl: list[Residue]) -> tuple[list[tuple[Residue, Residue]], int]:
    """Global sequence alignment of two chains; returns aligned identical pairs."""
    result = gemmi.align_string_sequences([r.name for r in ref], [r.name for r in mdl], [])
    pairs = []
    i = j = 0
    for count, op in re.findall(r"(\d+)([MID])", result.cigar_str()):
        n = int(count)
        if op == "M":
            for k in range(n):
                if ref[i + k].name == mdl[j + k].name:
                    pairs.append((ref[i + k], mdl[j + k]))
            i += n
            j += n
        elif op == "I":   # residues only in the reference
            i += n
        else:             # residues only in the model
            j += n
    return pairs, result.match_count


def map_to_reference(ref: Complex, mdl: Complex, ref_chains: list[str] | None = None
                     ) -> list[tuple[Residue, Residue]]:
    """Pair reference and model residues, chain by chain, by best sequence match."""
    ref_by_chain = ref.chains()
    mdl_by_chain = mdl.chains()
    pairs: list[tuple[Residue, Residue]] = []
    used: set[str] = set()
    for rc in ref_chains or list(ref_by_chain):
        best, best_score, best_chain = [], -1, None
        for mc, mres in mdl_by_chain.items():
            if mc in used:
                continue
            p, score = align_residues(ref_by_chain[rc], mres)
            if score > best_score:
                best, best_score, best_chain = p, score, mc
        if best_chain is not None:
            used.add(best_chain)
            pairs.extend(best)
    return pairs


def pocket_residues(cx: Complex, cutoff: float = 10.0) -> list[Residue]:
    """Residues with any heavy atom within ``cutoff`` A of the ligand."""
    lig = ligand_coords(cx.ligand)
    out = []
    for r in cx.residues:
        d = np.linalg.norm(r.coords[:, None, :] - lig[None, :, :], axis=-1)
        if d.min() <= cutoff:
            out.append(r)
    return out


def superpose_on_pocket(ref: Complex, mdl: Complex, pocket_cutoff: float = 10.0,
                        ref_chains: list[str] | None = None) -> Superposition:
    """Fit the model's pocket C-alpha atoms onto the reference pocket."""
    pairs = map_to_reference(ref, mdl, ref_chains)
    pocket = {r.key for r in pocket_residues(ref, pocket_cutoff)}
    ref_ca, mdl_ca = [], []
    for r, m in pairs:
        if r.key in pocket:
            a, b = r.atom("CA"), m.atom("CA")
            if a is not None and b is not None:
                ref_ca.append(a)
                mdl_ca.append(b)
    if len(ref_ca) < 3:
        raise ValueError(f"only {len(ref_ca)} pocket C-alpha atoms could be matched")
    ref_ca, mdl_ca = np.array(ref_ca), np.array(mdl_ca)
    rot, trans = kabsch(mdl_ca, ref_ca)
    fitted = mdl_ca @ rot.T + trans
    return Superposition(
        rotation=rot, translation=trans,
        pocket_ca_rmsd=rmsd(fitted, ref_ca), n_pocket_ca=len(ref_ca),
        mapping={m.key: r.key for r, m in pairs},
    )


def transform_complex(mdl: Complex, sp: Superposition, renumber: bool = True) -> Complex:
    """Return the model in the reference frame (and reference numbering).

    With ``renumber`` on, aligned residues take the reference chain and number;
    residues with no reference counterpart move to chain ``Z`` so they can
    never collide with a reference residue number.
    """
    from .structure import set_ligand_coords

    residues = []
    for r in mdl.residues:
        if renumber:
            chain, number, icode = sp.mapping.get(r.key, ("Z", r.number, r.icode))
        else:
            chain, number, icode = r.key
        residues.append(Residue(chain, number, icode, r.name, r.atom_names, r.elements, sp.apply(r.coords)))
    ligand = None
    if mdl.ligand is not None:
        ligand = set_ligand_coords(mdl.ligand, sp.apply(ligand_coords(mdl.ligand)))
    return Complex(residues=residues, ligand=ligand)
