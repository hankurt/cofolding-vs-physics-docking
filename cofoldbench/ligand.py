"""Ligand pose metrics: symmetry-aware RMSD and simple physical-validity checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from .structure import Complex, ligand_coords


def graph_mol(mol: Chem.Mol) -> Chem.Mol:
    """Heavy-atom graph with every bond single and no charges or aromaticity.

    Poses from different engines disagree on bond orders, protonation and atom
    names, but the heavy-atom graph is invariant, so matching on it makes the
    RMSD comparable across all of them.
    """
    rw = Chem.RWMol()
    idx = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        a = Chem.Atom(atom.GetAtomicNum())
        a.SetNoImplicit(True)
        idx[atom.GetIdx()] = rw.AddAtom(a)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in idx and j in idx:
            rw.AddBond(idx[i], idx[j], Chem.BondType.SINGLE)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    pos = mol.GetConformer().GetPositions()
    conf = Chem.Conformer(out.GetNumAtoms())
    for old, new in idx.items():
        conf.SetAtomPosition(new, pos[old].tolist())
    out.AddConformer(conf, assignId=True)
    return out


def symmetric_rmsd(pose: Chem.Mol, reference: Chem.Mol, max_matches: int = 10000) -> float:
    """RMSD without refitting, minimised over graph automorphisms.

    Returns NaN when the two heavy-atom graphs differ, which for predicted
    structures usually means the ligand came out with broken or extra bonds.
    """
    p, r = graph_mol(pose), graph_mol(reference)
    # Equal atom and bond counts make the substructure match an isomorphism;
    # otherwise a pose with a broken bond would match as a subgraph.
    if p.GetNumAtoms() != r.GetNumAtoms() or p.GetNumBonds() != r.GetNumBonds():
        return float("nan")
    matches = r.GetSubstructMatches(p, uniquify=False, useChirality=False, maxMatches=max_matches)
    if not matches:
        return float("nan")
    pp, rp = p.GetConformer().GetPositions(), r.GetConformer().GetPositions()
    best = min(((pp - rp[list(m)]) ** 2).sum(axis=1).mean() for m in matches)
    return float(np.sqrt(best))


def centroid_distance(pose: Chem.Mol, reference: Chem.Mol) -> float:
    return float(np.linalg.norm(ligand_coords(pose).mean(axis=0) - ligand_coords(reference).mean(axis=0)))


@dataclass
class Validity:
    min_distance: float        # closest protein-ligand heavy-atom distance (A)
    n_clashes: int             # protein-ligand heavy-atom pairs closer than clash_cutoff
    graph_matches_reference: bool

    @property
    def ok(self) -> bool:
        return self.graph_matches_reference and self.n_clashes == 0


def validity(cx: Complex, reference: Chem.Mol, clash_cutoff: float = 2.2) -> Validity:
    """Cheap PoseBusters-style sanity checks on a pose."""
    lig = ligand_coords(cx.ligand)
    prot = np.concatenate([r.coords for r in cx.residues])
    # Only look at protein atoms near the ligand to keep this fast.
    near = prot[np.abs(prot - lig.mean(axis=0)).max(axis=1) < 20.0]
    if len(near):
        d = np.linalg.norm(near[:, None, :] - lig[None, :, :], axis=-1)
        min_d, n_clash = float(d.min()), int((d < clash_cutoff).sum())
    else:
        min_d, n_clash = float("inf"), 0
    graph_ok = not np.isnan(symmetric_rmsd(cx.ligand, reference))
    return Validity(min_d, n_clash, graph_ok)
