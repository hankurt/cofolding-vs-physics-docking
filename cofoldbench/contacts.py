"""Residue-level interaction fingerprints and interaction recovery.

A pose can sit within 2 A RMSD and still miss the key hydrogen bond, or sit
at 3 A and make all the right contacts. Interaction recovery asks: of the
interactions seen in the crystal structure, how many does the pose reproduce?

This module has a fast geometric fingerprint with no external dependencies,
so the benchmark runs anywhere. For full PLIP profiles, run the aligned
complexes that ``cofoldbench evaluate`` writes through the
plip-interaction-fingerprints pipeline (they are already in reference
numbering).
"""

from __future__ import annotations

import numpy as np

from .structure import Complex, ligand_coords

HYDROPHOBIC_CUTOFF = 4.0   # carbon/halogen - carbon, as PLIP's hydrophobic contact
POLAR_CUTOFF = 3.5         # N/O - N/O donor-acceptor distance (no angle check)
IONIC_CUTOFF = 4.0         # ligand N/O - charged side-chain group
CONTACT_CUTOFF = 4.5       # any heavy atom pair

CHARGED_ATOMS = {
    "ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "HIS": {"ND1", "NE2"},
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
}
APOLAR_LIGAND = {"C", "Cl", "Br", "I", "F", "S"}
POLAR = {"N", "O"}

TYPES = ("contact", "hydrophobic", "polar", "ionic")


def fingerprint(cx: Complex) -> set[tuple[str, int, str, str]]:
    """Set of (chain, residue number, residue name, interaction type)."""
    lig = ligand_coords(cx.ligand)
    lig_el = [a.GetSymbol() for a in cx.ligand.GetAtoms()]
    lig_apolar = np.array([e in APOLAR_LIGAND for e in lig_el])
    lig_polar = np.array([e in POLAR for e in lig_el])
    center = lig.mean(axis=0)
    radius = np.linalg.norm(lig - center, axis=1).max() + CONTACT_CUTOFF + 5.0

    fp = set()
    for r in cx.residues:
        if np.linalg.norm(r.coords - center, axis=1).min() > radius:
            continue
        d = np.linalg.norm(r.coords[:, None, :] - lig[None, :, :], axis=-1)
        if d.min() > CONTACT_CUTOFF:
            continue
        tag = (r.chain, r.number, r.name)
        fp.add(tag + ("contact",))
        res_c = np.array([e == "C" for e in r.elements])
        res_p = np.array([e in POLAR for e in r.elements])
        if (d[np.ix_(res_c, lig_apolar)] <= HYDROPHOBIC_CUTOFF).any():
            fp.add(tag + ("hydrophobic",))
        if (d[np.ix_(res_p, lig_polar)] <= POLAR_CUTOFF).any():
            fp.add(tag + ("polar",))
        charged = CHARGED_ATOMS.get(r.name)
        if charged:
            res_q = np.array([n in charged for n in r.atom_names])
            if (d[np.ix_(res_q, lig_polar)] <= IONIC_CUTOFF).any():
                fp.add(tag + ("ionic",))
    return fp


def recovery(reference: set, pose: set, kind: str | None = None) -> float:
    """Fraction of reference interactions (optionally of one type) found in the pose.

    Residues are compared by (chain, number, type); the pose must already be
    in reference numbering.
    """
    ref = {(c, n, t) for c, n, _, t in reference if kind is None or t == kind}
    got = {(c, n, t) for c, n, _, t in pose if kind is None or t == kind}
    if not ref:
        return float("nan")
    return len(ref & got) / len(ref)


def tanimoto(a: set, b: set) -> float:
    a = {(c, n, t) for c, n, _, t in a}
    b = {(c, n, t) for c, n, _, t in b}
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)
