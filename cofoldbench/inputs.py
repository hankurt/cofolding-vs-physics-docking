"""Write co-folding inputs (Boltz-2, Chai-1, AlphaFold3) for every target.

Two variants are written per target:

* ``blind``  - sequence + SMILES only. The honest co-folding setting.
* ``pocket`` - Boltz-2 only: also a pocket constraint on the binding-site
  residues. Docking is always told where the pocket is, so this is the
  like-for-like comparison with HADDOCK3 / GNINA / Vina.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import gemmi

from .targets import Target


def _ligand_chain_id(chains: list[str]) -> str:
    for c in "LMNOPQRSTUVWXYZ":
        if c not in chains:
            return c
    raise ValueError("no free chain id for the ligand")


def sequences_and_index(t: Target) -> tuple[dict[str, str], dict[tuple[str, int], int]]:
    """Sequences to model per chain, and reference (chain, resnum) -> 1-based index.

    Co-folding tools address residues by their position in the input
    sequence, so pocket residues must be translated from reference numbering.
    """
    ref_seqs = t.chain_sequences()
    by_chain = t.reference.chains()
    seqs, index = {}, {}
    for c in t.chains:
        residues = by_chain[c]
        if t.sequence and len(t.chains) == 1:
            seq = t.sequence.strip().upper()
            aln = gemmi.align_string_sequences(list(ref_seqs[c]), list(seq), [])
            i = j = 0
            for count, op in re.findall(r"(\d+)([MID])", aln.cigar_str()):
                n = int(count)
                if op == "M":
                    for k in range(n):
                        if ref_seqs[c][i + k] == seq[j + k]:
                            index[(c, residues[i + k].number)] = j + k + 1
                    i += n
                    j += n
                elif op == "I":
                    i += n
                else:
                    j += n
        else:
            seq = ref_seqs[c]
            for k, r in enumerate(residues):
                index[(c, r.number)] = k + 1
        seqs[c] = seq
    return seqs, index


def pocket_contacts(t: Target, index: dict[tuple[str, int], int]) -> list[tuple[str, int]]:
    contacts = []
    for resnum in t.pocket_residues:
        key = (t.chains[0], resnum)
        if key not in index:
            raise ValueError(f"{t.target_id}: pocket residue {resnum} is not in the modelled sequence")
        contacts.append((t.chains[0], index[key]))
    return contacts


def boltz_yaml(t: Target, pocket: bool = False, affinity: bool = True) -> str:
    seqs, index = sequences_and_index(t)
    lig = _ligand_chain_id(t.chains)
    lines = ["version: 1", "sequences:"]
    for c, seq in seqs.items():
        lines += ["  - protein:", f"      id: {c}", f"      sequence: {seq}"]
    lines += ["  - ligand:", f"      id: {lig}", f"      smiles: '{t.smiles}'"]
    if pocket and t.pocket_residues:
        contacts = ", ".join(f"[{c}, {i}]" for c, i in pocket_contacts(t, index))
        lines += ["constraints:", "  - pocket:", f"      binder: {lig}", f"      contacts: [{contacts}]"]
    if affinity:
        lines += ["properties:", "  - affinity:", f"      binder: {lig}"]
    return "\n".join(lines) + "\n"


def chai_fasta(t: Target) -> str:
    seqs, _ = sequences_and_index(t)
    out = [f">protein|name={t.target_id}_{c}\n{seq}" for c, seq in seqs.items()]
    out.append(f">ligand|name={t.ligand_resname}\n{t.smiles}")
    return "\n".join(out) + "\n"


def af3_json(t: Target, seeds: tuple[int, ...] = (1,)) -> str:
    seqs, _ = sequences_and_index(t)
    entries = [{"protein": {"id": c, "sequence": seq}} for c, seq in seqs.items()]
    entries.append({"ligand": {"id": _ligand_chain_id(t.chains), "smiles": t.smiles}})
    job = {
        "name": t.target_id,
        "modelSeeds": list(seeds),
        "sequences": entries,
        "dialect": "alphafold3",
        "version": 1,
    }
    return json.dumps(job, indent=2) + "\n"


def write_inputs(targets: list[Target], outdir: str | Path) -> list[str]:
    """Write every input file; returns warnings (e.g. gaps in the reference chain)."""
    outdir = Path(outdir)
    warnings = []
    for sub in ("boltz/blind", "boltz/pocket", "chai", "af3"):
        (outdir / sub).mkdir(parents=True, exist_ok=True)
    for t in targets:
        (outdir / "boltz/blind" / f"{t.target_id}.yaml").write_text(boltz_yaml(t))
        if t.pocket_residues:
            (outdir / "boltz/pocket" / f"{t.target_id}.yaml").write_text(boltz_yaml(t, pocket=True))
        (outdir / "chai" / f"{t.target_id}.fasta").write_text(chai_fasta(t))
        (outdir / "af3" / f"{t.target_id}.json").write_text(af3_json(t))
        if not t.sequence:
            for chain, gaps in t.gaps().items():
                if gaps:
                    warnings.append(
                        f"{t.target_id} chain {chain}: numbering gaps {gaps}; the sequence has "
                        "missing loops - set the 'sequence' column to the full construct"
                    )
    return warnings
