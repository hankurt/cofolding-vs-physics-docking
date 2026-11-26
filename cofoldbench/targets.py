"""Benchmark target table.

``targets.csv`` columns:

target_id        short name used everywhere (file names, manifest, results)
structure        reference holo structure, relative to the CSV file
chain            receptor chain(s) in the reference, e.g. ``A`` or ``A B``
ligand_resname   residue name of the ligand in the reference
smiles           ligand SMILES given to the co-folding models
pocket_residues  optional, space-separated residue numbers (reference numbering)
                 used for pocket-conditioned co-folding and to box the docking
sequence         optional, full construct sequence; defaults to the modelled
                 residues of the reference chain
release_date     optional, PDB release date (YYYY-MM-DD); splits the report
                 into targets before / after the models' training cut-off
resolution, note optional, informational
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

from .structure import Complex, one_letter, read_complex


@dataclass
class Target:
    target_id: str
    structure: Path
    chains: list[str]
    ligand_resname: str
    smiles: str
    pocket_residues: list[int]
    sequence: str | None = None
    release_date: str | None = None
    note: str | None = None

    @cached_property
    def reference(self) -> Complex:
        cx = read_complex(self.structure, ligand_resname=self.ligand_resname, chains=self.chains)
        if cx.ligand is None:
            raise ValueError(f"{self.target_id}: ligand {self.ligand_resname} not found in {self.structure}")
        return cx

    def chain_sequences(self) -> dict[str, str]:
        """Sequence per receptor chain from the modelled reference residues."""
        by_chain = self.reference.chains()
        return {c: "".join(one_letter(r.name) for r in by_chain[c]) for c in self.chains}

    def gaps(self) -> dict[str, list[tuple[int, int]]]:
        """Breaks in residue numbering (missing loops) per chain."""
        out = {}
        for c, res in self.reference.chains().items():
            nums = [r.number for r in res]
            out[c] = [(a, b) for a, b in zip(nums, nums[1:]) if b - a > 1]
        return out


def _split(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return str(value).replace(",", " ").split()


def read_csv_skip_comments(path: str | Path) -> pd.DataFrame:
    """CSV whose comment lines start with ``#``.

    Not ``pd.read_csv(comment="#")``: that also cuts every line at a ``#``,
    which is the triple bond in SMILES (``C#N``, ``C#C``).
    """
    lines = Path(path).read_text().splitlines()
    body = "\n".join(ln for ln in lines if not ln.lstrip().startswith("#"))
    return pd.read_csv(io.StringIO(body), dtype=str, keep_default_na=False)


def _clean(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    value = str(value).strip()
    return value or None


def load_targets(path: str | Path) -> list[Target]:
    path = Path(path)
    df = read_csv_skip_comments(path)
    targets = []
    for row in df.to_dict("records"):
        seq = row.get("sequence")
        targets.append(Target(
            target_id=row["target_id"],
            structure=(path.parent / row["structure"]).resolve(),
            chains=_split(row.get("chain")) or ["A"],
            ligand_resname=row["ligand_resname"],
            smiles=row["smiles"],
            pocket_residues=[int(x) for x in _split(row.get("pocket_residues"))],
            sequence=seq if isinstance(seq, str) and seq.strip() else None,
            release_date=_clean(row.get("release_date")),
            note=_clean(row.get("note")),
        ))
    return targets
