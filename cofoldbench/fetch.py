"""Build benchmark targets from PDB IDs: download, clean, validate, add to targets.csv.

For every candidate (PDB ID + ligand residue name, chain optional):

1. download the entry mmCIF and the ligand's CCD definition from RCSB;
2. keep the receptor chain(s) and one copy of the ligand (no waters, ions or
   other het groups; first alternate conformation only);
3. take the ligand SMILES from the CCD (the SMILES the co-folding models get);
4. check that the SMILES and the crystal ligand have the **same heavy-atom
   graph**. Evaluation compares every pose to the crystal ligand on that
   graph, so a ligand with unmodelled atoms would make every pose fail;
5. derive the pocket residues (<= 4.5 A from the ligand) and the full
   construct sequence (SEQRES), record release date and resolution.

The release date decides whether a target is *after* a model's training
cut-off, the test that separates generalisation from memorisation.
"""

from __future__ import annotations

import gzip
import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from rdkit import Chem

from .ligand import graph_mol
from .structure import ADDITIVES, WATERS, is_amino_acid, ligand_from_atoms

ENTRY_URL = "https://files.rcsb.org/download/{pdb}.cif.gz"
CCD_URL = "https://files.rcsb.org/ligands/download/{ccd}.cif"
USER_AGENT = "cofoldbench (https://github.com/hankurt/cofolding-vs-physics-docking)"

POCKET_CUTOFF = 4.5
# A protein-ligand heavy-atom distance below this means a covalent bond, not a contact.
COVALENT_CUTOFF = 2.1
# Het groups that a protein needs to bind its ligand properly. Co-folding inputs
# here contain only protein + ligand, so a target with one of these next to the
# ligand is flagged: the comparison would be unfair to the co-folding models.
COFACTORS = {
    "HEM", "HEC", "HEA", "HEB", "NAD", "NAI", "NAP", "NDP", "NADH", "FAD", "FMN",
    "SAM", "SAH", "ADP", "ATP", "ANP", "ACP", "AMP", "GDP", "GTP", "GNP", "GCP",
    "GSP", "COA", "ACO", "PLP", "TPP", "BTN", "MTX", "SF4", "FES", "CLA", "BCL",
    "F3S", "SF3", "FS4", "PQQ", "MGD", "COB", "B12", "CNC", "H4B", "THF", "MQ7",
    "UQ1", "PEE", "LMN", "TDP", "FDA", "FAS", "NDE", "NAJ", "NAX", "P5F",
    "ZN", "MG", "MN", "FE", "FE2", "CU", "CO", "NI", "CA",
}

TARGET_COLUMNS = ["target_id", "structure", "chain", "ligand_resname", "smiles",
                  "pocket_residues", "sequence", "release_date", "resolution", "note"]


def download(url: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        data = fh.read()
    return gzip.decompress(data) if url.endswith(".gz") else data


def cached(url: str, cache_dir: Path | None) -> bytes:
    if cache_dir is None:
        return download(url)
    path = Path(cache_dir) / url.rsplit("/", 1)[-1].removesuffix(".gz")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(download(url))
    return path.read_bytes()


def ccd_smiles(ccd_cif: str) -> str:
    """Isomeric SMILES from a CCD definition (CACTVS, then OpenEye, then any, then InChI)."""
    block = gemmi.cif.read_string(ccd_cif).sole_block()
    table = block.find("_pdbx_chem_comp_descriptor.", ["type", "program", "descriptor"])
    found, inchis = {}, []
    for row in table:
        kind, program, desc = (gemmi.cif.as_string(row[i]) for i in range(3))
        if kind.upper().startswith("SMILES"):
            found[(kind.upper(), program.upper().split()[0])] = desc
        elif kind.upper() == "INCHI":
            inchis.append(desc)
    for key in [("SMILES_CANONICAL", "CACTVS"), ("SMILES_CANONICAL", "OPENEYE"),
                ("SMILES", "CACTVS"), ("SMILES", "OPENEYE")]:
        if key in found and Chem.MolFromSmiles(found[key]) is not None:
            return Chem.MolToSmiles(Chem.MolFromSmiles(found[key]))
    for desc in found.values():
        if Chem.MolFromSmiles(desc) is not None:
            return Chem.MolToSmiles(Chem.MolFromSmiles(desc))
    # Brand-new CCD entries sometimes carry only descriptors RDKit rejects
    # (hypervalent or radical notation); the InChI is usually still readable.
    for inchi in inchis:
        mol = Chem.MolFromInchi(inchi, sanitize=True, removeHs=True)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    raise ValueError("no parsable SMILES or InChI in the CCD entry")


def same_graph(crystal: Chem.Mol, smiles: str) -> tuple[bool, str]:
    """Does the crystal ligand have the heavy-atom graph of the SMILES?"""
    ref = Chem.MolFromSmiles(smiles)
    if ref is None:
        return False, f"SMILES does not parse: {smiles}"
    a, b = graph_mol(crystal), Chem.RemoveHs(ref)
    na, nb = a.GetNumAtoms(), b.GetNumHeavyAtoms()
    if na != nb:
        return False, f"crystal ligand has {na} heavy atoms, SMILES has {nb} (unmodelled atoms?)"
    # Compare as plain graphs: element + connectivity, all bonds single.
    rw = Chem.RWMol()
    for atom in b.GetAtoms():
        x = Chem.Atom(atom.GetAtomicNum())
        x.SetNoImplicit(True)
        rw.AddAtom(x)
    for bond in b.GetBonds():
        rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    g = rw.GetMol()
    g.UpdatePropertyCache(strict=False)
    if a.GetNumBonds() != g.GetNumBonds() or not a.HasSubstructMatch(g):
        return False, "crystal connectivity differs from the SMILES (bad geometry or covalent adduct?)"
    return True, "ok"


@dataclass
class Prepared:
    target_id: str
    chains: list[str]
    ligand_resname: str
    smiles: str
    pocket_residues: list[int]
    sequence: str
    release_date: str
    resolution: float
    structure: gemmi.Structure
    n_residues: int
    warnings: list[str] = field(default_factory=list)
    structure_path: Path | None = None


def _release_date(doc: gemmi.cif.Document) -> str:
    block = doc.sole_block()
    dates = [gemmi.cif.as_string(v) for v in block.find_values("_pdbx_audit_revision_history.revision_date")]
    dates = [d for d in dates if d and d not in ("?", ".")]
    if dates:
        return min(dates)
    for tag in ("_pdbx_database_status.recvd_initial_deposition_date",):
        v = block.find_value(tag)
        if v:
            return gemmi.cif.as_string(v)
    return ""


def _heavy_coords(res: gemmi.Residue) -> np.ndarray:
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res if not a.is_hydrogen()])


def prepare(st: gemmi.Structure, target_id: str, ligand_resname: str, chains: list[str] | None = None,
            smiles: str | None = None, release_date: str = "") -> Prepared:
    """Select the receptor chain(s) and one ligand copy from a parsed entry."""
    st = st.clone()
    st.setup_entities()
    st.remove_hydrogens()
    st.remove_alternative_conformations()
    model = st[0]
    warnings: list[str] = []

    # Every copy of the ligand; prefer the most complete, then the one with most contacts.
    copies = [(ch.name, res) for ch in model for res in ch if res.name == ligand_resname]
    if not copies:
        raise ValueError(f"ligand {ligand_resname} not found")

    def protein_chain_contacts(lig: gemmi.Residue) -> dict[str, set[int]]:
        lig_xyz = _heavy_coords(lig)
        out: dict[str, set[int]] = {}
        for ch in model:
            for res in ch:
                if not is_amino_acid(res.name) or res.find_atom("CA", "*") is None:
                    continue
                xyz = _heavy_coords(res)
                if len(xyz) and np.min(np.linalg.norm(xyz[:, None] - lig_xyz[None], axis=-1)) <= POCKET_CUTOFF:
                    out.setdefault(ch.name, set()).add(res.seqid.num)
        return out

    scored = []
    for cname, res in copies:
        contacts = protein_chain_contacts(res)
        if chains:
            n_contact = sum(len(contacts.get(c, ())) for c in chains)
        else:
            n_contact = sum(len(v) for v in contacts.values())
        scored.append((len(res), n_contact, cname, res, contacts))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    _, n_contact, _, lig_res, contacts = scored[0]
    if n_contact == 0 and chains:
        # The requested chain does not line the pocket. A chain hint often comes
        # from a search listing that only knows the entity's chain ids, not which
        # one binds the ligand, so fall back to the chains that actually do.
        rescored = []
        for cname, res in copies:
            c = protein_chain_contacts(res)
            rescored.append((len(res), sum(len(v) for v in c.values()), cname, res, c))
        rescored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        _, n_contact, _, lig_res, contacts = rescored[0]
        if n_contact:
            warnings.append(f"chain {' '.join(chains)} does not line the pocket; "
                            f"using chain {' '.join(sorted(contacts))} instead")
            chains = None
    if n_contact == 0:
        raise ValueError(f"no copy of {ligand_resname} touches any protein chain")

    if not chains:
        # Every protein chain lining the pocket with >= 3 residues, largest first.
        chains = [c for c, r in sorted(contacts.items(), key=lambda cr: -len(cr[1])) if len(r) >= 3]
        if not chains:
            chains = [max(contacts, key=lambda c: len(contacts[c]))]
    pocket = sorted(contacts.get(chains[0], ()))

    # Other het groups next to the ligand (cofactors, metals) that would be missing in co-folding.
    lig_xyz = _heavy_coords(lig_res)
    near = set()
    for ch in model:
        for res in ch:
            if res is lig_res or res.name in WATERS or is_amino_acid(res.name):
                continue
            xyz = _heavy_coords(res)
            if len(xyz) and np.min(np.linalg.norm(xyz[:, None] - lig_xyz[None], axis=-1)) <= 6.0:
                near.add(res.name)
    flagged = sorted(n for n in near if n in COFACTORS)
    if flagged:
        warnings.append(f"cofactor/metal next to the ligand: {' '.join(flagged)} "
                        "(absent from the co-folding inputs)")
    others = sorted(n for n in near - set(flagged) if n not in ADDITIVES and n != ligand_resname)
    if others:
        warnings.append(f"other het groups within 6 A: {' '.join(others)}")

    # A binding site at an oligomeric interface is only half a site when the
    # partner chain is a crystallographic symmetry mate rather than a chain of
    # the deposited model (STING, for one). Every method then fails on a target
    # that is broken by the setup, not hard.
    if st.cell.is_crystal() and st.spacegroup_hm:
        try:
            st.setup_cell_images()
            ns = gemmi.NeighborSearch(model, st.cell, 5.0).populate()
            own = sym = 0
            for atom in lig_res:
                if atom.is_hydrogen():
                    continue
                for mark in ns.find_atoms(atom.pos, "\0", radius=POCKET_CUTOFF):
                    cra = mark.to_cra(model)
                    if cra.residue.name == lig_res.name or not is_amino_acid(cra.residue.name):
                        continue
                    # image_idx reports symmetry operations but not plain lattice
                    # translations, so compare against the un-imaged distance.
                    if atom.pos.dist(cra.atom.pos) <= POCKET_CUTOFF:
                        own += 1
                    else:
                        sym += 1
            total = own + sym
            if sym and sym >= 0.25 * max(own, 1):
                warnings.append(
                    f"SITE SPANS A SYMMETRY MATE: {sym} of {total} protein contacts ({sym / total:.0%}) "
                    "come from a crystallographic image, not a deposited chain, so the pocket is "
                    "incomplete in this model and cannot be fixed by adding a chain. Drop the target, "
                    "or rebuild it from the biological assembly. Around half the contacts means a true "
                    "interface site; 20-30% can be ordinary packing against an open pocket mouth, so "
                    "check the biology before keeping it")
        except Exception as exc:                      # never fail a target over this check
            warnings.append(f"symmetry contact check skipped: {exc}")

    # Covalent ligands (a protein heavy atom bonded to the ligand) cannot be
    # reproduced by non-covalent docking or by co-folding, so every method
    # would fail for a reason that has nothing to do with pose prediction.
    covalent = []
    for ch in model:
        for res in ch:
            if not is_amino_acid(res.name) or res is lig_res:
                continue
            for atom in res:
                if atom.is_hydrogen() or atom.name in ("N", "C", "O", "CA"):
                    continue
                xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                if np.min(np.linalg.norm(lig_xyz - xyz, axis=-1)) <= COVALENT_CUTOFF:
                    covalent.append(f"{res.name}{res.seqid.num}.{atom.name}")
    if covalent:
        warnings.append(f"LIGAND LOOKS COVALENT: bonded to {' '.join(sorted(set(covalent)))} "
                        "- non-covalent docking and co-folding cannot reproduce this; "
                        "consider dropping this target")

    # New structure: the receptor chains (amino acids only) + the chosen ligand copy.
    out = gemmi.Structure()
    out.cell = st.cell
    out.spacegroup_hm = st.spacegroup_hm
    out_model = gemmi.Model("1")
    n_res = 0
    for cname in chains:
        src = model.find_chain(cname)
        if src is None:
            raise ValueError(f"chain {cname} not found")
        new = gemmi.Chain(cname)
        for res in src:
            if is_amino_acid(res.name) and res.find_atom("CA", "*") is not None and res.name != ligand_resname:
                new.add_residue(res)
                n_res += 1
        out_model.add_chain(new)
    lig_chain = gemmi.Chain(next(c for c in "LMNOPQRSTUVWXYZ" if c not in chains))
    lig_copy = lig_res.clone()
    lig_copy.het_flag = "H"
    lig_chain.add_residue(lig_copy)
    out_model.add_chain(lig_chain)
    out.add_model(out_model)
    out.setup_entities()

    # Full construct sequence (SEQRES) of the first receptor chain.
    sequence = ""
    ent = st.get_entity_of(model.find_chain(chains[0]).get_polymer())
    if ent is not None and ent.full_sequence:
        sequence = gemmi.one_letter_code([gemmi.Entity.first_mon(m) for m in ent.full_sequence]).upper()
        if "X" in sequence.strip("X") or not sequence:
            sequence = ""  # unusual residues: fall back to the modelled residues
    if len(chains) > 1:
        sequence = ""  # the sequence column only covers single-chain targets

    crystal = ligand_from_atoms([a.element.name for a in lig_res],
                                np.array([[a.pos.x, a.pos.y, a.pos.z] for a in lig_res]),
                                [a.name for a in lig_res])
    if smiles:
        ok, why = same_graph(crystal, smiles)
        if not ok:
            raise ValueError(f"ligand check failed: {why}")

    total = n_res if not sequence else len(sequence) * (1 if len(chains) == 1 else len(chains))
    if total > 900:
        warnings.append(f"{total} residues: may not fit a 12 GB GPU for co-folding")

    return Prepared(
        target_id=target_id, chains=chains, ligand_resname=ligand_resname, smiles=smiles or "",
        pocket_residues=pocket, sequence=sequence, release_date=release_date,
        resolution=float(st.resolution or np.nan), structure=out, n_residues=n_res, warnings=warnings,
    )


def write_structure(st: gemmi.Structure, path: Path) -> Path:
    """PDB when every residue name fits 3 characters, otherwise mmCIF."""
    long_names = any(len(res.name) > 3 for ch in st[0] for res in ch)
    path = path.with_suffix(".cif" if long_names else ".pdb")
    path.parent.mkdir(parents=True, exist_ok=True)
    if long_names:
        st.make_mmcif_document().write_file(str(path))
    else:
        st.write_pdb(str(path), gemmi.PdbWriteOptions(minimal=True))
    return path


def fetch_one(pdb_id: str, ligand_resname: str, chains: list[str] | None, structures_dir: Path,
              cache_dir: Path | None = None) -> Prepared:
    pdb_id = pdb_id.upper()
    text = cached(ENTRY_URL.format(pdb=pdb_id), cache_dir).decode()
    doc = gemmi.cif.read_string(text)
    st = gemmi.make_structure_from_block(doc.sole_block())
    st.merge_chain_parts()
    smiles = ccd_smiles(cached(CCD_URL.format(ccd=ligand_resname), cache_dir).decode())
    prep = prepare(st, pdb_id, ligand_resname, chains, smiles=smiles, release_date=_release_date(doc))
    path = write_structure(prep.structure, structures_dir / f"{pdb_id.lower()}_{''.join(prep.chains)}")
    prep.structure_path = path
    return prep


def target_row(prep: Prepared, structure_rel: str, note: str = "") -> dict:
    return {
        "target_id": prep.target_id,
        "structure": structure_rel,
        "chain": " ".join(prep.chains),
        "ligand_resname": prep.ligand_resname,
        "smiles": prep.smiles,
        "pocket_residues": " ".join(map(str, prep.pocket_residues)),
        "sequence": prep.sequence,
        "release_date": prep.release_date,
        "resolution": f"{prep.resolution:.2f}" if np.isfinite(prep.resolution) else "",
        "note": note,
    }


def read_targets_table(path: Path) -> tuple[list[str], pd.DataFrame]:
    """Targets CSV with its leading comment lines kept."""
    if not path.exists():
        return [], pd.DataFrame(columns=TARGET_COLUMNS)
    lines = path.read_text().splitlines()
    comments = [ln for ln in lines if ln.startswith("#")]
    body = "\n".join(ln for ln in lines if not ln.startswith("#"))
    df = pd.read_csv(io.StringIO(body), dtype=str, keep_default_na=False)
    for col in TARGET_COLUMNS:
        if col not in df:
            df[col] = ""
    return comments, df


def write_targets_table(path: Path, comments: list[str], df: pd.DataFrame) -> None:
    cols = TARGET_COLUMNS + [c for c in df.columns if c not in TARGET_COLUMNS]
    buf = io.StringIO()
    df[cols].to_csv(buf, index=False)
    path.write_text("\n".join(comments + [buf.getvalue().rstrip("\n")]) + "\n")
