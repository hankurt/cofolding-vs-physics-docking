"""Offline checks for target building (fetch / search), docking inputs and the time split."""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import pytest

from cofoldbench import fetch, report, search
from cofoldbench.dockprep import ligand_3d, pocket_box, write_dock_inputs
from cofoldbench.structure import read_complex
from cofoldbench.targets import load_targets

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "data" / "targets.csv"
PDB = ROOT / "data" / "structures" / "6dpt_A.pdb"
H7M = "Cn1cnnc1Sc1ccccc1NS(=O)(=O)c1cccc(Cl)c1O"


@pytest.fixture(scope="module")
def target():
    # By id, not by row: data/targets.csv holds the whole benchmark set.
    return next(t for t in load_targets(TARGETS) if t.target_id == "6DPT")


def test_prepare_selects_chain_pocket_and_ligand(tmp_path):
    st = gemmi.read_structure(str(PDB))
    prep = fetch.prepare(st, "6DPT", "H7M", smiles=H7M, release_date="2018-08-01")
    assert prep.chains == ["A"]
    assert prep.n_residues == 358
    assert {64, 152, 318} <= set(prep.pocket_residues)   # catalytic Ser64, Tyr150 region, Ala318
    path = fetch.write_structure(prep.structure, tmp_path / "6dpt_A")
    cx = read_complex(path, "H7M", ["A"])
    assert len(cx.residues) == 358 and cx.ligand.GetNumAtoms() == 25

    row = fetch.target_row(prep, path.name)
    table = tmp_path / "targets.csv"
    fetch.write_targets_table(table, ["# a comment"], pd.DataFrame([row]))
    (t,) = load_targets(table)
    assert t.smiles == H7M and t.release_date == "2018-08-01"
    assert t.reference.ligand.GetNumAtoms() == 25


def test_ligand_graph_check(target):
    assert fetch.same_graph(target.reference.ligand, H7M) == (True, "ok")
    ok, why = fetch.same_graph(target.reference.ligand, H7M.replace("(Cl)", ""))
    assert not ok and "heavy atoms" in why
    assert not fetch.same_graph(target.reference.ligand, "c1cc(")[0]
    # same atom count, different connectivity (methyl moved from N to C)
    ok, _ = fetch.same_graph(target.reference.ligand, "n1cnn(C)c1Sc1ccccc1NS(=O)(=O)c1cccc(Cl)c1O")
    assert not ok


def test_prepare_rejects_missing_ligand():
    st = gemmi.read_structure(str(PDB))
    with pytest.raises(ValueError, match="not found"):
        fetch.prepare(st, "6DPT", "XYZ")
    with pytest.raises(ValueError, match="ligand check failed"):
        fetch.prepare(st, "6DPT", "H7M", smiles="c1ccccc1")


def test_ccd_smiles():
    cif = """data_BEN
_chem_comp.id BEN
loop_
_pdbx_chem_comp_descriptor.comp_id
_pdbx_chem_comp_descriptor.type
_pdbx_chem_comp_descriptor.program
_pdbx_chem_comp_descriptor.program_version
_pdbx_chem_comp_descriptor.descriptor
BEN InChI InChI 1.03 InChI=1S/C7H8N2/c8-7(9)6-4-2-1-3-5-6/h1-5H,(H3,8,9)
BEN SMILES ACDLabs 10.04 "[N@H]=C(N)c1ccccc1"
BEN SMILES_CANONICAL CACTVS 3.341 "NC(=N)c1ccccc1"
BEN SMILES "OpenEye OEToolkits" 1.5.0 "c1ccc(cc1)C(=N)N"
"""
    assert fetch.ccd_smiles(cif) == "N=C(N)c1ccccc1"


def test_smiles_with_triple_bond_survives_csv(target, tmp_path):
    """'#' starts a comment line but is also the SMILES triple bond (erlotinib's C#C)."""
    erlotinib = "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
    from tests.test_benchmark import single_target_csv
    csv = single_target_csv(tmp_path, target, smiles=erlotinib)
    # A '#' also starts a comment line, so a naive reader truncates the SMILES.
    (loaded,) = load_targets(csv)
    assert loaded.smiles == erlotinib


def test_search_filter():
    def entry(pdb, protein, ligs, length=300, date="2024-05-01T00:00:00Z"):
        return {
            "rcsb_id": pdb,
            "rcsb_accession_info": {"initial_release_date": date},
            "rcsb_entry_info": {"resolution_combined": [1.6]},
            "polymer_entities": [{"entity_poly": {"rcsb_sample_sequence_length": length},
                                  "rcsb_polymer_entity": {"pdbx_description": protein},
                                  "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["B", "A"]}}],
            "nonpolymer_entities": [{"nonpolymer_comp": {"chem_comp": {"id": i, "name": i, "formula_weight": w,
                                                                        "type": "NON-POLYMER"}}}
                                    for i, w in ligs],
        }
    df = search.filter_entries([
        entry("1AAA", "kinase X", [("AAA", 420.0), ("GOL", 92.1)]),   # kept: additive ignored
        entry("1BBB", "kinase X", [("BBB", 380.0)]),                  # same protein again
        entry("1CCC", "reductase", [("CCC", 300.0), ("NAP", 743.0)]), # cofactor
        entry("1DDD", "protease", [("DDD", 120.0)]),                  # fragment, too small
        entry("1EEE", "protease", [("E1", 400.0), ("E2", 410.0)]),    # two ligands: ambiguous
        entry("1FFF", "big enzyme", [("FFF", 400.0)], length=1200),   # too long for 12 GB
        entry("1GGG", "esterase", [("GGG", 350.0)]),
        None,
    ])
    assert list(df["pdb_id"]) == ["1AAA", "1GGG"]
    assert df.iloc[0]["chain"] == "A" and df.iloc[0]["release_date"] == "2024-05-01"
    q = search.search_query("2023-06-01", 2.0, 50)
    assert q["query"]["nodes"][0]["parameters"]["value"] == "2023-06-01"


def test_dock_inputs(target, tmp_path):
    (d,) = write_dock_inputs([target], tmp_path)
    center, size = pocket_box(target)
    xyz = target.reference.ligand.GetConformer().GetPositions()
    assert (np.abs(xyz - center) <= size / 2).all()            # crystal ligand inside the box
    box = dict(line.split(" = ") for line in (d / "box.txt").read_text().splitlines())
    assert float(box["center_x"]) == pytest.approx(center[0], abs=1e-3)
    rec = read_complex(d / "receptor.pdb")
    assert rec.ligand is None and len(rec.residues) == 358
    from rdkit import Chem
    lig = Chem.SDMolSupplier(str(d / "ligand.sdf"), removeHs=False)[0]
    assert lig.GetNumHeavyAtoms() == 25 and lig.GetNumAtoms() > 25   # with hydrogens
    # Starts from a generated conformer, not the crystal pose.
    assert fetch.same_graph(target.reference.ligand, Chem.MolToSmiles(Chem.RemoveHs(lig)))[0]


def test_dock_pdbqt(target, tmp_path):
    pytest.importorskip("meeko")
    pytest.importorskip("openbabel")
    from cofoldbench.dockprep import write_pdbqt
    from cofoldbench.structure import read_pdbqt_poses
    (d,) = write_dock_inputs([target], tmp_path)
    write_pdbqt(d)
    assert " HD" in (d / "receptor.pdbqt").read_text()
    (mol, _), = read_pdbqt_poses(d / "ligand.pdbqt")
    assert mol.GetNumAtoms() == 25


def test_ligand_3d_is_deterministic():
    a = ligand_3d(H7M).GetConformer().GetPositions()
    b = ligand_3d(H7M).GetConformer().GetPositions()
    assert np.allclose(a, b)


def test_time_split_report():
    res = pd.DataFrame([
        dict(target_id=t, method=m, rank=1, ligand_rmsd=r, graph_ok=True, n_clashes=0, status="ok",
             recovery_all=0.5, recovery_polar=0.5, pocket_ca_rmsd=0.3)
        for t, m, r in [("OLD", "boltz2", 0.5), ("NEW", "boltz2", 6.0), ("OLD", "gnina", 1.0), ("NEW", "gnina", 1.5)]
    ])
    split = report.time_split({"OLD": "2015-01-01", "NEW": "2024-02-01", "NODATE": None})
    assert split == {"OLD": "before cutoff", "NEW": "after cutoff"}
    s = report.summarize(res, subsets=split).set_index(["method", "subset"])["top1_success"]
    assert s[("boltz2", "all")] == 0.5
    assert s[("boltz2", "before cutoff")] == 1.0 and s[("boltz2", "after cutoff")] == 0.0
    assert s[("gnina", "after cutoff")] == 1.0


def test_fetch_one_from_cached_mmcif(tmp_path):
    """The mmCIF path of `fetch`, offline: pre-filled download cache."""
    cache = tmp_path / "raw"
    cache.mkdir()
    st = gemmi.read_structure(str(PDB))
    st.setup_entities()
    doc = st.make_mmcif_document()
    doc.sole_block().set_mmcif_category("_pdbx_audit_revision_history.", {
        "ordinal": ["1", "2"], "revision_date": ["2018-08-15", "2020-01-01"]})
    doc.write_file(str(cache / "6DPT.cif"))
    (cache / "H7M.cif").write_text(
        "data_H7M\nloop_\n_pdbx_chem_comp_descriptor.comp_id\n_pdbx_chem_comp_descriptor.type\n"
        "_pdbx_chem_comp_descriptor.program\n_pdbx_chem_comp_descriptor.program_version\n"
        f"_pdbx_chem_comp_descriptor.descriptor\nH7M SMILES_CANONICAL CACTVS 3.385 '{H7M}'\n")
    prep = fetch.fetch_one("6dpt", "H7M", None, tmp_path / "structures", cache_dir=cache)
    assert prep.release_date == "2018-08-15"
    assert prep.structure_path.name == "6dpt_A.pdb"
    assert fetch.same_graph(read_complex(prep.structure_path, "H7M").ligand, prep.smiles)[0]


def test_sdf_poses_are_read_quietly(target, tmp_path, capfd):
    """GNINA tags its SDF 2D while writing 3D coordinates: one warning per pose."""
    from rdkit import Chem
    from cofoldbench.structure import read_sdf_poses
    sdf = tmp_path / "poses.sdf"
    with Chem.SDWriter(str(sdf)) as w:
        for _ in range(3):
            w.write(target.reference.ligand)
    # Mark the connection table as 2D, as GNINA does.
    sdf.write_text(sdf.read_text().replace("2D", "3D").replace("3D", "2D"))
    poses = read_sdf_poses(sdf)
    assert len(poses) == 3 and poses[0][0].GetNumAtoms() == 25
    assert "tagged as 2D" not in capfd.readouterr().err


def test_covalent_ligand_is_flagged(tmp_path):
    """A ligand atom bonded to a side chain (as in nirmatrelvir / Cys145)."""
    st = gemmi.read_structure(str(PDB))
    prep = fetch.prepare(st.clone(), "6DPT", "H7M", smiles=H7M)
    assert not any("COVALENT" in w for w in prep.warnings)

    # Move the ligand so one atom sits at bonding distance from the Ser64 OG.
    model = st[0]
    ser = next(r for ch in model for r in ch if r.seqid.num == 64 and r.name == "SER")
    og = ser.find_atom("OG", "*").pos
    lig = next(r for ch in model for r in ch if r.name == "H7M")
    first = lig[0].pos
    shift = gemmi.Position(og.x - first.x + 1.7, og.y - first.y, og.z - first.z)
    for atom in lig:
        atom.pos = gemmi.Position(atom.pos.x + shift.x, atom.pos.y + shift.y, atom.pos.z + shift.z)
    prep = fetch.prepare(st, "6DPT", "H7M", smiles=H7M)
    assert any("COVALENT" in w and "SER64.OG" in w for w in prep.warnings), prep.warnings


def test_wrong_chain_hint_falls_back(tmp_path):
    """A search listing's chain column need not be the chain that binds the ligand."""
    st = gemmi.read_structure(str(PDB))
    # Rename the receptor chain: the hint "A" no longer matches anything.
    for ch in st[0]:
        if ch.name == "A":
            ch.name = "B"
    prep = fetch.prepare(st, "6DPT", "H7M", chains=["A"], smiles=H7M)
    assert prep.chains == ["B"]
    assert any("does not line the pocket" in w for w in prep.warnings), prep.warnings
    assert {64, 152} <= set(prep.pocket_residues)


def test_missing_poses_count_as_failures():
    """Boltz skips a target it cannot fit in memory; that must not flatter its score."""
    rows = []
    for tid in ("T1", "T2", "T3", "T4"):
        rows.append(dict(target_id=tid, method="vina", rank=1, ligand_rmsd=1.0, graph_ok=True,
                         n_clashes=0, status="ok", recovery_all=0.5, recovery_polar=0.5,
                         pocket_ca_rmsd=0.3))
    # boltz2 only produced two of the four targets, both good.
    for tid in ("T1", "T2"):
        rows.append(dict(target_id=tid, method="boltz2", rank=1, ligand_rmsd=0.5, graph_ok=True,
                         n_clashes=0, status="ok", recovery_all=0.9, recovery_polar=0.9,
                         pocket_ca_rmsd=0.2))
    res = pd.DataFrame(rows)
    ids = ["T1", "T2", "T3", "T4"]
    assert report.missing_poses(res, ids) == {"boltz2": ["T3", "T4"]}

    s = report.summarize(res, targets=ids).set_index(["method", "subset"])
    # Scored over all four targets: 2/4, not 2/2.
    assert s.loc[("boltz2", "all"), "top1_success"] == 0.5
    assert s.loc[("boltz2", "all"), "n_missing"] == 2
    assert s.loc[("boltz2", "all"), "n_targets"] == 4
    assert s.loc[("vina", "all"), "top1_success"] == 1.0
    assert s.loc[("vina", "all"), "n_missing"] == 0

    # Also within a subset of the split.
    split = {"T1": "before cutoff", "T2": "before cutoff", "T3": "after cutoff", "T4": "after cutoff"}
    s = report.summarize(res, subsets=split, targets=ids).set_index(["method", "subset"])
    assert s.loc[("boltz2", "before cutoff"), "top1_success"] == 1.0
    assert s.loc[("boltz2", "after cutoff"), "top1_success"] == 0.0
    assert s.loc[("boltz2", "after cutoff"), "n_missing"] == 2


def test_pdbqt_macrocycle_glue_atoms(tmp_path):
    """Meeko opens macrocycles and adds CG/G pseudo-atoms (1YET geldanamycin).

    The CG atoms are real carbons; the G atoms are glue and must be dropped, or
    the pose's heavy-atom graph will not match the crystal ligand.
    """
    from cofoldbench.structure import read_pdbqt_poses
    lines = ["MODEL 1", "REMARK VINA RESULT:    -9.3      0.000      0.000"]
    # A ring of 6 carbons, opened by Meeko: two CG carbons plus two G glue atoms.
    coords = [(1.5, 0.0, 0.0), (0.75, 1.3, 0.0), (-0.75, 1.3, 0.0),
              (-1.5, 0.0, 0.0), (-0.75, -1.3, 0.0), (0.75, -1.3, 0.0)]
    types = ["C", "C", "C", "C", "CG0", "CG0"]
    for i, ((x, y, z), ad) in enumerate(zip(coords, types), 1):
        lines.append(f"ATOM  {i:5d}  C   LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}  "
                     f"1.00  0.00     0.000 {ad:<2}")
    for i, ad in enumerate(["G0", "G0"], 7):
        lines.append(f"ATOM  {i:5d}  G   LIG A   1    {0.75:8.3f}{-1.3:8.3f}{0.0:8.3f}  "
                     f"1.00  0.00     0.000 {ad:<2}")
    lines.append("ENDMDL")
    path = tmp_path / "macrocycle.pdbqt"
    path.write_text("\n".join(lines) + "\n")

    (mol, scores), = read_pdbqt_poses(path)
    assert scores["vina_affinity"] == -9.3
    # 6 carbons kept, both glue atoms dropped.
    assert mol.GetNumAtoms() == 6
    assert {a.GetSymbol() for a in mol.GetAtoms()} == {"C"}


def test_report_exclude(tmp_path, capsys):
    """--exclude drops targets from the denominator as well as from the rows."""
    from cofoldbench.cli import main
    rows = []
    for tid in ("T1", "T2", "BIG"):
        for method, rmsd in (("vina", 1.0), ("chai1", 0.5)):
            if tid == "BIG" and method == "chai1":
                continue                      # ran out of memory, no pose
            rows.append(dict(target_id=tid, method=method, rank=1, score=1.0, score_name="s",
                             status="ok", ligand_rmsd=rmsd, centroid_dist=0.5, pocket_ca_rmsd=0.2,
                             n_pocket_ca=20, min_dist=3.0, n_clashes=0, graph_ok=True,
                             recovery_all=0.8, recovery_hydrophobic=0.8, recovery_polar=0.8,
                             recovery_ionic=0.8, fp_tanimoto=0.8, n_ref_interactions=10,
                             aligned_pdb=""))
    results = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(results, index=False)

    # Without --exclude: chai1 is scored over all three targets, so 2/3.
    main(["report", "--results", str(results), "--outdir", str(tmp_path / "r1"),
          "--targets", "", "--no-plots"])
    s = pd.read_csv(tmp_path / "r1" / "summary.csv").set_index(["method", "subset"])
    assert s.loc[("chai1", "all"), "top1_success"] == pytest.approx(2 / 3)
    assert s.loc[("chai1", "all"), "n_missing"] == 1

    # Excluding the target no method could run: chai1 is 2/2.
    main(["report", "--results", str(results), "--outdir", str(tmp_path / "r2"),
          "--targets", "", "--exclude", "BIG", "--no-plots"])
    s = pd.read_csv(tmp_path / "r2" / "summary.csv").set_index(["method", "subset"])
    assert s.loc[("chai1", "all"), "top1_success"] == 1.0
    assert s.loc[("chai1", "all"), "n_missing"] == 0
    assert s.loc[("chai1", "all"), "n_targets"] == 2


def _tiny_crystal(lig_x: list[float], cell: float = 12.0) -> gemmi.Structure:
    """A P1 crystal with a short alanine chain near the origin and a ligand.

    With a small cell, a ligand placed near the far edge contacts the lattice
    translation of the protein, i.e. a symmetry mate rather than a chain of the
    deposited model.
    """
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(cell, cell, cell, 90, 90, 90)
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for i in range(1, 4):
        res = gemmi.Residue()
        res.name, res.seqid = "ALA", gemmi.SeqId(i, " ")
        for name, dx in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 2.5), ("CB", 1.5)):
            atom = gemmi.Atom()
            atom.name, atom.element = name, gemmi.Element("C" if name != "N" and name != "O" else name)
            atom.pos = gemmi.Position(1.0 + dx, float(i) * 1.5, 0.0)
            res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    lig_chain = gemmi.Chain("L")
    lig = gemmi.Residue()
    lig.name, lig.seqid, lig.het_flag = "LIG", gemmi.SeqId(1, " "), "H"
    for k, x in enumerate(lig_x):
        atom = gemmi.Atom()
        atom.name, atom.element = f"C{k + 1}", gemmi.Element("C")
        atom.pos = gemmi.Position(x, 3.0, 0.0)
        lig.add_atom(atom)
    lig_chain.add_residue(lig)
    model.add_chain(lig_chain)
    st.add_model(model)
    st.setup_entities()
    return st


def test_symmetry_mate_site_is_flagged():
    # Ligand beside the protein in the same copy: no symmetry warning.
    near = fetch.prepare(_tiny_crystal([4.0, 5.0, 6.0]), "OWN", "LIG")
    assert not any("SYMMETRY" in w for w in near.warnings), near.warnings

    # Ligand spanning the cell edge, as at an interface: part of its contacts
    # come from the deposited chain, part from the lattice translation.
    far = fetch.prepare(_tiny_crystal([6.5, 9.5, 10.5]), "SYM", "LIG")
    assert any("SITE SPANS A SYMMETRY MATE" in w for w in far.warnings), far.warnings


def test_symmetry_check_skipped_without_a_cell():
    st = _tiny_crystal([4.0, 5.0, 6.0])
    st.cell = gemmi.UnitCell()        # no crystal information, e.g. a predicted model
    st.spacegroup_hm = ""
    prep = fetch.prepare(st, "NOCELL", "LIG")
    assert not any("SYMMETRY" in w or "symmetry" in w for w in prep.warnings), prep.warnings


def test_year_windows_cover_the_range_without_gaps():
    w = search.year_windows("2015-06-01", "2021-01-01")
    assert w[0] == ("2015-06-01", "2016-01-01")      # starts at the given date
    assert w[-1] == ("2020-01-01", "2021-01-01")     # ends at the given date
    for (_, hi), (lo, _) in zip(w, w[1:]):
        assert hi == lo                              # contiguous, no gaps
    assert search.year_windows("2020-03-01", "2020-09-01") == [("2020-03-01", "2020-09-01")]
    assert search.year_windows("2021-01-01", "2021-01-01") == []


def test_search_query_date_bounds():
    q = search.search_query("2016-01-01", 2.0, 100, before="2017-01-01")
    dates = {n["parameters"]["operator"]: n["parameters"]["value"]
             for n in q["query"]["nodes"]
             if n["parameters"]["attribute"].endswith("initial_release_date")}
    assert dates == {"greater": "2016-01-01", "less": "2017-01-01"}
    # Without `before` there is no upper bound.
    q = search.search_query("2023-06-01", 2.0, 100)
    assert sum(1 for n in q["query"]["nodes"]
               if n["parameters"]["attribute"].endswith("initial_release_date")) == 1


def test_additive_ligands_are_rejected():
    """A PEG or detergent is large enough to pass a weight filter but is not a ligand."""
    def entry(pdb, lig_id, lig_name, mw):
        return {"rcsb_id": pdb,
                "rcsb_accession_info": {"initial_release_date": "2016-12-01T00:00:00Z"},
                "rcsb_entry_info": {"resolution_combined": [1.6]},
                "polymer_entities": [{"entity_poly": {"rcsb_sample_sequence_length": 286},
                                      "rcsb_polymer_entity": {"pdbx_description": f"protein {pdb}"},
                                      "rcsb_polymer_entity_container_identifiers":
                                          {"auth_asym_ids": ["A"]}}],
                "nonpolymer_entities": [{"nonpolymer_comp": {"chem_comp": {
                    "id": lig_id, "name": lig_name, "formula_weight": mw,
                    "type": "NON-POLYMER"}}}]}
    kept = search.filter_entries([
        entry("5AO9", "PE8", "3,6,9,12,15,18,21-heptaoxatricosane-1,23-diol", 428.0),
        entry("39JO", "P6G", "hexaethylene glycol", 282.0),
        entry("1DET", "BOG", "octyl beta-D-glucoside", 292.0),
        entry("5MME", "8Q6", "dimethyl 5-[(5-ethanoyl-2-ethoxy-phenyl)amino]benzene"
                             "-1,3-dicarboxylate", 385.0),
    ])
    assert list(kept["pdb_id"]) == ["5MME"]
    # PE8 is also in the structure-level additive list, so it is never picked as
    # the ligand of a complex either.
    from cofoldbench.structure import ADDITIVES
    assert {"PE8", "P6G", "PGE"} <= ADDITIVES


def test_search_pagination_offset_is_seeded():
    q = search.search_query("2016-01-01", 2.0, 100, before="2017-01-01", start=250)
    assert q["request_options"]["paginate"] == {"start": 250, "rows": 100}
