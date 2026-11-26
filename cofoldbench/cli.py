"""Command-line interface: ``cofoldbench <search|fetch|inputs|dockprep|collect|evaluate|report|stats>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import collect, inputs, report
from .evaluate import evaluate
from .targets import load_targets, read_csv_skip_comments


def cmd_search(args):
    from .search import search
    out = Path(args.out)
    # Raw hits must never overwrite a curated candidate list: the curated files
    # carry a comment header explaining what was left out and why, and that
    # cannot be regenerated.
    if out.exists() and not args.force:
        head = out.read_text()[:4096]
        if any(ln.startswith("#") for ln in head.splitlines()):
            raise SystemExit(
                f"{out} looks curated (it has comment lines). Writing raw search hits there "
                f"would destroy that.\n  Use --out data/candidates_search.csv, or --force to "
                f"overwrite {out} anyway.")
    df = search(args.after, n=args.n, max_resolution=args.max_resolution, before=args.before, seed=args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False) if len(df) else "no hits")
    print(f"\nwrote {len(df)} candidate(s) -> {out}. Review them, then: cofoldbench fetch {out}")


def cmd_fetch(args):
    from . import fetch
    targets_path = Path(args.targets)
    structures = targets_path.parent / "structures"
    cand = read_csv_skip_comments(args.candidates)
    comments, table = fetch.read_targets_table(targets_path)
    have = set(table["target_id"].str.upper())
    added, failed = [], []
    for rec in cand.to_dict("records"):
        pdb = rec["pdb_id"].strip().upper()
        if pdb in have and not args.force:
            print(f"{pdb}: already in {targets_path}, skipped (--force to redo)")
            continue
        chains = rec.get("chain", "").replace(",", " ").split() or None
        try:
            prep = fetch.fetch_one(pdb, rec["ligand_resname"].strip().upper(), chains, structures,
                                   cache_dir=Path(args.cache))
        except Exception as exc:  # report and carry on with the other candidates
            print(f"{pdb}: FAILED - {exc}")
            failed.append(pdb)
            continue
        rel = prep.structure_path.relative_to(targets_path.parent).as_posix()
        note = "; ".join(x for x in [rec.get("protein", ""), rec.get("ligand_name", ""),
                                     rec.get("set", "")] if x)
        row = fetch.target_row(prep, rel, note)
        table = pd.concat([table[table["target_id"].str.upper() != pdb], pd.DataFrame([row])],
                          ignore_index=True)
        have.add(pdb)
        added.append(pdb)
        print(f"{pdb}: ok  chain {row['chain']}  {prep.n_residues} res  {len(prep.pocket_residues)} pocket res  "
              f"released {row['release_date'] or '?'}  {row['resolution'] or '?'} A")
        for w in prep.warnings:
            print(f"      warning: {w}")
        fetch.write_targets_table(targets_path, comments, table)  # save as we go
    print(f"\nadded {len(added)}, failed {len(failed)}{': ' + ' '.join(failed) if failed else ''}; "
          f"{len(table)} targets in {targets_path}")


def cmd_dockprep(args):
    from .dockprep import write_dock_inputs, write_pdbqt
    targets = load_targets(args.targets)
    try:
        import meeko, openbabel  # noqa: F401
        pdbqt = not args.no_pdbqt
    except ImportError:
        pdbqt = False
        if not args.no_pdbqt:
            print('warning: Open Babel / Meeko missing - no PDBQT files for Vina. pip install -e ".[dock]"',
                  file=sys.stderr)
    for d in write_dock_inputs(targets, args.outdir):
        files = "receptor.pdb ligand.sdf box.txt"
        if pdbqt:
            write_pdbqt(d)
            files += " receptor_h.pdb receptor.pdbqt ligand.pdbqt"
        print(f"{d}: {files}")


def cmd_inputs(args):
    targets = load_targets(args.targets)
    for w in inputs.write_inputs(targets, args.outdir):
        print(f"warning: {w}", file=sys.stderr)
    print(f"wrote inputs for {len(targets)} target(s) to {args.outdir}")


def cmd_collect(args):
    engine = args.engine
    if engine in ("boltz", "chai", "af3"):
        fn = getattr(collect, engine)
        rows = fn(args.results, **({"method": args.method} if args.method else {}))
    elif engine in ("gnina", "vina"):
        kw = {"method": args.method} if args.method else {}
        if engine == "gnina" and args.score_name:
            kw["score_name"] = args.score_name
        rows = getattr(collect, engine)(args.target, args.receptor, args.poses, **kw)
    elif engine == "haddock3":
        rows = collect.haddock3(args.target, args.run_dir, **({"method": args.method} if args.method else {}))
    else:
        rows = collect.single(args.target, args.structure, args.method, args.ligand)
    df = collect.write_manifest(rows, args.manifest, append=not args.overwrite)
    print(f"collected {len(rows)} pose(s); manifest now has {len(df)} rows -> {args.manifest}")
    if rows and all(pd.isna(r["score"]) for r in rows):
        name = rows[0]["score_name"] or "score"
        print(f"warning: no {name} found in the output; poses keep the engine's file order",
              file=sys.stderr)


def cmd_evaluate(args):
    targets = load_targets(args.targets)
    manifest = pd.read_csv(args.manifest, dtype={"ligand": str}, keep_default_na=False)
    manifest["score"] = pd.to_numeric(manifest["score"], errors="coerce")
    res = evaluate(targets, manifest, aligned_dir=args.aligned_dir, max_rank=args.max_rank,
                   pocket_cutoff=args.pocket_cutoff, verbose=args.verbose)
    res.to_csv(args.out, index=False)
    bad = res[~res["status"].isin(["ok"])]
    print(f"evaluated {len(res)} pose(s) -> {args.out}  ({len(bad)} not ok)")
    for rec in bad.head(10).to_dict("records"):
        print(f"  {rec['target_id']} {rec['method']} rank {rec['rank']}: {rec['status']}", file=sys.stderr)


def cmd_stats(args):
    from . import stats
    res = pd.read_csv(args.results)
    res["target_id"] = res["target_id"].astype(str)
    targets = load_targets(args.targets) if args.targets and Path(args.targets).exists() else []
    ids = {t.target_id for t in targets} or set(res["target_id"])
    if args.exclude:
        drop = {x.strip().upper() for x in args.exclude.replace(",", " ").split()}
        ids = {t for t in ids if t.upper() not in drop}
    print(f"Top-1 success (RMSD <= {args.success_rmsd:g} A) over {len(ids)} targets, "
          "95% Wilson intervals\n")
    print(stats.format_table(stats.rate_table(res, ids, args.success_rmsd)))
    print("\nPaired comparison on the same targets (exact McNemar; a_only/b_only "
          "are the targets where they disagree)\n")
    print(stats.format_table(stats.pairwise(res, ids, args.success_rmsd)))
    if args.per_target:
        split_all = (report.time_split({t.target_id: t.release_date for t in targets}, args.cutoff)
                     if targets else {})
        notes = {}
        for t in targets:
            note = getattr(t, "note", None)
            if note:
                notes[t.target_id] = str(note)[:28]
        print("\nTop-1 ligand RMSD per target (A); 'pocket_*' is the predicted pocket Ca RMSD, "
              "'.' = no pose\n")
        print(stats.format_per_target(
            stats.per_target(res, ids, split_all, notes or None, args.success_rmsd)))
        print()
    if targets:
        split = (report.exposure_split(targets, args.cutoff) if args.split == "exposure"
                 else report.time_split({t.target_id: t.release_date for t in targets}, args.cutoff))
        subsets = {}
        for tid, label in split.items():
            if tid in ids:
                subsets.setdefault(label, set()).add(tid)
        order = [l for l in report.SPLIT_ORDER if l in subsets]
        if len(order) >= 2:
            print(f"\nTop-1 success by exposure (cut-off {args.cutoff}); "
                  "95% Wilson intervals\n")
            for label in order:
                print(f"  {label} (n = {len(subsets[label])})")
                print(stats.format_table(stats.rate_table(res, subsets[label], args.success_rmsd)))
                print()
            print(f"{order[0]} vs {order[-1]}; unpaired, so the interval is wide\n")
            print(stats.format_table(stats.subset_difference(
                res, subsets, order[0], order[-1], args.success_rmsd)))
            for label in order[1:]:
                print(f"\n{label} only, paired between methods\n")
                print(stats.format_table(stats.pairwise(res, subsets[label], args.success_rmsd)))


def cmd_report(args):
    res = pd.read_csv(args.results)
    subsets, ids = None, None
    if args.targets and Path(args.targets).exists():
        targets = load_targets(args.targets)
        ids = [t.target_id for t in targets]
        subsets = (report.exposure_split(targets, args.cutoff) if args.split == "exposure"
                   else report.time_split({t.target_id: t.release_date for t in targets},
                                          args.cutoff)) or None
    if args.exclude:
        drop = {x.strip().upper() for x in args.exclude.replace(",", " ").split()}
        before = len(res)
        res = res[~res["target_id"].astype(str).str.upper().isin(drop)]
        if ids:
            ids = [t for t in ids if t.upper() not in drop]
        print(f"excluded {len(drop)} target(s) ({' '.join(sorted(drop))}): "
              f"{before - len(res)} pose(s) dropped, {len(ids) if ids else '?'} targets left")
    missing = report.missing_poses(res, ids)
    if missing:
        print("warning: no pose at all for some targets; they count as failures "
              "(a co-folding model that ran out of memory, or a crashed docking run):",
              file=sys.stderr)
        for method, gone in sorted(missing.items()):
            print(f"  {method}: {len(gone)} missing -> {' '.join(gone)}", file=sys.stderr)
    summary = report.summarize(res, top_k=args.top_k, subsets=subsets, targets=ids)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "summary.csv", index=False)
    md = report.to_markdown(summary)
    (outdir / "summary.md").write_text(md)
    print(md)
    if not args.no_plots:
        for p in report.plot(res, summary, outdir):
            print(f"wrote {p}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cofoldbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="find recent PDB complexes (time-split candidates; needs internet)")
    s.add_argument("--after", default=report.DEFAULT_CUTOFF, help="released after this date (default: %(default)s)")
    s.add_argument("--before", help="released strictly before this date; the range is then "
                                    "sampled year by year (for a pre-cut-off control set)")
    s.add_argument("--n", type=int, default=20, help="number of candidates")
    s.add_argument("--seed", type=int, default=0, help="sampling seed when --before is given")
    s.add_argument("--force", action="store_true",
                   help="overwrite --out even if it looks like a curated list")
    s.add_argument("--max-resolution", type=float, default=2.0)
    s.add_argument("--out", default="data/candidates_search.csv",
                   help="raw hits to review (default: %(default)s); the curated set is "
                        "data/candidates_recent.csv")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("fetch", help="download, clean and validate candidates into targets.csv (needs internet)")
    s.add_argument("candidates", help="CSV with pdb_id, ligand_resname and optional chain columns")
    s.add_argument("--targets", default="data/targets.csv")
    s.add_argument("--cache", default="data/raw", help="downloaded mmCIF / CCD files")
    s.add_argument("--force", action="store_true", help="redo targets already in targets.csv")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("dockprep", help="write receptor / ligand / box for GNINA and Vina")
    s.add_argument("--targets", default="data/targets.csv")
    s.add_argument("--outdir", default="inputs/dock")
    s.add_argument("--no-pdbqt", action="store_true", help="skip hydrogens / PDBQT (Open Babel + Meeko)")
    s.set_defaults(func=cmd_dockprep)

    s = sub.add_parser("inputs", help="write Boltz-2 / Chai-1 / AF3 inputs from targets.csv")
    s.add_argument("--targets", default="data/targets.csv")
    s.add_argument("--outdir", default="inputs")
    s.set_defaults(func=cmd_inputs)

    s = sub.add_parser("collect", help="add an engine's poses to the manifest")
    s.add_argument("engine", choices=["boltz", "chai", "af3", "gnina", "vina", "haddock3", "single"])
    s.add_argument("--manifest", default="results/manifest.csv")
    s.add_argument("--method", help="method label (default: engine name, e.g. boltz2)")
    s.add_argument("--results", help="output directory (boltz / chai / af3)")
    s.add_argument("--target", help="target id (gnina / vina / haddock3 / single)")
    s.add_argument("--receptor", help="receptor PDB the ligand was docked into (gnina / vina)")
    s.add_argument("--poses", help="docked poses: SDF (gnina) or PDBQT (vina)")
    s.add_argument("--run-dir", help="HADDOCK3 run directory")
    s.add_argument("--structure", help="complex or receptor file (single)")
    s.add_argument("--ligand", help="separate ligand file (single)")
    s.add_argument("--score-name", help="SDF property used as the GNINA score (default CNNscore)")
    s.add_argument("--overwrite", action="store_true", help="start a new manifest")
    s.set_defaults(func=cmd_collect)

    s = sub.add_parser("evaluate", help="score all poses in the manifest")
    s.add_argument("--targets", default="data/targets.csv")
    s.add_argument("--manifest", default="results/manifest.csv")
    s.add_argument("--out", default="results/results.csv")
    s.add_argument("--aligned-dir", default="results/aligned",
                   help="write each pose in the reference frame and numbering (PLIP-ready)")
    s.add_argument("--max-rank", type=int, default=5, help="only evaluate the top-N poses per method")
    s.add_argument("--pocket-cutoff", type=float, default=10.0)
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("stats", help="success rates with confidence intervals and paired tests")
    s.add_argument("--results", default="results/results.csv")
    s.add_argument("--targets", default="data/targets.csv")
    s.add_argument("--cutoff", default=report.DEFAULT_CUTOFF)
    s.add_argument("--exclude", help="target ids to leave out")
    s.add_argument("--success-rmsd", type=float, default=report.SUCCESS_RMSD)
    s.add_argument("--split", choices=["time", "exposure"], default="time",
                   help="'exposure' splits the pre-cut-off half into famous and obscure targets, "
                        "which separates heavily represented from merely seen")
    s.add_argument("--per-target", action="store_true",
                   help="also print the top-1 RMSD of every target and method")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("report", help="summary table and figures")
    s.add_argument("--results", default="results/results.csv")
    s.add_argument("--outdir", default="results/report")
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--targets", default="data/targets.csv",
                   help="for release dates: adds before/after training cut-off rows")
    s.add_argument("--cutoff", default=report.DEFAULT_CUTOFF, help="training cut-off date (default: %(default)s)")
    s.add_argument("--exclude", help="target ids to leave out, e.g. ones no method could run "
                                     "on this GPU (comma- or space-separated)")
    s.add_argument("--split", choices=["time", "exposure"], default="time",
                   help="'exposure' splits the pre-cut-off half into famous and obscure targets")
    s.add_argument("--no-plots", action="store_true")
    s.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    if args.cmd == "collect":
        need = {"boltz": ["results"], "chai": ["results"], "af3": ["results"],
                "gnina": ["target", "receptor", "poses"], "vina": ["target", "receptor", "poses"],
                "haddock3": ["target", "run_dir"], "single": ["target", "structure", "method"]}[args.engine]
        missing = [f"--{n.replace('_', '-')}" for n in need if not getattr(args, n)]
        if missing:
            p.error(f"collect {args.engine} needs {' '.join(missing)}")
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    elif args.cmd == "evaluate":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
