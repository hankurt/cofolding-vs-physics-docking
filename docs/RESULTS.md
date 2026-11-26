# Results: co-folding vs physics-based docking

**One run, 59 targets, 5 methods, 1437 scored poses.** Boltz-2 (blind and
pocket-conditioned), Chai-1, GNINA and AutoDock Vina, on one RTX 4070 Ti
(12 GB). Every number below comes from `results/results.csv`; reproduce the
tables with `cofoldbench report` and `cofoldbench stats`.

The target set is deliberately split in three:

* **22 classic complexes** (imatinib/ABL, erlotinib/EGFR, sildenafil/PDE5,
  venetoclax/BCL-2 …), released long before every model's training cut-off and
  heavily represented in it;
* **21 obscure complexes released 2015–2023**, also before the cut-off but
  ordinary and little-cited — the control that separates *heavily represented*
  from merely *seen*;
* **16 complexes released after 2023-06-01**, the later of the two training
  cut-offs (Boltz-2; Chai-1 2021-01-12). Most were released in late 2026,
  weeks before the run.

---

## Headline

Being released before a model's cut-off is not one condition but two. The
classic drug complexes are in the training data many times over — every
benchmark, every tutorial, redundant entries — while an ordinary structure from
the same years is in it once. Splitting on that gives a three-level gradient
that a simple time split hides:

| Top-1 success (RMSD ≤ 2 Å) | famous, pre-cut-off (n=19) | **obscure, pre-cut-off (n=21)** | after cut-off (n=15) |
|---|---|---|---|
| Boltz-2 (blind) | **95%** [75–99] | **71%** [50–86] | **27%** [11–52] |
| Boltz-2 (pocket) | **95%** [75–99] | 81% [60–92] | **27%** [11–52] |
| Chai-1 | 89% [69–97] | 48% [28–68] | 27% [11–52] |
| GNINA | 74% [51–88] | 71% [50–86] | **73%** [48–89] |
| AutoDock Vina | 63% [41–81] | 62% [41–79] | 47% [25–70] |

**Docking is flat across all three levels. Co-folding falls monotonically.**
GNINA scores 74% / 71% / 73%; Boltz-2 scores 95% / 71% / 27%.

Two consequences, and the first was a surprise:

**1. Mere presence in the training set is not what drives co-folding's
accuracy.** The obscure pre-cut-off targets were released 2015–2023 and are
every bit as "seen" as the classics, yet Boltz-2 drops from 95% to 71% and
Chai-1 from 89% to 48%. What distinguishes the classics is not exposure but
*redundancy* — how many times the same complex appears.

**2. On ordinary seen structures, co-folding has no advantage over docking.**
Boltz-2 and GNINA both score 71% on the obscure pre-cut-off set, and paired on
those same 21 targets they are indistinguishable (3 targets each way,
p = 1.0). Co-folding's apparent superiority is confined to famous complexes;
its deficit is confined to unseen ones.

**On post-cut-off targets GNINA beats both co-folding models**, paired on the
same targets: GNINA vs Boltz-2 **p = 0.039** (8 targets to 1), GNINA vs Chai-1
**p = 0.016** (7 to 0).

**Aggregated over all 55 targets the methods barely separate** (GNINA 73%,
Boltz-2-pocket 71%, Boltz-2 67%, Vina 58%, Chai-1 56%). Of the ten paired
comparisons only Boltz-2 (pocket) vs Chai-1 reaches significance (p = 0.02);
GNINA vs Boltz-2, the comparison the whole question turns on, is p = 0.65.
The overall average hides the entire structure of the result — which is the
main methodological point of this benchmark.

### Matched pairs

Three proteins appear on both sides of the cut-off, holding protein identity
and chemotype roughly fixed so that only the release date moves:

| protein | pre-cut-off | post-cut-off | Boltz-2 | GNINA |
|---|---|---|---|---|
| BRD4 | 3MXF (2010) | 9T15 (2026) | 0.32 → 3.16 Å | 0.59 → 6.72 Å |
| BTK | 8FLH (2023-03) | 9T1T (2026-11) | 1.14 → 4.79 Å | 2.38 → 1.86 Å |
| DCAF1 | 8F8E (2023-03) | 9YV2 (2026-11) | 0.82 → 2.71 Å | 0.86 → 0.86 Å |

Boltz-2 succeeds on all three pre-cut-off members and fails on all three
post-cut-off ones. GNINA is 2/3 either side. With three pairs this is
suggestive, not significant (McNemar p = 0.25 at best), but it is the cleanest
design in the set and it points the same way as the aggregate. Note that BRD4's
post-cut-off ligand defeats every method, so that pair isolates protein
identity rather than difficulty.

## Is it memorisation, or is the recent set just harder?

Three checks, because this is the claim the whole project turns on.

**1. The failures are not in the awkward ligands.** The post-cut-off set
contains some intrinsically nasty ligands (a sphingoid lipid, ADP-ribose, a
macrocyclic natural product, a myristoyl-peptide analogue). If "recent" were a
proxy for "hard ligand", Boltz-2 should fail on those and succeed on the
conventional drugs. It does not:

| post-cut-off ligands | n | Boltz-2 | GNINA |
|---|---|---|---|
| conventional drug-like | 12 | 25% | 67% |
| awkward (lipid, nucleotide, macrocycle, peptide-like) | 4 | 25% | 75% |

Boltz-2 fails at the same rate on both. It missed PLK4, BTK, BRD4, soluble
epoxide hydrolase, WDR5, DCAF1 and CBL-B — ordinary medicinal-chemistry
targets with ordinary ligands, the same kind it placed to 0.3–0.6 Å before the
cut-off. It *succeeded* on ADP-ribose (0.26 Å), one of the "awkward" ones.

**2. Those targets are dockable.** GNINA solved 8 targets Boltz-2 missed
(29IA, 37OC, 9T1T, 9T3U, 9UY8, 9XJQ, 9YV2, 9ZBG). They cannot be dismissed as
intrinsically unsolvable.

**3. The protein is predicted well; the ligand is still wrong.** Restricting Boltz-2 to targets whose pocket it predicted
almost perfectly (pocket Cα RMSD ≤ 1 Å after superposition):

| Boltz-2, pocket Cα RMSD ≤ 1 Å | n | ligand success |
|---|---|---|
| before cut-off | 33 | **27/33 (82%)** |
| after cut-off | 10 | **3/10 (30%)** |

Boltz-2's protein prediction does degrade on novel targets (median pocket Cα
RMSD 0.38 Å → 0.76 Å), but that is not what is driving the ligand failures:
even where the pocket comes out essentially right, the ligand lands correctly
82% of the time before the cut-off and 30% after.

**4. And the obscure pre-cut-off control settles the biggest one.** If the
post-cut-off drop were just "this target set is harder", the obscure
pre-cut-off set — ordinary structures, ordinary ligands, no fame — should look
like the post-cut-off set. It does not: Boltz-2 gets 71% there against 27%
after the cut-off. Equally, if exposure alone explained the 95%, the obscure
set should match the classics. It does not either. Both halves of the gradient
are real.

---

## Other findings

**Telling Boltz-2 where the pocket is does not fix the novel targets.** The
constraint buys a little overall (71% vs 67% on the 55 scored targets, 3
discordant targets to 1, p = 0.63) and most of that comes from the obscure
pre-cut-off set (81% vs 71%). After the cut-off it buys nothing: 27% either
way, 1 discordant target each way, p = 1.0. The constraint does shrink the
error on several bad poses — 9XJQ 6.75 → 1.55 Å, 9UY8 7.95 → 2.42 Å, 35XE
7.96 → 3.30 Å — but rarely below 2 Å. So co-folding's problem on novel targets
is not that it looks in the wrong place.

**Co-folding is clearly better where docking's conformer search struggles.**
1YET (geldanamycin, a macrocycle): Boltz-2 0.68 Å vs GNINA 9.12 Å and Vina
6.52 Å. 1HSG (indinavir, HIV-1 protease dimer): Boltz-2 0.62 Å vs Vina
10.52 Å. Macrocycles and large flexible ligands are a real weakness of
docking, and co-folding handles them well.

**GNINA > Vina, but not significantly** (73% vs 58%, p = 0.10): a plausible
benefit from CNN rescoring that this target set cannot establish.

**Interaction recovery tracks RMSD but is less discriminating** — all methods
recover 81–89% of crystal contacts on average, even when the pose is wrong,
because a ligand in roughly the right pocket touches roughly the right
residues. Pose RMSD is the sharper measure.

**Physical validity was rarely the problem.** `top1_success_valid` equals
`top1_success` for GNINA, Vina and Chai-1. Only Boltz-2 loses anything to it:
2 of its successful top-1 poses clash with the receptor, blind and
pocket-conditioned alike (37 → 35 and 39 → 37 of 55). Across all 1437 poses
just 2 had a ligand graph mismatch, though 60 contain at least one
protein-ligand pair closer than 2.2 Å.

---

## Limitations

These matter for how far the headline can be pushed.

**1. This is re-docking into the crystal receptor, which favours docking.**
GNINA and Vina were given the holo receptor from the reference structure — its
pocket Cα RMSD is 0.00 Å by construction — while the co-folding models had to
predict the protein *and* the ligand from sequence. That is a large,
asymmetric advantage, and it is the single biggest caveat here. Cross-docking
into apo or MD-ensemble receptors (on the roadmap) is where docking should
lose ground.

**2. n = 15 scored after the cut-off** (16 in the set, less 9XHG). The paired
tests reach p < 0.05, but the interval on Boltz-2's post-cut-off rate is
27% [11–52%]. The *direction* is solid; the exact number is not.

**3. The target set excludes whole classes of target.** Covalent ligands,
pockets needing a catalytic metal (four carbonic anhydrases among them) or a
cofactor such as myristoyl-CoA, and crystallisation additives were all
filtered out, because the co-folding inputs and the docking receptor represent
neither metals nor cofactors. **The conclusions therefore apply to simple,
non-covalent, metal-free pockets only** — not to metalloenzymes, which are a
large slice of real drug targets. This is a gap in the tooling, not a finding
about the methods.

**4. One target has a broken site.** 9XHG (STING) binds its agonist at the
homodimer interface, and only one chain is in the deposited model, so every
method failed (9–28 Å). `cofoldbench fetch` now warns about this; the target
should be rebuilt with both chains or dropped. Excluding it does not change
any conclusion. 9ZBG also deserves a look: blind Boltz-2 put the pocket 19.7 Å
out, while the pocket-conditioned run placed it at 0.8 Å.

**5. Five targets are missing from Chai-1,** all of them the longest
constructs in the set: COX-2 (587), acetylcholinesterase (543), HIV-1 reverse
transcriptase p66, MurE (496) and acetyl-CoA synthetase (694 residues). None
fit Chai-1 on a 12 GB GPU. The first three are the pre-cut-off classics the
tables above exclude for every method (`--exclude 1EVE,1VRT,3LN1`); the other
two are obscure pre-cut-off targets that stay in and count as failures
(`n_missing = 2`), so Chai-1's 48% on that subset is a floor rather than a
measurement. All five are pre-cut-off, so the missing poses *understate*
Chai-1's before/after drop. That is a hardware limit, not a model one.

**6. The three-way split rests on a judgement about which targets are
"famous".** The classic set was chosen as recognisable drug complexes and the
control set was sampled at random from the same era, so the labels are not
arbitrary — but "famous" is not quantified here. Counting citations, or PDB
sequence-cluster redundancy, would put a number on it. Imatinib/ABL and
sildenafil/PDE5 are not just in the training data, they are in it many times
over. So this split confounds *seen in training* with *heavily represented*,
and the 95% may partly measure fame rather than exposure. That is exactly what
the obscure pre-cut-off control set is for, and it is what the three-level
table reports — but the *labelling* of a target as famous or obscure remains a
judgement call, and moving a single borderline target between the two columns
shifts those rates by a few points.

**7. One seed per method.** Docking is stochastic and no run-to-run variance
was measured, so the precision implied by a single number is optimistic
(`SEED=2 METHOD_SUFFIX=_s2 scripts/run_docking.sh` collects a second seed as
separate methods).

**8. Not yet analysed:** Boltz-2's predicted affinities (collected but
unexamined) and full PLIP interaction profiles.

---

## Reproducing

```bash
cofoldbench report --exclude 1EVE,1VRT,3LN1,9XHG --split exposure --outdir results/report_fits
cofoldbench stats  --exclude 1EVE,1VRT,3LN1,9XHG --split exposure --per-target
```

`--exclude` drops the three targets Chai-1 cannot fit on a 12 GB GPU and
9XHG, whose binding site is incomplete in the deposited model. Both
exclusions follow a priori structural criteria, decided before any pose was
scored.

`results/report/` holds the same tables over all 59 targets with a plain
before/after time split, where Chai-1 carries `n_missing = 5`.
