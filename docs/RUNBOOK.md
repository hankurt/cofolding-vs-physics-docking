# Runbook: the full benchmark on one workstation (no sudo)

Written for a shared Linux PC with one NVIDIA GPU (tested target: GeForce
RTX 4070 Ti, 12 GB, driver 535). Everything installs into your home
directory. Nothing needs root.

The plan has two phases:

* **Phase A — smoke test on 6DPT** (one target, every tool). Small, fast,
  and it shakes out installation problems before you spend GPU hours.
* **Phase B — the full target set**: classic complexes plus a time-split set
  released after the models' training cut-offs.

At each **📋 checkpoint** paste the requested output to your assistant (or
push it to the branch) before moving on.

---

## 0. Before you start

```bash
nvidia-smi            # GPU free? On a shared PC, check nobody else is using it
df -h ~               # need ~30 GB free: model weights (~10 GB) + MSAs + outputs
which conda           # conda available?
```

No conda? Install Miniforge in your home directory (no sudo):

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash && exec bash
```

Long jobs: run them with `nohup ... &` (shown below) so they survive if your
SSH session drops; check progress with `tail -f logs/<name>.log`.

## 1. Get the code

```bash
git clone https://github.com/hankurt/cofolding-vs-physics-docking.git
cd cofolding-vs-physics-docking
git checkout claude/festive-einstein-ao6qkz      # until it is merged into main
mkdir -p logs
```

(Already cloned? `git fetch origin && git checkout claude/festive-einstein-ao6qkz && git pull`.)

## 2. Environments

Three conda environments; the prediction tools pin different PyTorch versions,
so they are kept apart.

**Evaluation + docking prep** (no GPU):

```bash
conda env create -f environment.yml          # name: cofoldbench
conda activate cofoldbench
pytest -q                                    # expect: 25 passed
```

**Boltz-2:**

```bash
conda create -n boltz python=3.11 -y
conda activate boltz
pip install boltz
scripts/fix_torch_cuda.sh                    # only changes torch if CUDA does not work
```

**Chai-1:**

```bash
conda create -n chai python=3.11 -y
conda activate chai
pip install chai_lab "pyarrow<21"           # pyarrow: Chai saves its MSAs as parquet
scripts/fix_torch_cuda.sh
```

**Docking binaries** (GNINA, AutoDock Vina into `~/.local/bin`):

```bash
scripts/install_docking_tools.sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && exec bash   # if it told you to
```

> 📋 **Checkpoint 1:** `conda activate cofoldbench && scripts/check_env.sh`. Paste the output.

## Phase A — smoke test on 6DPT

```bash
conda activate cofoldbench
cofoldbench inputs          # Boltz YAML / Chai FASTA
cofoldbench dockprep        # receptor, ligand (from SMILES), box, PDBQT
```

Co-folding (the first run downloads model weights, several GB). One target
takes minutes, so run in the foreground; `tee` keeps a log:

```bash
conda activate boltz
scripts/run_boltz.sh blind  2>&1 | tee logs/boltz_blind.log
scripts/run_boltz.sh pocket 2>&1 | tee logs/boltz_pocket.log
conda activate chai
scripts/run_chai.sh 2>&1 | tee logs/chai.log
```

Docking:

```bash
conda activate cofoldbench
scripts/run_docking.sh 2>&1 | tee logs/docking.log
```

Evaluate:

```bash
cofoldbench evaluate
cofoldbench report
```

> 📋 **Checkpoint 2:** paste `results/report/summary.md` and the last lines of
> any log that shows an error (`tail -30 logs/<name>.log`).

Each run script skips targets that already have output, so after fixing a
problem just rerun it. If one tool fails, the others still count; the report
shows whatever has been collected.

## Phase B — the full target set

### B1. Classic targets

```bash
conda activate cofoldbench
cofoldbench fetch data/candidates.csv
```

Every candidate is downloaded from RCSB, cleaned (receptor chain + one ligand
copy) and **checked**: the CCD SMILES must have the same heavy-atom graph as
the crystal ligand, otherwise every pose would be scored against a
different molecule. Failures are listed and skipped; warnings flag a cofactor
next to the ligand, a **covalent ligand** (drop those: no non-covalent method can
reproduce them) or a protein too large for 12 GB.

> 📋 **Checkpoint 3:** paste the full `fetch` output.

### B2. Time-split targets (the generalisation test)

```bash
cofoldbench fetch data/candidates_recent.csv   # the curated set in this repo
```

To build your own instead, search and review the raw hits first:

```bash
cofoldbench search --after 2023-06-01 --n 45   # -> data/candidates_search.csv
cat data/candidates_search.csv                 # real drug-like ligands? diverse proteins?
cofoldbench fetch data/candidates_search.csv
```

Delete rows you do not want before fetching: a ligand that is really a buffer
molecule or a fragment, a covalent ligand, or a pocket that needs a metal or
cofactor (`fetch` warns about the last two). The reasons behind the curated
set are recorded as comments in `data/candidates_recent.csv`.

> 📋 **Checkpoint 4:** paste both outputs. Then save the target set:
>
> ```bash
> git add data/targets.csv data/structures data/candidates_recent.csv
> git commit -m "Target set: classic + time-split complexes"
> git push
> ```
>
> (Pushing needs a GitHub token or SSH key on this PC; no sudo needed.)

### B3. Run everything

```bash
conda activate cofoldbench && cofoldbench inputs && cofoldbench dockprep
conda activate boltz
nohup sh -c 'scripts/run_boltz.sh blind && scripts/run_boltz.sh pocket' > logs/boltz.log 2>&1 &
```

Boltz and Chai share the one GPU, so run them **one after the other**, not in
parallel. Wait for Boltz to finish (`tail -f logs/boltz.log`), then:

```bash
conda activate chai
nohup scripts/run_chai.sh > logs/chai.log 2>&1 &
```

Vina is CPU-only, so it can run while a model has the GPU. **GNINA cannot**:
it scores poses with a CNN on the GPU and will abort with out-of-memory if
Boltz or Chai is using the card. So run Vina now and GNINA afterwards:

```bash
conda activate cofoldbench
nohup scripts/run_docking.sh vina > logs/vina.log 2>&1 &
```

Once the GPU is free:

```bash
conda activate cofoldbench
nohup scripts/run_docking.sh gnina > logs/gnina.log 2>&1 &
```

Then:

```bash
cofoldbench evaluate
cofoldbench report          # adds "before cutoff" / "after cutoff" rows per method
cofoldbench stats           # confidence intervals and paired tests: read this before
                            # believing any difference between two methods
```

### Controls worth running before publishing

**A pre-cut-off control that is not famous.** The 23 classic targets are not
only in the training data, they are in it many times over (every benchmark,
every tutorial, redundant PDB entries). So the split confounds *seen in
training* with *heavily represented*. An obscure pre-cut-off set separates
them: if the co-folding models score as well on those as on the famous ones,
plain exposure is enough; if they score much worse, fame is doing part of the
work.

```bash
cofoldbench fetch data/candidates_control.csv   # the curated control set in this repo
```

To build your own instead:

```bash
cofoldbench search --after 2015-01-01 --before 2023-06-01 --n 40 \
    --out data/candidates_search_old.csv   # one random sample per year across the window
```

Ask for more candidates than you need: the metal, cofactor and symmetry-mate
warnings will remove several. Drop every target that gets one, for the same
reasons as in the post-cut-off set, or the control will not be comparable.

`--before` matters: without it the search returns the newest entries whatever
`--after` says, because results come back newest-first.

**A second docking seed**, to see how much of a method-to-method difference is
just run-to-run noise:

```bash
SEED=2 METHOD_SUFFIX=_s2 scripts/run_docking.sh
cofoldbench evaluate && cofoldbench report     # gnina vs gnina_s2, vina vs vina_s2
```

### Targets that do not fit the GPU

If a target produces no Boltz output even after the retry pass, you have two
honest options, and they answer different questions:

* **Drop it from `data/targets.csv`** (recommended) and say so in the write-up.
  The benchmark then compares model *quality* on targets every method could
  attempt. Re-run `evaluate`/`report` afterwards.
* **Leave it in.** It counts as a failure for the methods that produced
  nothing, and `report` prints `n_missing` per method. This answers "what do I
  actually get on a 12 GB GPU", which is a hardware result, not a model result.

Chai-1 may also have needed **reduced sampling** (fewer trunk recycles and
diffusion steps) to fit a target on the GPU. Those targets are listed in
`runs/chai1/_reduced_settings.txt`; say so when reporting Chai results, since
they did not get the same sampling budget as the rest.

With `--exclude` you can report both views without editing `data/targets.csv`:

```bash
cofoldbench report --outdir results/report                      # all targets; n_missing shows the gaps
cofoldbench report --exclude 1EVE,1VRT,3LN1 --outdir results/report_fits   # level field
```

What you must not do is let the missing targets vanish: scoring each method
only over the targets it managed would reward failing on the hard ones.
`cofoldbench report` scores every method over the same target set and warns
about missing poses for exactly this reason.

> 📋 **Checkpoint 5:** paste `results/report/summary.md`, and push the results:
>
> ```bash
> git add -f results/results.csv results/report
> git commit -m "Benchmark results" && git push
> ```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `NVIDIA driver on your system is too old` / `cuda_available=False` | `scripts/fix_torch_cuda.sh` inside that env |
| `WARNING: ran out of memory, skipping batch` (Boltz) | Boltz skips that target and carries on, producing nothing for it. The script retries each missing target alone with a shallower MSA and lists any that still fail. A target that never fits should be **dropped from `data/targets.csv` for every method** (see "Targets that do not fit" below) rather than left failing for Boltz only. |
| `CUDA out of memory` (Chai) | Chai-1 needs more memory than Boltz-2. The script retries the target with fewer trunk recycles and diffusion steps and records it in `runs/chai1/_reduced_settings.txt`; mention those targets when reporting Chai results. One failure no longer ends the run. |
| `No module named 'cuequivariance_torch'` | `scripts/run_boltz.sh` already passes `--no_kernels`; keep `KERNELS=0` |
| `cofoldbench not found` inside the boltz/chai env | the scripts look in `~/.conda/envs`, `~/miniforge3/envs`, `~/miniconda3/envs`; otherwise set `CFB_ENV` or run the `collect` line they print from the cofoldbench env |
| GNINA aborts / `CUDA out of memory` | GNINA uses the GPU for CNN scoring. Wait until Boltz and Chai are done, or run `GNINA_ARGS=--no_gpu scripts/run_docking.sh gnina`. Failures are listed at the end of the run and re-running retries only those targets. |
| GNINA will not start at all (library error) | another binary from the GNINA releases page via `GNINA_URL=... scripts/install_docking_tools.sh` |
| Chai: `cannot write parquet` / `Unable to find a usable engine` | Chai needs a parquet engine for its MSAs. `pip install -U pyarrow` can pull a build that needs NumPy 2 while chai_lab pins NumPy 1 (`pyarrow requires NumPy 2.0 or newer`); then install a NumPy-1 build instead: `pip install "pyarrow<21"`, or `pip install fastparquet`. Do not upgrade NumPy in this env. Rerun after fixing (the script clears the half-written target folder). |
| MSA server errors / timeouts | rerun later (the server is shared and rate-limited); finished targets are skipped |
| `fetch`: `ligand check failed` | the crystal ligand has unmodelled atoms or is covalently modified; leave the target out |
| `fetch`: `SITE SPANS A SYMMETRY MATE` | the pocket is at an oligomeric interface whose partner is a crystallographic neighbour, so the deposited chain holds only half the site and **every** method fails. Add the partner chain to the `chain` column if the model contains it, otherwise drop the target. |
| `Element 'G' not found` when collecting Vina poses | Meeko opens macrocyclic ligands and adds `CG`/`G` pseudo-atoms. Fixed in `structure.ad_element`; update the repo and re-collect with `cofoldbench collect vina --target <ID> --receptor inputs/dock/<ID>/receptor.pdb --poses runs/vina/<ID>.pdbqt`. |

For anything else: `scripts/check_env.sh` plus the last 30 lines of the log.
