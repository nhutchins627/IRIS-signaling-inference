# IRIS — reproducibility package

Reproduces the analyses in *Reconstructing signaling histories of single cells
via perturbation screens and transfer learning* (Hutchins et al.), organized by
figure number.

IRIS (Intracellular Response to Infer Signaling State) is a CVAE/scANVI-based
semi-supervised classifier. It learns transcriptome-wide signal response
signatures from in vitro perturbation screens and transfers them to in vivo
cell types it has never seen.

---

## Quick start

```bash
# 1. Environment matching the published figures (see requirements.txt).
#    scvi-tools 0.20.3 is required -- 1.1.x changes SCANVI defaults.
conda create -n iris-repro python=3.9 && conda activate iris-repro
pip install -r requirements.txt

# 2. Point the config at your data (see config/paths.yaml, or):
export IRIS_WORK_DIR=/path/to/working
export IRIS_ARCHIVE_DIR=/path/to/archive

# 3. Check what resolves before running anything
PYTHONPATH=src python scripts/check_setup.py

# 4. Rebuild every figure that does not need a GPU
PYTHONPATH=src python scripts/run_all.py

# 5. See the full step list, including the slow ones
PYTHONPATH=src python scripts/run_all.py --list
```

> **Data is not bundled.** Obtain it from the accessions in
> [`docs/DATA.md`](docs/DATA.md) (GSE289836, GSE122009, GSE136689, GSE246368,
> E-MTAB-6967) and point `config/paths.yaml` at your copy.

Outputs (SVG + PNG + CSV + provenance JSON) land in `outputs/fig<N>/`.

For interactive work, start Jupyter in this environment and open
`notebooks/00_overview.ipynb`.

---

## Layout

```
config/         paths.yaml, model_params.yaml   <- edit these, not the code
src/iris_repro/ shared library (config, data, model, metrics, plotting, provenance)
figures/fig<N>/ one script per panel; each is runnable on its own
notebooks/      interactive counterparts of the same analyses
scripts/        run_all.py orchestrator
outputs/        generated figures and tables (created on first run)
docs/           DATA.md, METHODS.md, GITHUB_UPDATES.md
```

`figures/*.py` and `notebooks/*.ipynb` call the same library functions, so the
interactive and batch paths cannot drift apart.

---

## Two speeds

Most panels rebuild from **saved predictions** — the per-cell CSVs the original
runners wrote. These need no GPU and take seconds.

Steps that **refit models** are opt-in:

| step | flag | cost |
|---|---|---|
| Fig 1c IRIS predictions | `--retrain` | GPU, ~20 min |
| Fig 2e cross-species baselines | `--compute` | CPU, ~1 h |
| Fig 3 atlas inference | `--predict` | GPU, ~1 h |
| Fig 4a mesenchyme inference | `--predict` | GPU, ~30 min |

Run them once; results are cached and reused. Caches live in `cache/`, which
sits **outside** `outputs/` on purpose — clearing regenerated figures should
never destroy an hour of compute.

`run_all.py` reports these as `SKIP`, not `FAIL`, when their cache is absent:
each step still writes everything it can (Fig. 1c, for instance, emits the
response-gene baseline on its own) and prints the exact command to run next.
A non-zero exit means something genuinely broke.

---

## Figure map

Named by the **submitted** manuscript (Fig. 1-4 + Supp. 1-21). The archive
uses an older six-figure scheme -- see [`docs/FIGURE_MAPPING.md`](docs/FIGURE_MAPPING.md)
for the translation (`fig5v2` = Fig. 3, `fig6v2` = Fig. 4).

### Main figures

| Panel | Script | Reproduces |
|---|---|---|
| Fig 1c | `fig1/fig1c_insample_accuracy.py` | IRIS vs the response-gene baseline |
| Fig 2c | `fig2/fig2cd_generalization.py` | leave-one-screen-out cross-validation |
| Fig 2d | `fig2/fig2cd_generalization.py --panel 2d` | endoderm to mesoderm transfer |
| Fig 2e | `fig2/fig2e_model_benchmarking.py --regime cross-species` | IRIS vs EN / SVM / RF |
| Fig 2f | `fig2/fig2f_gene_ablation.py` | thousands of genes carry the signature |
| Fig 3d-f | `fig3/fig3_lineage_dynamics.py` | signaling histories along two lineages |
| Fig 4a | `fig4/fig4a_mesenchyme_enrichment.py` | WNT/BMP in respiratory mesenchyme |

### Supplementary figures

Coverage is **partial and explicitly mapped** -- run this for the live status:

```bash
PYTHONPATH=src python figures/supp/supp_coverage.py        # all 21
PYTHONPATH=src python figures/supp/supp_coverage.py --gaps # only what is missing
```

As of the last run, **16 of 21** run end to end, 2 more are one
rebuilt input away, and 3 are genuinely out of reach:

| Route | Count | Supp. figures |
|---|---|---|
| Script in this package | 9 | 1, 3, 4, 6, 7, 8, 10, 12, 19 |
| Wrapped source-data script | 7 | 13, 14, 15, 16, 17, 18, 20 |
| Script present, one input to rebuild | 2 | 2 (needs the atlas matrix), 11 (needs `hM_d4_step_labels.csv`) |
| Inputs present, no script yet | 1 | 5 (hyperparameter sweep; 6,729 archived model dirs) |
| Needs a model refit | 1 | 9 (train with/without hM_d4) |
| Imaging, no computational path | 1 | 21 (HCR-FISH microscopy) |

The seven "wrapped" ones reuse the numbered scripts in
`in_vivo_results/source_data_scripts/`, which produced the published source
data. They are not reimplemented here -- a second implementation would drift:

```bash
PYTHONPATH=src python figures/supp/supp_source_data.py --list      # script -> figure
PYTHONPATH=src python figures/supp/supp_source_data.py --check     # what exists
PYTHONPATH=src python figures/supp/supp_source_data.py --run 11    # run one
```

---

## Configuration

Everything that touches the filesystem resolves through `config/paths.yaml`.
To move the analysis, edit that file or set environment variables:

```bash
export IRIS_WORK_DIR=/path/to/working
export IRIS_ARCHIVE_DIR=/path/to/archive
export IRIS_OUTPUT_DIR=/path/to/outputs
```

`config/model_params.yaml` holds the per-pathway architectures, the batch →
screen mapping, the data splits, and the response-gene panel. These values are
cross-checked against three independent sources (the runner scripts, the
on-disk checkpoint names, and Supp. Fig. 5c) and agree.

### Legacy paths

The original analysis ran from `/lab/tambora_li/Nicholas/figures-paper/`, which
has since been archived to `/archive/li/Nicholas-IRIS-figures-paper-archive/`.
Scripts in the working directory still contain the old absolute paths and will
fail if run as-is. `iris_repro.config.resolve()` rewrites them transparently;
the mapping lives under `legacy_rewrites:` in `paths.yaml`.

---

## Environment

Matches the Methods:

| package | version |
|---|---|
| scvi-tools | 0.20.3 |
| scanpy | 1.10.0 |
| anndata | 0.10.6 |
| torch | 2.2.2+cu121 |
| scikit-learn | 1.4.1.post1 |

`provenance.check_environment()` warns on any mismatch at the start of every
run. Every script also writes a `<analysis>.provenance.json` next to its
outputs recording package versions, git commit, host, input fingerprints and
parameters — so a regenerated panel stays auditable.

GPU jobs on this cluster must go through Slurm; see `docs/METHODS.md`.

---

## Reproducibility caveats

* **GPU nondeterminism.** Seeds are fixed, but SCANVI training is not
  bit-for-bit reproducible on GPU. The paper reports variance over 10 random
  initialisations (Supp. Fig. 10c); expect small numeric drift on refit.
* **Ablation summary.** The genes-to-50% table is read off an exponential fit
  to the noisy raw curve (`--half-method fit`, as published). Raw
  interpolation (`--half-method interp`) gives systematically smaller numbers.
* **Batch 2 ("mixed")** is excluded from cross-validation because it overlaps
  mP_d1 and mE_d2.
