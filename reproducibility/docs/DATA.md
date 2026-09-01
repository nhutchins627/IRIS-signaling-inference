# Datasets

Every path below resolves through `config/paths.yaml`. Sizes are approximate.

## Training data

### `screens_ref_full.h5ad` (1.7 GB, 38,307 × 13,222)
The combined mESC + hESC perturbation reference — the object every model
trains on. Raw counts in `.X`.

`obs` columns: `batch`, `celltype`, `species`, and one `{Signal}_class` per
pathway with values `Stim` / `Ctrl`.

| batch | screen | species | lineage | cells | description |
|---|---|---|---|---|---|
| 1 | mP_d1 | mouse | pluripotent | 4,751 | mESC 24 h exit from pluripotency |
| 2 | mixed | mouse | mixed | 175 | mixed collection batch; **excluded from CV** |
| 3 | mE_d2 | mouse | endoderm | 5,224 | anterior-PS-derived, endoderm-biased |
| 5 | hM_d4 | human | mesoderm | 6,379 | hESC mesoderm d4, 2-step sequential screen |
| 6 | hE_d8 | human | endoderm | 9,921 | hESC endoderm d8, full 2^6 combinations |
| 7 | hM_d7 | human | mesoderm | 11,857 | hESC mesoderm d7, 4 subtypes × 2^4 |

Batch 2 is excluded from cross-validation because it overlaps mP_d1 and mE_d2.
Mouse data (batches 1–3) is from Yeo et al. 2020 (GSE122009); the three human
screens are new (GSE289836).

## In vivo prediction targets

### `fig5v2_data.h5ad` (12 GB, 170,265 × 13,286)
Pijuan-Sala E6.5–E8.5 mouse gastrulation atlas (E-MTAB-6967) concatenated with
the screen reference. ~139,331 atlas cells, matching the n in Supp. Fig. 14b.
Used for Fig. 3 and Supp. Figs. 14–18.

### `detailed_celltypes_augmented_lung_intestine_mesenchyme_screens.h5ad` (Fig. 4a)
Bundles several atlases (731,856 cells, 246 cell-type labels). Fig. 4a uses
only the four divergent E9–E9.5 foregut mesenchymal fates, which
`_canonical()` in `figures/fig4/` collapses out of the fine-grained labels:

| fate | cells | source labels merged |
|---|---|---|
| respiratory | 1,122 | `respiratory`, `respiratory-lung`, `respiratory-trachea` |
| esophageal | 794 | `esophagus`, `esophagus-1`, `esophagus-2` |
| pharyngeal | 705 | `pharynx`, `pharyngeal 4`, … |
| dorsal lateral foregut | 429 | `dorsal lateral foregut`, `dorsal-lateral foregut` |

Everything else (other atlases in the same file) is dropped, so the Fisher test
compares respiratory against *other foregut mesenchyme*, not against every cell
in the object.

### `human_airway_epithelium.h5ad` (1.0 GB)
McCauley et al. 2024 adult human airway epithelium (GSE246368), stimulated for
7 days with BMP4, FGF2/FGF10, TGF-β2/Activin A, or CHIR. Supp. Fig. 19.

### `miram_lung.h5ad` (0.5 GB, 12,044 cells)
E10–E11.5 mouse lung, Seurat-clustered, used by `miram_lung_rerun.py`. Note
this is **not** the E9–E9.5 foregut of Fig. 4a — it carries no mesenchymal
fate labels.

## Accessions

| dataset | accession |
|---|---|
| hM_d4, hM_d7, hE_d8 (this study) | GSE289836 |
| mESC screen (Yeo et al. 2020) | GSE122009 |
| mouse foregut organogenesis (Han et al. 2020) | GSE136689 |
| human airway epithelium (McCauley et al. 2024) | GSE246368 |
| mouse gastrulation atlas (Pijuan-Sala et al. 2019) | E-MTAB-6967 |

## Derived results consumed by the figures

| what | where | used by |
|---|---|---|
| per-cell predictions | `clean_splits/`, `clean_splits2/`, `cross_species_*.csv` | Fig. 2c–e |
| gene ablation | `ablation_results/` (27,301 CSVs) | Fig. 2f |
| in vivo diffusion map + codes | `in_vivo_results/diffmap_with_labels.csv` | Fig. 3 |
| sklearn benchmarks | `benchmark_batch<N>_*.csv` | Supp. Fig. 10c |
| trained checkpoints | `zorn_*_rs/`, `miram_out_*_model/` | inference |
| source-data workbooks | `in_vivo_results/source_data_scripts/*.xlsx` | most supplementaries |

### Source-data workbooks

Twelve numbered scripts in `in_vivo_results/source_data_scripts/` produced the
source data behind the published supplementary panels, each writing an `.xlsx`.
They are **wrapped, not reimplemented** (`figures/supp/supp_source_data.py`) --
a parallel implementation would drift from the published numbers.

`12_ablation_expfit.py` is the clearest example: it fits `c + A*exp(-k*x)` with
a floor `c`, because ablating every gene leaves the model at chance rather than
at zero. Its `n50` values are in 73-gene blocks and reproduce the Fig. 2f inset
table almost exactly (e.g. WNT/mE_d2 = 926 genes vs 949 published). Fig. 2f in
this package reads that workbook by default; `--recompute` refits locally.

### Prediction CSV schema

Two generations exist. Both carry `scanvi_pred_Stim` (probability) and
`ground_truth`; **only the newer ones carry a `batch` column**. For the older
files the loader infers held-out rows from the trailing block, because the
runner concatenated train-then-test before saving. See `_held_out_mask()` in
`figures/fig2/fig2cd_generalization.py`.

Filenames encode the split: `..._<Signal>_in_<batches>_out_<batches>_results.csv`.
Panels are selected on that split identity, not on the filename prefix, because
`clean_splits2/` mixes several experiments in one directory.
