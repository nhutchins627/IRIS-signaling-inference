# Figure numbering: archive → submitted manuscript

The analysis predates the current figure order. Archive directories use a
**six-figure** scheme (`fig1`–`fig6`, with `v2` reruns); the submitted paper
has **four main figures plus Supplementary Figures 1–21**.

Everything in this package is named by the **submitted** scheme. This table
exists so a result can still be traced back to the directory it came from.

## Archive directory → submitted figure

| Archive dir | Contents | Submitted figure |
|---|---|---|
| `fig1/` | response-gene method, gut/spinal-cord pseudotime | Supp. 1, Supp. 2 |
| `fig2/` | early model comparisons, mESC lineage KDEs | Fig. 1c, Supp. 3 |
| `fig3/` | CellChat/CellPhoneDB comparisons, endoderm | Fig. 2 context, Discussion |
| `fig3-4-rerun/` | cross-validation and benchmarking reruns | **Fig. 2c–e**, Supp. 6–10 |
| `fig4/`, `fig4v2/` | gene ablation, generalization AUROCs | **Fig. 2f**, Supp. 12 |
| `fig5/`, `fig5v2/` | gastrulation atlas predictions | **Fig. 3**, Supp. 14–18 |
| `fig6/`, `fig6v2/` | organ-specific mesenchyme, lung | **Fig. 4**, Supp. 20–21 |
| `supp-fig1/` | hyperparameter sweep (6,729 model dirs) | **Supp. 5** |
| `fig_ex/` | Tabula Muris/Senis exploratory | not in the submitted paper |
| `new-version-tests/` | scArches / model-inheritance trials | not in the submitted paper |

Two are worth remembering because the names actively mislead:

* **`fig5v2` is now Fig. 3** (in vivo lineages) — `fig5v2_data.h5ad` is the
  gastrulation atlas object.
* **`fig6v2` is now Fig. 4** (mesenchyme) — hence
  `07_fig6v2_resp_meso_diffmap.py` produces Fig. 4a source data.

## Source-data scripts → submitted figure

`in_vivo_results/source_data_scripts/` predates the renumbering too:

| Script | Submitted figure |
|---|---|
| `01_meso_endo_diffmap_pseudotime.py` | Fig. 3c–f, Supp. 15 |
| `02_neuroectoderm_diffmap_pseudotime.py` | Supp. 18 |
| `03_nmp_diffmap_multiroot_pseudotime.py` | Supp. 18 |
| `04_somitic_mesoderm_diffmap.py` | Fig. 3g–i, Supp. 17 |
| `05_signal_combos_dotplot_stats.py` | Fig. 3b, Supp. 14 |
| `06_subset_leiden_enrichment.py` | Supp. 17d–e |
| `07_fig6v2_resp_meso_diffmap.py` | Fig. 4a, Supp. 20 |
| `08_endoderm_quant.py` | Supp. 16c–d |
| `09_cardiomyocytes_analysis_quant.py` | Supp. 16b |
| `10_cardio.py` | Fig. 3d, Supp. 16a |
| `11_saliency_maps.py` | Supp. 13 |
| `12_ablation_expfit.py` | Fig. 2f, Supp. 12 |

Run `python figures/supp/supp_source_data.py --list` for the live version of
this map, and `--check` to see which workbooks already exist.

## Naming rule

New files use submitted numbering only: `fig<N>_*` for main panels and
`supp<NN>_*` for supplementary. Archive names (`fig5v2`, `fig6v2`) appear only
in `config/paths.yaml`, where they are real directory names, and in this table.
