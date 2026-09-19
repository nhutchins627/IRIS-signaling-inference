# Methods notes

Implementation details needed to reproduce the numbers, condensed from the
paper's Methods and the original scripts.

## The IRIS model

A two-stage semi-supervised pipeline built on scANVI (scvi-tools 0.20.3):

1. **SCVI** fits a batch-corrected latent space over *all* cells — held-out
   ones included — with no access to signaling labels. ZINB likelihood,
   40 epochs.
2. **SCANVI** initialises from that model and jointly refines the latent space
   together with a feed-forward classifier, 5 epochs. Only training cells
   carry labels; held-out cells get the sentinel `unknown`.

Because held-out cells are never labelled, the held-out screen is a genuine
test even though its expression contributed to the latent space.

`setup_anndata` uses `layer="counts"`, `batch_key="batch"` and categorical
covariates `[batch, celltype, species]`.

### Per-pathway architecture

Chosen by the cross-batch hyperparameter screen (Supp. Fig. 5): the mESC data
splits into 3 collection batches, two train and one held out, scored by AUPRC
averaged over all three iterations. The sweep covered layers 1–3, hidden
2^n (n=5..10), latent 10n (n=1,3,5,7,9). **Shallow (1-layer) architectures won
for every pathway.**

| pathway | hidden | latent |
|---|---|---|
| WNT | 64 | 30 |
| TGF-β | 1024 | 30 |
| FGF | 128 | 70 |
| RA | 256 | 70 |
| BMP | 256 | 70 |
| SHH | 256 | 30 |

SHH is excluded from the main benchmarks — too few cells were SHH-stimulated
in the mESC screen (772 of 16,529) for statistical power.

Pathways are modelled **independently**; a cell's combinatorial code is the
concatenation of the per-pathway calls, in the order TGF-β, WNT, FGF, BMP, RA.

### Thresholding

Predicted probabilities pile up near 0 and 1 (Supp. Fig. 3), plausibly because
the screens used saturating ligand concentrations. A fixed 0.5 threshold is
used, chosen to stay agnostic to which kind of downstream error matters.

## The response-gene baseline

Score = summed log-normalised expression of a pathway's annotated response
genes (unit weights), modified from Han et al. 2020. Genes are stored as
**human** symbols; mouse data is converted with `mousipy` before analysis.

Turning that score into a call needs a threshold, which is picked per pathway
as the point maximising TPR − FPR on the ROC (Youden's J). That there is no
shared cross-cell-type cut-off is precisely the paper's objection to the method.

## Evaluation

* **AUROC / AUPRC / F1** from scikit-learn; F1 at the Youden threshold.
* **Random baseline F1** is the always-predict-positive rule: precision = the
  prevalence *p*, recall = 1, so F1 = 2p/(1+p). This is the dashed reference in
  Fig. 2c/2d.
* Metrics are computed on **held-out rows only**.

### Statistical tests, by figure

| test | where |
|---|---|
| two-tailed Spearman | pseudotime/pseudospace trends (Fig. 3d,h,i) |
| one-tailed Mann-Whitney (`greater`) | combination ordering (Fig. 3e,f), marker enrichment |
| one-sided Fisher's exact | mesenchyme enrichment (Fig. 4a) |
| one-sided binomial | IRIS vs each competitor (Supp. Fig. 10b,c) |
| one-sided Student's t | HCR-FISH quantification (Fig. 4e) |
| bootstrap (n=1000) | FPR confidence intervals (Supp. Fig. 11) |

## Pseudotime and pseudospace

Diffusion maps (`scanpy.tl.diffmap`) with a Gaussian kernel; components are
chosen as the pair that orders known cell-type markers correctly. Trajectories
are discretised into **35 bins** for plotting (or smoothed with bandwidth 0.1).

For somitic mesoderm, pseudotime doubles as a **pseudospatial** axis from
posterior to anterior, since cells differentiate as they move anteriorly.

## Gene ablation

Genes are ranked by mutual information, dispersion, or mean expression, then
progressively destroyed by **resampling each gene's values with replacement
across cells**. This preserves the marginal distribution while severing the
gene-to-label relationship, so the AUROC drop isolates that gene's contribution.

Ablation proceeds in blocks of **73 genes** (`variable_genes[73*i : 73*j]` in
the runners), so the trailing index in each result filename counts *blocks*,
not genes. Curves are fit to `exp(-a·x)` and min-max normalised; the published
genes-to-50% table is read off the fit as `ln(2)/a`.

## Cluster execution

GPU work must go through Slurm — never the head node.

```bash
srun --partition=nvidia-t4-20 --gres=gpu:1 --mem=200GB \
     --cpus-per-task=8 --time=8:00:00 --pty bash
```

Check partitions and limits first:

```bash
curl -s http://slurmstatus.wi.mit.edu/limits.html
```

CPU-only steps (the fast path, sklearn benchmarks) run fine in a modest
interactive session:

```bash
srun --partition=20 --cpus-per-task=4 --mem=32G --time=2:00:00 --pty bash
```

Namespace job names and temp files with `$USER`, per cluster policy.

## Known sources of drift

* SCANVI training is not bit-for-bit reproducible on GPU even with fixed
  seeds; the paper reports variance over 10 random initialisations.
* The ablation summary depends on the exponential fit, which is sensitive to
  how noisy the tail of a curve is.
* Diffusion-component choice is a manual step in the original analysis
  ("visually inspected"); the scripts here fix DC1 as the ordering axis for the
  saved lineage tables.
