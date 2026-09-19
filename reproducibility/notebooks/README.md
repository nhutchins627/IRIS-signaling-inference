# Notebooks

Interactive counterparts of the scripts in `figures/`. Both call the same
`iris_repro` functions, so the two paths cannot drift apart — the scripts are
the authoritative, non-interactive route; these are for exploring.

Run order:

| notebook | figure | needs |
|---|---|---|
| `00_overview.ipynb` | — | nothing; verifies setup and summarises the atlas |
| `01_fig1_response_genes.ipynb` | Fig. 1c, Supp. 1 | screens h5ad (GPU only for the IRIS comparison) |
| `02_fig2_generalization.ipynb` | Fig. 2c–f | saved prediction + ablation CSVs |
| `03_fig3_in_vivo_lineages.ipynb` | Fig. 3d–f | `in_vivo_results/diffmap_with_labels.csv` |
| `04_fig4_mesenchyme.ipynb` | Fig. 4a | foregut predictions (GPU to generate) |

## Kernel

Use the environment that matches the published figures:

```
/lab/li_lab/Nicholas_keep/miniconda3/envs/scvi-env-new/bin/python
```

Register it once:

```bash
/lab/li_lab/Nicholas_keep/miniconda3/envs/scvi-env-new/bin/python \
  -m ipykernel install --user --name scvi-env-new --display-name "IRIS (scvi-env-new)"
```

Each notebook's first cell walks up from the working directory to find `src/`,
so you can start Jupyter from anywhere in the package.

GPU work must go through Slurm — never the head node:

```bash
srun --partition=nvidia-t4-20 --gres=gpu:1 --mem=200GB --time=8:00:00 --pty bash
```
