#!/usr/bin/env python3
"""Supp. Fig. 4 -- cell-state diversity within and across the screens.

Two questions the screens have to answer before any generalization claim is
credible: did the perturbations actually produce distinct cell states, and are
the screens distinct from one another?

Diversity is measured as the transcriptome-wide Pearson correlation between
the mean profiles of distinct signal conditions -- the same quantity behind
Fig. 2b's hierarchical clustering.

Outputs
    supp04_condition_correlation_<screen>.*   within-screen condition similarity
    supp04_screen_summary.csv                 cells and conditions per screen
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from iris_repro import config, plotting, provenance
from iris_repro.data import combination_code, load_screens

ANALYSIS = "supp04"


def condition_profiles(adata, min_cells: int = 10) -> pd.DataFrame:
    """Mean log-normalised expression per signal combination."""
    import scanpy as sc

    sub = adata.copy()
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)

    codes = combination_code(sub)
    keep = codes.value_counts()
    keep = keep[keep >= min_cells].index
    if len(keep) < 2:
        return pd.DataFrame()

    X = sub.X
    dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    rows = {c: dense[(codes == c).to_numpy()].mean(axis=0) for c in keep}
    return pd.DataFrame(rows, index=sub.var_names)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-cells", type=int, default=10,
                    help="minimum cells for a condition to be included")
    args = ap.parse_args()

    outdir = config.output_dir("supp")
    provenance.check_environment()

    adata = load_screens()
    plt = plotting.set_style()
    summary = []

    for batch, meta in sorted(config.batch_table().items()):
        sub = adata[adata.obs["batch"].values == batch]
        if sub.n_obs == 0:
            continue
        name = meta["name"]
        profiles = condition_profiles(sub, args.min_cells)
        if profiles.empty:
            print(f"  {name}: too few conditions with >= {args.min_cells} cells")
            summary.append({"screen": name, "n_cells": int(sub.n_obs),
                            "n_conditions": 0, "mean_correlation": np.nan})
            continue

        corr = profiles.corr(method="pearson")
        iu = np.triu_indices_from(corr.values, k=1)
        offdiag = corr.values[iu]
        summary.append({"screen": name, "n_cells": int(sub.n_obs),
                        "n_conditions": corr.shape[0],
                        "mean_correlation": float(np.nanmean(offdiag)),
                        "min_correlation": float(np.nanmin(offdiag))})
        print(f"  {name}: {sub.n_obs:6,d} cells, {corr.shape[0]:3d} conditions, "
              f"mean r = {np.nanmean(offdiag):.3f}")

        fig, ax = plt.subplots(figsize=(2.4, 2.1))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{name}: condition similarity")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
        plotting.save(fig, outdir, f"supp04_condition_correlation_{name}")
        plt.close(fig)
        corr.to_csv(outdir / f"supp04_condition_correlation_{name}.csv")

    table = pd.DataFrame(summary)
    table.to_csv(outdir / "supp04_screen_summary.csv", index=False)
    print("\n" + table.to_string(index=False))

    provenance.record(ANALYSIS, outdir,
                      inputs=[config.data_path("screens_ref_full")],
                      params={"min_cells": args.min_cells},
                      results={"summary": summary})
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
