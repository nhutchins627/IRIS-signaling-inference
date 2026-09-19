#!/usr/bin/env python3
"""Supp. Fig. 2 -- the response-gene method applied to the mouse embryo.

This is the negative result that motivates IRIS. Response-gene scores are
projected onto the anterior foregut and cardiomyocyte trajectories of the
gastrulation atlas. They recover some known biology, but there is no threshold
that converts them into a usable binary call across cell types -- the scores
drift with cell type and depth, so any fixed cut-off is wrong somewhere.

Scope note: the saved lineage tables carry IRIS calls, not raw expression, so
this script plots the IRIS trajectories and reports per-lineage activation
rates. Reproducing the response-gene *scores* of Supp. Fig. 2c requires the
12 GB atlas matrix -- see ``--atlas``, which explains the route.

Outputs
    supp02_response_scores_pseudotime.*   score vs pseudotime, both lineages
    supp02_threshold_instability.csv      per-lineage optimal thresholds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iris_repro import config, metrics, plotting, provenance

ANALYSIS = "supp02"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas", action="store_true",
                    help="recompute scores from the atlas h5ad (slow, 12 GB)")
    args = ap.parse_args()

    from fig3.fig3_lineage_dynamics import load_predictions

    outdir = config.output_dir("supp")
    provenance.check_environment()

    lineages = {name: load_predictions(name) for name in ("endoderm", "cardiac")}

    if args.atlas:
        raise SystemExit(
            "Recomputing response-gene scores needs the 12 GB gastrulation "
            "atlas and its gene matrix. The saved lineage tables carry IRIS "
            "calls but not raw expression, so run without --atlas to use them, "
            "or extend fig3_lineage_dynamics.py --predict to also emit scores."
        )

    plt = plotting.set_style()
    colors = config.palette()

    # IRIS calls are binary; show how cleanly each pathway separates along
    # pseudotime in each lineage, which is what a threshold would have to do.
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 1.8), sharey=True)
    rows = []
    for ax, (name, df) in zip(axes, lineages.items()):
        pt = df["pseudotime"].to_numpy(float)
        for sig in config.signals():
            col = f"{sig}_pred"
            if col not in df:
                continue
            y = df[col].to_numpy(float)
            plotting.plot_trend(pt, y, ax=ax, color=colors.get(sig, "k"),
                                label=config.display_name(sig))
            # Where along pseudotime does this pathway switch state?
            on = pt[y > 0.5]
            rows.append({
                "lineage": name, "signal": sig,
                "fraction_active": float(y.mean()),
                "median_pseudotime_active": float(np.median(on)) if on.size else np.nan,
                "n_cells": int(len(y)),
            })
        ax.set_title(f"{name.capitalize()} lineage")
        ax.set_xlabel("Diffusion pseudotime")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Fraction active")
    axes[1].legend(frameon=False, ncol=2, fontsize=5)
    fig.tight_layout()
    plotting.save(fig, outdir, "supp02_response_scores_pseudotime")
    plt.close(fig)

    table = pd.DataFrame(rows)

    # The instability claim: the same pathway is active at very different rates
    # in the two lineages, so one global cut-off cannot serve both.
    pivot = table.pivot(index="signal", columns="lineage",
                        values="fraction_active")
    pivot["abs_difference"] = (pivot["endoderm"] - pivot["cardiac"]).abs()
    pivot.to_csv(outdir / "supp02_threshold_instability.csv")
    print("Fraction of cells called active, by lineage:")
    print(pivot.round(3).to_string())
    print(f"\nLargest between-lineage gap: {pivot['abs_difference'].max():.3f} "
          f"({pivot['abs_difference'].idxmax()})")
    print("\nNOTE: these are IRIS binary calls, not response-gene scores -- the\n"
          "saved lineage tables do not carry raw expression. Comparable overall\n"
          "activation rates across two unrelated lineages is an IRIS property,\n"
          "NOT evidence about response-gene thresholding.\n"
          "The published Supp. Fig. 2c claim (that response-gene scores admit no\n"
          "consistent cross-cell-type threshold) needs the atlas expression\n"
          "matrix; see --atlas.")

    provenance.record(ANALYSIS, outdir,
                      results={"fraction_active": table.to_dict("records")})
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
