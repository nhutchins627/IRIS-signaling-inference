#!/usr/bin/env python3
"""Supp. Fig. 19 -- IRIS predictions on adult human airway epithelium.

A genuine out-of-domain test: organotypic cultures of primary adult human
bronchial epithelium (McCauley et al. 2024), stimulated for 7 days. These
cultures contain several cell types with differing competence to respond, and
nothing like them appears in the training screens.

Competence is approximated by expression of the pathway's receptors, so the
question is whether IRIS-positive cells are enriched among receptor-expressing
cells. Two-tailed Fisher's exact test on the 2x2 table.

CHIR is the informative exception: it activates WNT downstream of the receptor,
so receptor expression should *not* predict response -- and it doesn't
(p = 0.1), which is a control rather than a failure.

Caveat carried from the paper: receptor dropout in scRNAseq makes the
"competent" set an underestimate, biasing this test conservative.

Outputs
    supp19_receptor_enrichment.*      odds ratio and -log10(p) per pathway
    supp19_receptor_enrichment.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from iris_repro import config, metrics, plotting, provenance

ANALYSIS = "supp19"

# Ligands used per pathway in the source study; ActA/TGFB1/Fgf2/Fgf10 are the
# individual-ligand arms reported alongside the pooled pathway rows.
POOLED = ["Wnt", "Bmp", "Fgf", "TgfB"]


def load_summary() -> pd.DataFrame:
    p = Path(config.results_dir("receptor_tests")) / "receptor_prediction_tests_summary.csv"
    if not p.exists():
        raise SystemExit(
            f"Missing {p}.\n"
            "Produced by the receptor-prediction test over the McCauley et al. "
            "airway dataset (config key 'airway_epithelium')."
        )
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-ligands", action="store_true",
                    help="also show the individual-ligand arms")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute Fisher tests from the contingency counts")
    args = ap.parse_args()

    outdir = config.output_dir("supp")
    provenance.check_environment()

    df = load_summary()
    if not args.all_ligands:
        df = df[df["pathway"].isin(POOLED)]

    # The saved table already carries fisher_or/fisher_p; recomputing from the
    # counts is a cheap check that the two agree.
    if args.recompute:
        rows = []
        for _, r in df.iterrows():
            # ct<receptor><prediction>: ct11 = receptor+ and predicted active
            table = [[int(r["ct11"]), int(r["ct10"])],
                     [int(r["ct01"]), int(r["ct00"])]]
            rows.append(metrics.fisher_enrichment(table, alternative="two-sided"))
        recomputed = pd.DataFrame(rows, index=df.index)
        df = df.assign(recomputed_or=recomputed["odds_ratio"],
                       recomputed_p=recomputed["p"])
        delta = (df["recomputed_p"] - df["fisher_p"]).abs().max()
        print(f"  max |p_recomputed - p_saved| = {delta:.2e}")

    df = df.assign(significant=df["fisher_p"] < 0.05)
    df.to_csv(outdir / "supp19_receptor_enrichment.csv", index=False)
    cols = ["pathway", "n_cells_stim", "fisher_or", "fisher_p", "significant"]
    print(df[cols].to_string(index=False))
    print("\nWNT is expected to be non-significant: CHIR bypasses the receptor.")

    plt = plotting.set_style()
    colors = config.palette()
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 1.7))
    x = np.arange(len(df))
    cols_ = [colors.get(p, "#888888") for p in df["pathway"]]

    axes[0].bar(x, df["fisher_or"], color=cols_, edgecolor="none")
    axes[0].axhline(1.0, ls="--", lw=0.6, c="grey")
    axes[0].set_ylabel("Odds ratio")

    axes[1].bar(x, [-np.log10(max(p, 1e-300)) for p in df["fisher_p"]],
                color=cols_, edgecolor="none")
    axes[1].axhline(-np.log10(0.05), ls="--", lw=0.6, c="grey")
    axes[1].set_ylabel("-log10(p)")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([config.display_name(p) for p in df["pathway"]],
                           rotation=45, ha="right")
    fig.suptitle("IRIS+ cells vs receptor expression (airway epithelium)",
                 fontsize=8)
    fig.tight_layout()
    plotting.save(fig, outdir, "supp19_receptor_enrichment")
    plt.close(fig)

    provenance.record(ANALYSIS, outdir,
                      inputs=[config.data_path("airway_epithelium", require=False)],
                      results={"enrichment": df[cols].to_dict("records")})
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
