#!/usr/bin/env python3
"""Supp. Fig. 3 -- distribution of IRIS predicted probabilities.

Predictions pile up near 0 and 1, plausibly because the screens used
saturating ligand concentrations. That bimodality is why a fixed 0.5
threshold is defensible: the model is agnostic to which kind of downstream
error matters, and few cells sit near the cut.

Outputs
    supp03_probability_histograms.*
    supp03_threshold_sensitivity.csv   F1 across candidate thresholds
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

ANALYSIS = "supp03"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--held-out", type=int, default=None,
                    help="restrict to one held-out screen (batch code)")
    args = ap.parse_args()

    from fig2.fig2cd_generalization import collect

    outdir = config.output_dir("supp")
    provenance.check_environment()

    df = collect("clean_splits", single_holdout_only=True)
    if args.held_out is not None:
        df = df[df["held_out"] == config.batch_name(args.held_out)]
        if df.empty:
            raise SystemExit(f"No predictions for batch {args.held_out}")

    plt = plotting.set_style()
    colors = config.palette()
    signals = [s for s in config.signals() if s in set(df["signal"])]

    fig, axes = plt.subplots(1, len(signals), figsize=(1.5 * len(signals), 1.5),
                             sharey=True)
    rows = []
    for ax, sig in zip(np.atleast_1d(axes), signals):
        sub = df[df["signal"] == sig]
        ax.hist(sub["y_score"], bins=40, color=colors.get(sig, "k"),
                edgecolor="none")
        ax.axvline(0.5, ls="--", lw=0.6, c="grey")
        ax.set_title(config.display_name(sig))
        ax.set_xlim(0, 1)

        # How much would a different cut-off change F1?
        for thr in np.arange(0.1, 0.95, 0.05):
            m = metrics.classification_metrics(sub["y_true"], sub["y_score"],
                                               threshold=float(thr))
            rows.append({"signal": sig, "threshold": round(float(thr), 2),
                         "F1": m["F1"], "AUROC": m["AUROC"]})
        frac_mid = float(((sub["y_score"] > 0.3) & (sub["y_score"] < 0.7)).mean())
        print(f"  {sig:5s} n={len(sub):7,d}  fraction in [0.3, 0.7]: {frac_mid:.3f}")

    np.atleast_1d(axes)[0].set_ylabel("Count")
    fig.supxlabel("Predicted response probability", fontsize=7)
    fig.tight_layout()
    plotting.save(fig, outdir, "supp03_probability_histograms")
    plt.close(fig)

    sens = pd.DataFrame(rows)
    sens.to_csv(outdir / "supp03_threshold_sensitivity.csv", index=False)
    best = (sens.loc[sens.groupby("signal")["F1"].idxmax()]
                [["signal", "threshold", "F1"]])
    print("\nF1-maximising threshold per pathway (vs the 0.5 default):")
    print(best.to_string(index=False))

    provenance.record(ANALYSIS, outdir,
                      params={"held_out": args.held_out},
                      results={"best_thresholds": best.to_dict("records")})
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
