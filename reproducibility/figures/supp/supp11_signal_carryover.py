#!/usr/bin/env python3
"""Supp. Fig. 11 -- false positives caused by signal carry-over.

The hM_d4 screen stimulates in two sequential steps, so a pathway switched on
in Step 2 and off in Step 3 lets us ask a pointed question: does a *prior*
exposure make IRIS call the pathway active when it currently is not?

For each pathway the Step-3 false-positive rate is compared between cells that
did and did not receive that same signal in Step 2. The published conclusion is
that carry-over raises FPR for BMP/WNT/TGF-beta, lowers it for RA, and leaves
FGF unchanged -- but FPRs stay below ~0.3, so the effect is real yet bounded.

Outputs
    supp11_carryover_fpr.*      FPR with bootstrap CIs, split by prior exposure
    supp11_carryover_stats.csv  two-tailed z-test per pathway
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from iris_repro import config, metrics, plotting, provenance

ANALYSIS = "supp11"
HM_D4 = 5
N_BOOT = 1000


def two_proportion_ztest(k1, n1, k2, n2):
    """Two-tailed z-test comparing two proportions."""
    from scipy.stats import norm
    if min(n1, n2) == 0:
        return {"z": np.nan, "p": np.nan}
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": np.nan, "p": np.nan}
    z = (p1 - p2) / se
    return {"z": float(z), "p": float(2 * (1 - norm.cdf(abs(z))))}


def load_step_labels() -> pd.DataFrame | None:
    """Per-cell Step 2 / Step 3 stimulation labels for hM_d4, if available."""
    work = Path(config.load_config()["roots"]["work"])
    for name in ("hM_d4_step_labels.csv", "carryover_labels.csv"):
        p = work / name
        if p.exists():
            print(f"  reading {p.name}")
            return pd.read_csv(p, index_col=0)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    outdir = config.output_dir("supp")
    provenance.check_environment()

    labels = load_step_labels()
    if labels is None:
        raise SystemExit(
            "Supp. Fig. 11 needs per-cell Step 2 and Step 3 stimulation labels "
            "for the hM_d4 screen, which are not saved as a standalone table in "
            "the working directory.\n\n"
            "They are recoverable from the screen barcodes: each hM_d4 barcode "
            "encodes a 12-bit Step2-Step3 combination (see the Methods, "
            "'Sequential combinatorial signal screen'). Reconstruct them into "
            f"{Path(config.load_config()['roots']['work']) / 'hM_d4_step_labels.csv'} "
            "with columns: index, <Signal>_step2, <Signal>_step3.\n\n"
            "The published panel was produced from the ground-truth labels shown "
            "in Supp. Fig. 11a."
        )

    from fig2.fig2cd_generalization import collect
    preds = collect("clean_splits", single_holdout_only=True)
    preds = preds[preds["held_out"] == config.batch_name(HM_D4)]

    rows = []
    for sig in config.signals():
        s2, s3 = f"{sig}_step2", f"{sig}_step3"
        if s2 not in labels or s3 not in labels:
            print(f"  ! {sig}: missing {s2}/{s3}, skipping")
            continue
        sub = preds[preds["signal"] == sig].join(labels, how="inner")
        # False positives are only defined where the signal is OFF in Step 3.
        off = sub[sub[s3] == 0]
        for prior in (0, 1):
            grp = off[off[s2] == prior]
            if len(grp) == 0:
                continue
            fp = int((grp["y_score"] > 0.5).sum())
            ci = metrics.bootstrap_ci((grp["y_score"] > 0.5).astype(float),
                                      n_boot=N_BOOT)
            rows.append({"signal": sig, "prior_exposure": bool(prior),
                         "n": len(grp), "n_false_positive": fp,
                         "FPR": fp / len(grp), **{f"ci_{k}": v for k, v in ci.items()}})

    if not rows:
        raise SystemExit("No usable carry-over comparisons.")

    table = pd.DataFrame(rows)
    stats = []
    for sig, sub in table.groupby("signal"):
        a = sub[sub["prior_exposure"]]
        b = sub[~sub["prior_exposure"]]
        if len(a) and len(b):
            r = two_proportion_ztest(int(a["n_false_positive"].iloc[0]), int(a["n"].iloc[0]),
                                     int(b["n_false_positive"].iloc[0]), int(b["n"].iloc[0]))
            stats.append({"signal": sig,
                          "FPR_prior": float(a["FPR"].iloc[0]),
                          "FPR_no_prior": float(b["FPR"].iloc[0]),
                          **r, "significant": (r["p"] or 1) < 0.05})

    table.to_csv(outdir / "supp11_carryover_fpr_table.csv", index=False)
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(outdir / "supp11_carryover_stats.csv", index=False)
    print(stats_df.to_string(index=False))

    plt = plotting.set_style()
    colors = config.palette()
    fig, ax = plt.subplots(figsize=(2.8, 1.7))
    sigs = list(stats_df["signal"])
    for i, sig in enumerate(sigs):
        for j, prior in enumerate((False, True)):
            r = table[(table["signal"] == sig) & (table["prior_exposure"] == prior)]
            if not len(r):
                continue
            x = i + (j - 0.5) * 0.3
            ax.errorbar(x, r["FPR"].iloc[0],
                        yerr=[[r["FPR"].iloc[0] - r["ci_lo"].iloc[0]],
                              [r["ci_hi"].iloc[0] - r["FPR"].iloc[0]]],
                        fmt="v" if not prior else "^",
                        color=colors.get(sig, "k"), ms=4, capsize=2, lw=0.8)
    ax.set_xticks(range(len(sigs)))
    ax.set_xticklabels([config.display_name(s) for s in sigs])
    ax.set_ylabel("False positive rate")
    ax.set_ylim(0, 1)
    ax.set_title("Step-3 FPR by prior Step-2 exposure")
    plotting.save(fig, outdir, "supp11_carryover_fpr")
    plt.close(fig)

    provenance.record(ANALYSIS, outdir, results={"stats": stats})
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
