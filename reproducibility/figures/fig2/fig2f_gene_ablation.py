#!/usr/bin/env python3
"""Figure 2f -- gene ablation: how distributed is the response signature?

Genes are ranked (by mutual information with the signaling label, by
dispersion, or by mean expression), then progressively destroyed in the
held-out screen by resampling each gene's expression with replacement across
cells. This preserves the marginal distribution while severing the
gene-to-label relationship, so the drop in AUROC isolates that gene's
contribution.

The published result is that thousands of genes must be randomized before
accuracy falls to chance -- the signature is transcriptome-wide, not carried
by a handful of canonical response genes.

Outputs
    fig2f_ablation_curves.*     normalized AUROC vs number of genes ablated
    fig2f_genes_to_50pct.csv    genes needed to halve performance (the inset table)
    fig2f_ablation_points.csv   the underlying curve data
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from iris_repro import config, plotting, provenance

ANALYSIS = "fig2f"

# Genes were ablated in fixed-size blocks: the runners walk the ranked gene
# list as `variable_genes[73 * i : 73 * j]`, so the trailing index in each
# filename counts BLOCKS, not genes. Converting is what makes the summary
# table land on the published scale (thousands of genes, e.g. mP_d1 TGF-beta
# = 2920 = 40 blocks x 73).
GENES_PER_BLOCK = 73

# e.g. Bmp_ablation_accuracy_mi_0_120_independent_5.csv
ABLATION = re.compile(
    r"^(?P<prefix>in_)?(?P<signal>Bmp|Fgf|RA|Wnt|TgfB)_ablation_accuracy_"
    r"(?P<assay>[a-z]+)_0_(?P<n>\d+)_independent_(?P<heldout>\d+)\.csv$"
)


SOURCE_DATA_XLSX = ("in_vivo_results/source_data_scripts/"
                    "source_data_ablation_expfit.xlsx")


def load_published_fits() -> pd.DataFrame | None:
    """Read the authoritative exponential fits, if the workbook is present.

    ``in_vivo_results/source_data_scripts/12_ablation_expfit.py`` produced the
    numbers in the published Fig. 2f inset. Preferring its output avoids
    maintaining a second implementation that can silently drift; recomputing
    from the raw CSVs stays available via ``--recompute``.

    ``n50`` there is in 73-gene blocks, so it is converted to genes here.
    """
    path = Path(config.load_config()["roots"]["work"]) / SOURCE_DATA_XLSX
    if not path.exists():
        return None
    try:
        df = pd.read_excel(path, sheet_name="50pct_drop_points")
    except Exception as exc:
        print(f"  ! could not read {path.name}: {exc}")
        return None
    df = df.rename(columns={"sig": "signal"})
    df["held_out"] = [config.batch_name(int(b)) for b in df["heldout"]]
    df["genes_to_50pct"] = df["n50_interp"] * GENES_PER_BLOCK
    print(f"  using published fits from {path.name}")
    return df[["signal", "assay", "held_out", "genes_to_50pct",
               "reached_50_down"]]


def load_ablation_results(scope: str = "out") -> pd.DataFrame:
    """Collect the per-chunk ablation CSVs written by ``runner_cached_*.py``.

    Files prefixed ``in_`` record in-sample accuracy; the figure uses the
    held-out (``out``) curves.
    """
    root = config.results_dir("ablation")
    paths = list(Path(root).rglob("*_ablation_accuracy_*.csv"))
    if not paths:
        raise SystemExit(
            f"No ablation CSVs under {root}.\n"
            "These are produced by runner_cached_mi.py / runner_cached_hvg.py "
            "/ runner_high_expr.py via orchestrator_ablation.py."
        )

    rows = []
    for p in paths:
        m = ABLATION.match(p.name)
        if not m:
            continue
        is_in = bool(m.group("prefix"))
        if (scope == "in") != is_in:
            continue
        try:
            df = pd.read_csv(p)
        except Exception as exc:
            print(f"  ! unreadable {p.name}: {exc}")
            continue

        col = next((c for c in (f"AUROC_{scope}", "AUROC_out", "AUROC")
                    if c in df.columns), None)
        if col is None:
            continue
        rows.append({
            "signal": m.group("signal"),
            "assay": m.group("assay"),
            "held_out": config.batch_name(int(m.group("heldout"))),
            "n_blocks": int(m.group("n")),
            "n_ablated": int(m.group("n")) * GENES_PER_BLOCK,
            "AUROC": float(df[col].mean()),
        })

    if not rows:
        raise SystemExit(f"No '{scope}' ablation records parsed from {root}.")
    return (pd.DataFrame(rows)
              .groupby(["signal", "assay", "held_out", "n_ablated"], as_index=False)
              .median()   # median across repeats, as in 12_ablation_expfit.py
              .sort_values(["signal", "assay", "held_out", "n_ablated"]))


def normalize_curve(sub: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale AUROC to [0, 1] within one curve, as the figure does.

    0 corresponds to the worst (chance-level) AUROC observed for that curve
    and 1 to the intact model, which makes curves comparable across pathways
    whose absolute AUROCs differ.
    """
    sub = sub.sort_values("n_ablated").copy()
    lo, hi = sub["AUROC"].min(), sub["AUROC"].max()
    sub["AUROC_norm"] = 0.5 if hi <= lo else (sub["AUROC"] - lo) / (hi - lo)
    return sub


def fit_exponential(sub: pd.DataFrame):
    """Fit ``y = c + A*exp(-k*x)`` to an ablation curve.

    Three parameters, not two: ablation plateaus at a floor ``c`` rather than
    decaying to zero, because randomizing every gene still leaves the model at
    chance (AUROC ~0.5) rather than at nothing. Forcing a bare ``exp(-a x)``
    biases the half-point badly.

    This matches ``in_vivo_results/source_data_scripts/12_ablation_expfit.py``,
    which produced the published Fig. 2f numbers.

    Returns ``(params, ok)`` with ``params = (A, k, c)``.
    """
    from scipy.optimize import curve_fit

    s = sub.sort_values("n_ablated")
    x, y = s["n_ablated"].to_numpy(float), s["AUROC"].to_numpy(float)
    if len(x) < 4:
        return (np.nan, np.nan, np.nan), False
    try:
        p0 = [max(y.max() - y.min(), 1e-3), 1e-3 / max(GENES_PER_BLOCK, 1), y.min()]
        (A, k, c), _ = curve_fit(
            lambda t, A, k, c: c + A * np.exp(-k * t), x, y, p0=p0,
            maxfev=20000,
            bounds=([0, 0, 0.0], [np.inf, np.inf, 1.0]),
        )
        return (float(A), float(k), float(c)), k > 0
    except Exception:
        return (np.nan, np.nan, np.nan), False


def _normalized_fit(x, params):
    """Fitted curve rescaled to [0, 1] over the observed range."""
    A, k, c = params
    yfit = c + A * np.exp(-k * np.asarray(x, dtype=float))
    lo, hi = np.nanmin(yfit), np.nanmax(yfit)
    return np.full_like(yfit, 0.5) if hi <= lo else (yfit - lo) / (hi - lo)


def genes_to_half(sub: pd.DataFrame, method: str = "fit") -> float:
    """Genes to ablate before half the normalized performance is lost.

    ``method='fit'`` locates where the min-max normalized *fitted* curve
    crosses 0.5, matching the published Fig. 2f inset table.
    ``method='interp'`` interpolates the raw points instead (noisier).
    """
    s = sub.sort_values("n_ablated")
    x = s["n_ablated"].to_numpy(float)

    if method == "fit":
        params, ok = fit_exponential(s)
        if not ok:
            return float("nan")
        grid = np.linspace(x.min(), x.max(), 4000)
        ynorm = _normalized_fit(grid, params)
        below = np.where(ynorm <= 0.5)[0]
        return float(grid[below[0]]) if below.size else float("nan")

    y = s["AUROC_norm"].to_numpy(float)
    below = np.where(y <= 0.5)[0]
    if below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(x[0])
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return float(x0 if y0 == y1 else x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assay", default="mi",
                    help="gene ranking: mi (mutual information), hvg, expr")
    ap.add_argument("--scope", default="out", choices=["out", "in"])
    ap.add_argument("--recompute", action="store_true",
                    help="refit the curves here instead of reading the "
                         "published source-data workbook")
    ap.add_argument("--half-method", default="fit", choices=["fit", "interp"],
                    help="'fit' reads the 50%% point off the exponential fit "
                         "(as published); 'interp' uses the raw points")
    args = ap.parse_args()

    outdir = config.output_dir("fig2")
    provenance.check_environment()

    df = load_ablation_results(args.scope)
    print(f"Loaded {len(df)} ablation points; assays present: "
          f"{sorted(df['assay'].unique())}")
    df.to_csv(outdir / "fig2f_ablation_points.csv", index=False)

    sub_assay = df[df["assay"] == args.assay]
    if sub_assay.empty:
        raise SystemExit(f"No records for assay {args.assay!r}; "
                         f"available: {sorted(df['assay'].unique())}")

    plt = plotting.set_style()
    colors = config.palette()
    signals = [s for s in config.signals() if s in set(sub_assay["signal"])]

    fig, axes = plt.subplots(1, len(signals), figsize=(1.5 * len(signals), 1.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    half_rows = []

    for ax, sig in zip(axes, signals):
        for held, grp in sub_assay[sub_assay["signal"] == sig].groupby("held_out"):
            curve = normalize_curve(grp)
            c = colors.get(sig, "k")
            # Raw evaluations are noisy; show them faintly under the fit.
            ax.plot(curve["n_ablated"], curve["AUROC_norm"],
                    color=c, lw=0.4, alpha=0.25)
            params, ok = fit_exponential(curve)
            if ok:
                xs = np.linspace(0, curve["n_ablated"].max(), 400)
                ax.plot(xs, _normalized_fit(xs, params), color=c, lw=0.8, label=held)
            half_rows.append({
                "signal": sig, "assay": args.assay, "held_out": held,
                "genes_to_50pct": genes_to_half(curve, args.half_method),
                "A": params[0], "k": params[1], "c": params[2],
            })
        ax.set_title(config.display_name(sig))
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=4)
    axes[0].set_ylabel("Normalized AUROC (min-max)")
    # One shared x-label: per-axes labels collide at this figure width.
    fig.supxlabel(f"Genes randomized (ranked by {args.assay})", fontsize=7)
    fig.tight_layout()
    plotting.save(fig, outdir, "fig2f_ablation_curves")
    plt.close(fig)

    half = pd.DataFrame(half_rows)

    # Prefer the published fits unless the caller asked to recompute.
    if not args.recompute:
        published = load_published_fits()
        if published is not None:
            sub = published[published["assay"] == args.assay]
            if not sub.empty:
                half = sub.copy()

    table = half.pivot_table(index="held_out", columns="signal",
                             values="genes_to_50pct")
    table.to_csv(outdir / "fig2f_genes_to_50pct.csv")
    print("\nGenes required to reduce accuracy by 50%:")
    print(table.round(0).to_string())

    provenance.record(
        ANALYSIS, outdir, params={"assay": args.assay, "scope": args.scope},
        results={"genes_to_50pct": half.to_dict("records")},
    )
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
