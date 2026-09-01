#!/usr/bin/env python3
"""Figure 3d/e/f -- signaling histories along the endodermal and cardiac lineages.

Cells from the Pijuan-Sala E6.5-E8.5 gastrulation atlas are ordered along a
diffusion pseudotime; IRIS predicts each pathway independently and the
combinatorial code is the concatenation of those calls.

Reproduces the published claims:
  * BMP activates along the cardiac but not the endodermal lineage
  * WNT inactivates before cardiomyocyte differentiation
  * RA activates late in both (anterior foregut / atrial cardiomyocytes)
  * combinations appear in the expected temporal order (Mann-Whitney, one-tailed)

Outputs
    fig3d_endoderm_trends.*     per-pathway activation frequency vs pseudotime
    fig3d_cardiac_trends.*
    fig3ef_combination_order.*  stacked combination frequency per lineage
    fig3_trend_stats.csv        Spearman rho / p per pathway per lineage
    fig3_order_stats.csv        Mann-Whitney tests on combination ordering
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from iris_repro import config, metrics, plotting, provenance

ANALYSIS = "fig3_lineages"

# Cell populations forming each lineage, per the Methods. Both spellings of
# the definitive-endoderm cluster are listed because the atlas and the saved
# in vivo tables label it differently.
LINEAGES = {
    "endoderm": ["Primitive Streak", "Anterior Primitive Streak",
                 "Def. endoderm", "Definitive endoderm", "Gut",
                 "Visceral endoderm"],
    "cardiac": ["Primitive Streak", "Nascent mesoderm", "Mixed mesoderm",
                "Pharyngeal mesoderm", "Cardiomyocytes"],
}

# Expected order of combinations along each lineage (Fig. 3e/f).
EXPECTED_ORDER = {
    "endoderm": ["11000", "10000", "00001"],
    "cardiac": ["11110", "?1110", "?1111"],
}


# The saved code strings spell BMP in caps; normalise to the config's names.
_CODE_ALIASES = {"BMP": "Bmp", "TGFB": "TgfB", "FGF": "Fgf", "WNT": "Wnt",
                 "RA": "RA", "Bmp": "Bmp", "TgfB": "TgfB", "Fgf": "Fgf",
                 "Wnt": "Wnt"}


def _parse_code_string(code: str) -> dict:
    """Decode ``'BMP+TgfB+Fgf-Wnt+RA-'`` into per-pathway 0/1 calls."""
    import re as _re
    out = {}
    for sig, sign in _re.findall(r"(BMP|TGFB|TgfB|FGF|Fgf|WNT|Wnt|RA)([+-])", code):
        out[f"{_CODE_ALIASES.get(sig, sig)}_pred"] = 1 if sign == "+" else 0
    return out


def load_predictions(lineage: str) -> pd.DataFrame:
    """Load per-cell IRIS calls, diffusion coordinates and pseudotime.

    Primary source is ``diffmap_with_labels.csv`` from the in vivo analysis,
    which already carries the diffusion components, the cell-type annotation
    and the IRIS combinatorial code per cell. Pseudotime is taken as DC1,
    the diffusion component along which the lineage is ordered.
    """
    root = Path(config.results_dir("in_vivo"))
    combined = root / "diffmap_with_labels.csv"
    if combined.exists():
        print(f"  reading {combined.name}")
        df = pd.read_csv(combined, index_col=0)
        wanted = set(LINEAGES[lineage])
        sub = df[df["celltype"].astype(str).isin(wanted)].copy()
        if sub.empty:
            raise SystemExit(
                f"No {lineage} cells in {combined.name}; "
                f"present: {sorted(df['celltype'].unique())}"
            )
        calls = pd.DataFrame([_parse_code_string(c) for c in sub["code"]],
                             index=sub.index)
        out = pd.concat([sub, calls], axis=1)
        # Order along the lineage: DC1 is the dominant trajectory axis here.
        out["pseudotime"] = out["DC1"].rank(pct=True)
        # Orient so the least differentiated population sits at low pseudotime.
        roots = out["celltype"].astype(str) == "Primitive Streak"
        if roots.any() and out.loc[roots, "pseudotime"].mean() > 0.5:
            out["pseudotime"] = 1.0 - out["pseudotime"]
        print(f"  {len(out):,} cells in the {lineage} lineage")
        return out

    for p in (root / f"{lineage}_lineage_predictions.csv",
              root / f"{lineage}_predictions.csv"):
        if p.exists():
            print(f"  reading {p.name}")
            return pd.read_csv(p, index_col=0)

    raise SystemExit(
        f"No prediction table for the {lineage} lineage under {root}.\n"
        "Generate it with --predict (requires the gastrulation atlas and a GPU)."
    )


def predict_lineage(lineage: str, outdir: Path) -> pd.DataFrame:
    """Run IRIS over a lineage and compute its diffusion pseudotime."""
    import scanpy as sc
    from iris_repro.data import load_gastrulation
    from iris_repro.model import IRISModel

    adata = load_gastrulation()
    wanted = set(LINEAGES[lineage])
    present = set(adata.obs["celltype"].astype(str))
    keep = wanted & present
    if not keep:
        raise SystemExit(
            f"None of the {lineage} populations are present. "
            f"Expected some of {sorted(wanted)}."
        )
    if keep != wanted:
        print(f"  ! missing populations: {sorted(wanted - present)}")
    sub = adata[adata.obs["celltype"].astype(str).isin(keep)].copy()
    print(f"  {sub.n_obs:,} cells in the {lineage} lineage")

    # Diffusion pseudotime, rooted at the least differentiated population.
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    sc.pp.pca(sub)
    sc.pp.neighbors(sub)
    sc.tl.diffmap(sub)
    root_pop = "Primitive Streak"
    roots = np.where(sub.obs["celltype"].astype(str) == root_pop)[0]
    sub.uns["iroot"] = int(roots[0]) if len(roots) else 0
    sc.tl.dpt(sub)

    out = pd.DataFrame({"pseudotime": sub.obs["dpt_pseudotime"].values,
                        "celltype": sub.obs["celltype"].astype(str).values},
                       index=sub.obs_names)
    for sig in config.signals():
        model_dir = find_pretrained(sig)
        print(f"  predicting {sig} with {Path(model_dir).name}")
        m = IRISModel.load(model_dir, sub, sig)
        out[f"{sig}_proba"] = m.predict_proba().values
        out[f"{sig}_pred"] = (out[f"{sig}_proba"] >
                              m.params["classifier_threshold"]).astype(int)

    path = Path(config.results_dir("in_vivo")) / f"{lineage}_lineage_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    print(f"  wrote {path}")
    return out


def find_pretrained(signal: str) -> str:
    """Locate the gastrulation-trained checkpoint for a pathway."""
    root = Path(config.load_config()["models"]["gastrulation"])
    arch = config.architecture(signal)
    pattern = (f"zorn_scanvi_layers_1_hidden={arch['n_hidden']}_"
               f"latent={arch['n_latent']}_{signal}_class_rs")
    hits = sorted(root.glob(pattern + "*"))
    if not hits:
        raise SystemExit(f"No pretrained model for {signal} matching {pattern}*")
    # These directories hold per-epoch snapshots; take the last one saved.
    snaps = sorted(p for p in hits[0].iterdir() if p.is_dir())
    return str(snaps[-1] if snaps else hits[0])


def code_series(df: pd.DataFrame) -> pd.Series:
    """Concatenate per-pathway binary calls into the 5-bit code."""
    bits = [df[f"{s}_pred"].astype(int).astype(str) for s in config.signals()]
    return pd.Series(["".join(t) for t in zip(*bits)], index=df.index)


def analyse_lineage(lineage: str, df: pd.DataFrame, outdir: Path):
    plt = plotting.set_style()
    colors = config.palette()
    pt = df["pseudotime"].to_numpy(float)

    # --- per-pathway trends (Fig. 3d) -------------------------------------
    fig, ax = plt.subplots(figsize=(2.8, 1.7))
    stats = []
    for sig in config.signals():
        col = f"{sig}_pred"
        if col not in df:
            continue
        y = df[col].to_numpy(float)
        plotting.plot_trend(pt, y, ax=ax, color=colors.get(sig, "k"),
                            label=config.display_name(sig))
        s = metrics.spearman_trend(pt, y)
        stats.append({"lineage": lineage, "signal": sig, **s,
                      "significant": s["p"] < 0.05})
    ax.set_xlabel("Diffusion pseudotime")
    ax.set_ylabel("Frequency")
    ax.set_ylim(0, 1)
    ax.set_title(f"{lineage.capitalize()} lineage")
    ax.legend(frameon=False, ncol=2, fontsize=5)
    plotting.save(fig, outdir, f"fig3d_{lineage}_trends")
    plt.close(fig)

    # --- combination ordering (Fig. 3e/f) ---------------------------------
    codes = code_series(df)
    fig, ax = plt.subplots(figsize=(3.0, 1.4))
    plotting.combination_frequency_plot(codes, pt, ax=ax)
    ax.set_title(f"{lineage.capitalize()} lineage: signal combinations")
    plotting.save(fig, outdir, f"fig3ef_{lineage}_combinations")
    plt.close(fig)

    # Test that each expected combination precedes the next along pseudotime.
    order_stats = []
    expected = [c for c in EXPECTED_ORDER[lineage] if "?" not in c]
    for a, b in zip(expected, expected[1:]):
        ta, tb = pt[(codes == a).values], pt[(codes == b).values]
        if len(ta) < 3 or len(tb) < 3:
            print(f"  ! too few cells to test {a} < {b} "
                  f"(n={len(ta)}, {len(tb)})")
            continue
        # One-tailed: is b LATER than a?
        r = metrics.mannwhitney_greater(tb, ta)
        order_stats.append({"lineage": lineage, "comparison": f"{a} < {b}",
                            "n_a": len(ta), "n_b": len(tb), **r,
                            "significant": r["p"] < 0.05})
    return stats, order_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lineage", choices=["endoderm", "cardiac", "both"],
                    default="both")
    ap.add_argument("--predict", action="store_true",
                    help="run IRIS on the atlas instead of loading saved predictions")
    args = ap.parse_args()

    outdir = config.output_dir("fig3")
    provenance.check_environment()

    lineages = ["endoderm", "cardiac"] if args.lineage == "both" else [args.lineage]
    trend_stats, order_stats = [], []

    for lin in lineages:
        print(f"\n=== {lin} lineage ===")
        df = predict_lineage(lin, outdir) if args.predict else load_predictions(lin)
        t, o = analyse_lineage(lin, df, outdir)
        trend_stats += t
        order_stats += o

    trends = pd.DataFrame(trend_stats)
    trends.to_csv(outdir / "fig3_trend_stats.csv", index=False)
    print("\nPathway trends (two-tailed Spearman):")
    print(trends.to_string(index=False))

    if order_stats:
        orders = pd.DataFrame(order_stats)
        orders.to_csv(outdir / "fig3_order_stats.csv", index=False)
        print("\nCombination ordering (one-tailed Mann-Whitney):")
        print(orders.to_string(index=False))

    provenance.record(
        ANALYSIS, outdir,
        inputs=[config.data_path("gastrulation_atlas", require=False)],
        params={"lineages": lineages, "predicted": args.predict},
        results={"trends": trend_stats, "order": order_stats},
    )
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
