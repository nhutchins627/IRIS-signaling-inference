#!/usr/bin/env python3
"""Figure 4a -- signaling in organ-specific foregut mesenchyme.

IRIS predicts pathway activity across mesenchymal populations of the E9-E9.5
mouse foregut (Han et al. 2020). The published result is that WNT and BMP are
significantly enriched in respiratory mesenchyme relative to all other
mesenchymal fates -- the prediction that motivated the ex vivo foregut
experiment and the revised hESC differentiation protocol.

Enrichment uses a one-sided Fisher's exact test of respiratory vs all other
mesenchyme, matching the figure legend.

Outputs
    fig4a_enrichment.csv         odds ratio + p per pathway
    fig4a_enrichment_bars.*      -log10(p) per pathway, respiratory vs rest
    fig4a_fractions.csv          fraction of cells called active, per population
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from iris_repro import config, metrics, plotting, provenance

ANALYSIS = "fig4a"

# The four mesenchymal fates that diverge in the E9-E9.5 foregut.
TARGET = "respiratory"
POPULATIONS = ["respiratory", "esophageal", "pharyngeal",
               "dorsal_lateral_foregut"]


def _canonical(label: str) -> str | None:
    """Collapse the fine-grained annotations onto the four Fig. 4a fates.

    The source object annotates sub-states separately ('respiratory-lung',
    'respiratory-trachea', 'esophagus-1', ...). Fig. 4a compares the four
    divergent organ fates, so those sub-states are merged. Labels outside
    this lineage (other atlases in the same object) return None and are
    dropped -- the comparison is respiratory vs *other foregut mesenchyme*,
    not vs every cell in the file.
    """
    s = str(label).strip().lower().replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    if s.startswith("respiratory") or s in {"lung", "trachea"}:
        return "respiratory"
    if s.startswith("esophagus") or s.startswith("esophageal"):
        return "esophageal"
    if "pharyn" in s and "mesoderm" not in s:
        return "pharyngeal"
    if "dorsal lateral foregut" in s or "dorsal   lateral foregut" in s:
        return "dorsal_lateral_foregut"
    return None


def load_predictions() -> pd.DataFrame:
    """Per-cell IRIS calls for the foregut mesenchyme populations.

    Prefers a saved prediction table; otherwise runs inference with the
    ``miram_out_*`` checkpoints trained for this dataset.
    """
    root = Path(config.results_dir("in_vivo"))
    for name in ("foregut_mesenchyme_predictions.csv",
                 "mesenchyme_predictions.csv"):
        p = root / name
        if p.exists():
            print(f"  reading {p.name}")
            return pd.read_csv(p, index_col=0)
    raise SystemExit(
        f"No foregut mesenchyme prediction table under {root}.\n"
        "Generate it with --predict (needs the foregut h5ad and a GPU)."
    )


def find_pretrained(signal: str) -> Path:
    """The ``miram_out_<Signal>_...`` checkpoint for the foregut analysis."""
    root = Path(config.load_config()["models"]["foregut"])
    arch = config.architecture(signal)
    pattern = (f"miram_out_{signal}_scviALL_scanvi_"
               f"h{arch['n_hidden']}_z{arch['n_latent']}_model")
    hits = sorted(root.glob(pattern))
    if not hits:
        raise SystemExit(f"No foregut model for {signal} matching {pattern}")
    return hits[0]


def predict() -> pd.DataFrame:
    """Run IRIS over the foregut mesenchyme dataset."""
    import scanpy as sc
    from iris_repro.model import IRISModel

    from iris_repro.data import ensure_counts_layer

    path = config.data_path("foregut_mesenchyme")
    print(f"  reading {path.name}")
    adata = sc.read_h5ad(path)
    adata.obs_names_make_unique()

    label_col = next((c for c in ("celltype", "cell_type", "annotation")
                      if c in adata.obs), None)
    if label_col is None:
        raise SystemExit(f"No cell-type column in obs: {list(adata.obs.columns)}")

    # The file bundles several atlases; keep only the foregut mesenchymal
    # fates compared in Fig. 4a before running inference.
    pops = pd.Series([_canonical(v) for v in adata.obs[label_col]],
                     index=adata.obs_names)
    keep = pops.notna().to_numpy()
    if not keep.any():
        raise SystemExit(
            f"No foregut mesenchyme populations matched in {path.name}. "
            f"Labels present: {sorted(set(adata.obs[label_col].astype(str)))[:20]}"
        )
    adata = adata[keep].copy()
    ensure_counts_layer(adata)
    print(f"  {adata.n_obs:,} mesenchymal cells x {adata.n_vars:,} genes")

    out = pd.DataFrame({"population": pops[keep].values}, index=adata.obs_names)
    for sig in config.signals():
        mdir = find_pretrained(sig)
        print(f"  predicting {sig} with {mdir.name}")
        m = IRISModel.load(mdir, adata, sig)
        proba = m.predict_proba()
        out[f"{sig}_proba"] = proba.values
        out[f"{sig}_pred"] = (proba.values >
                              m.params["classifier_threshold"]).astype(int)

    path = Path(config.results_dir("in_vivo")) / "foregut_mesenchyme_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    print(f"  wrote {path}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predict", action="store_true",
                    help="run inference instead of loading saved predictions")
    args = ap.parse_args()

    outdir = config.output_dir("fig4")
    provenance.check_environment()

    df = predict() if args.predict else load_predictions()
    df["population"] = [_canonical(v) for v in df["population"]]
    n_before = len(df)
    df = df[df["population"].notna()].copy()
    print(f"\nKept {len(df):,}/{n_before:,} cells in the four foregut "
          f"mesenchymal fates")
    print("\nPopulations:")
    print(df["population"].value_counts().to_string())

    if TARGET not in set(df["population"]):
        raise SystemExit(f"No {TARGET!r} population found; "
                         f"have {sorted(df['population'].unique())}")

    # --- enrichment: respiratory vs all other mesenchyme ------------------
    is_target = (df["population"] == TARGET).to_numpy()
    rows, fractions = [], []
    for sig in config.signals():
        col = f"{sig}_pred"
        if col not in df:
            continue
        active = df[col].to_numpy().astype(bool)
        # 2x2: rows = respiratory / other, cols = active / inactive
        table = [[int((is_target & active).sum()), int((is_target & ~active).sum())],
                 [int((~is_target & active).sum()), int((~is_target & ~active).sum())]]
        r = metrics.fisher_enrichment(table, alternative="greater")
        rows.append({"signal": sig, "n_respiratory_active": table[0][0],
                     "n_respiratory": int(is_target.sum()),
                     "n_other_active": table[1][0],
                     "n_other": int((~is_target).sum()),
                     **r, "significant": r["p"] < 0.05})
        for pop, sub in df.groupby("population"):
            fractions.append({"population": pop, "signal": sig,
                              "fraction_active": float(sub[col].mean()),
                              "n": len(sub)})

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "fig4a_enrichment.csv", index=False)
    print("\nEnrichment in respiratory mesenchyme (one-sided Fisher's exact):")
    print(table.to_string(index=False))

    frac = pd.DataFrame(fractions)
    frac.to_csv(outdir / "fig4a_fractions.csv", index=False)

    # --- plots -------------------------------------------------------------
    plt = plotting.set_style()
    colors = config.palette()

    fig, ax = plt.subplots(figsize=(2.2, 1.7))
    sigs = list(table["signal"])
    # Clip -log10(p) so a p of exactly 0 stays on-scale.
    logp = [-np.log10(max(p, 1e-300)) for p in table["p"]]
    ax.bar(range(len(sigs)), logp,
           color=[colors.get(s, "k") for s in sigs], edgecolor="none")
    ax.axhline(-np.log10(0.05), ls="--", lw=0.6, c="grey")
    ax.set_xticks(range(len(sigs)))
    ax.set_xticklabels([config.display_name(s) for s in sigs], rotation=45,
                       ha="right")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Enrichment in respiratory mesenchyme")
    plotting.save(fig, outdir, "fig4a_enrichment_bars")
    plt.close(fig)

    # Fraction of active cells per population, grouped by pathway.
    pops = [p for p in POPULATIONS if p in set(frac["population"])]
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    width = 0.8 / max(len(pops), 1)
    for j, pop in enumerate(pops):
        vals = [float(frac[(frac["population"] == pop) &
                           (frac["signal"] == s)]["fraction_active"].iloc[0])
                if len(frac[(frac["population"] == pop) & (frac["signal"] == s)])
                else np.nan for s in sigs]
        xs = [i + j * width - 0.4 + width / 2 for i in range(len(sigs))]
        ax.bar(xs, vals, width=width * 0.9, label=pop.replace("_", " "),
               edgecolor="none")
    ax.set_xticks(range(len(sigs)))
    ax.set_xticklabels([config.display_name(s) for s in sigs])
    ax.set_ylabel("Fraction of cells active")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=5, ncol=2)
    plotting.save(fig, outdir, "fig4a_population_fractions")
    plt.close(fig)

    provenance.record(
        ANALYSIS, outdir,
        inputs=[config.data_path("foregut_mesenchyme", require=False)],
        params={"target": TARGET, "predicted": args.predict},
        results={"enrichment": table.to_dict("records")},
    )
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
