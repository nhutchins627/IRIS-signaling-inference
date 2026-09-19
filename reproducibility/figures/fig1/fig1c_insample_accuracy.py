#!/usr/bin/env python3
"""Figure 1c -- IRIS vs the response-gene method, in-sample (mESC screen).

An 80/20 random split of the mESC differentiation data (Yeo et al. 2020).
IRIS sees signaling labels for the 80%; the held-out 20% is scored against
ground truth. The response-gene baseline (Supp. Fig. 1) is scored on the same
cells for a like-for-like comparison.

Outputs
    fig1c_roc.{svg,png}          ROC, IRIS (solid) vs response gene (dashed)
    fig1c_precision_recall.*     PR curves for the same comparison
    fig1c_metrics.csv            AUROC / AUPRC / F1 per pathway per method
    fig1c.provenance.json

Usage
    python fig1c_insample_accuracy.py                 # reuse cached predictions
    python fig1c_insample_accuracy.py --retrain       # refit models (GPU, ~20 min)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from iris_repro import config, metrics, plotting, provenance
from iris_repro.data import binary_labels, load_screens

ANALYSIS = "fig1c"
MESC_BATCHES = [1, 2, 3]   # mP_d1, mixed, mE_d2 -- the mouse screens
TEST_FRACTION = 0.20
SEED = 42


def compute_response_gene_scores(adata) -> pd.DataFrame:
    """Summed log-normalised response-gene expression, per pathway."""
    import scanpy as sc

    norm = adata.copy()
    sc.pp.normalize_total(norm, target_sum=1e6)
    sc.pp.log1p(norm)

    scores = {}
    for sig in config.signals():
        genes = config.response_genes(sig)
        present = [g for g in genes if g in norm.var_names]
        if not present:
            print(f"  ! {sig}: no response genes present, skipping")
            continue
        if len(present) < len(genes):
            missing = sorted(set(genes) - set(present))
            print(f"  ! {sig}: {len(present)}/{len(genes)} genes found "
                  f"(missing {', '.join(missing)})")
        scores[sig] = metrics.response_gene_score(norm, present)
    return pd.DataFrame(scores, index=adata.obs_names)


def train_iris(adata, test_idx) -> pd.DataFrame:
    """Fit one IRIS model per pathway, hiding labels for the test cells.

    This split is a random 80/20 over cells rather than a held-out batch, so
    the mask is applied per cell; everything else (architecture, epochs,
    likelihood, seed) comes from the shared config via ``IRISModel``.
    """
    import scvi
    from iris_repro.data import ensure_counts_layer
    from iris_repro.model import UNLABELED, IRISModel, set_seed

    preds = {}
    for sig in config.signals():
        print(f"  training IRIS [{sig}] ...")
        sub = adata.copy()
        ensure_counts_layer(sub)

        col = f"{sig}_class_masked"
        labels = sub.obs[f"{sig}_class"].astype("category")
        if UNLABELED not in labels.cat.categories:
            labels = labels.cat.add_categories([UNLABELED])
        labels = labels.copy()
        labels.iloc[test_idx] = UNLABELED
        sub.obs[col] = labels

        model = IRISModel(sig)
        model.label_key = col
        model._setup(sub, col)

        set_seed(SEED)
        vae = scvi.model.SCVI(sub, n_layers=model.n_layers,
                              n_latent=model.n_latent, n_hidden=model.n_hidden,
                              gene_likelihood=model.params["gene_likelihood"])
        vae.train(max_epochs=model.params["scvi_epochs"])
        scanvae = scvi.model.SCANVI.from_scvi_model(
            vae, labels_key=col, unlabeled_category=UNLABELED)
        scanvae.train(max_epochs=model.params["scanvi_epochs"])
        preds[sig] = scanvae.predict(soft=True)["Stim"].values
    return pd.DataFrame(preds, index=adata.obs_names)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrain", action="store_true",
                    help="refit IRIS instead of loading cached predictions")
    ap.add_argument("--cache", type=Path,
                    default=config.output_dir("fig1") / "fig1c_iris_predictions.csv")
    args = ap.parse_args()

    outdir = config.output_dir("fig1")
    provenance.check_environment()

    print("Loading mESC screens ...")
    adata = load_screens()
    adata = adata[np.isin(adata.obs["batch"].values, MESC_BATCHES)].copy()
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    rng = np.random.default_rng(SEED)
    n_test = int(round(TEST_FRACTION * adata.n_obs))
    test_idx = rng.choice(adata.n_obs, size=n_test, replace=False)
    test_mask = np.zeros(adata.n_obs, dtype=bool)
    test_mask[test_idx] = True
    print(f"  train {(~test_mask).sum():,} / test {test_mask.sum():,}")

    # --- response-gene baseline ------------------------------------------
    print("Scoring response-gene baseline ...")
    rg_scores = compute_response_gene_scores(adata)

    # --- IRIS -------------------------------------------------------------
    if args.retrain:
        print("Training IRIS (GPU recommended) ...")
        iris_scores = train_iris(adata, test_idx)
        iris_scores.to_csv(args.cache)
        print(f"  cached -> {args.cache}")
    elif args.cache.exists():
        print(f"Loading cached IRIS predictions from {args.cache}")
        iris_scores = pd.read_csv(args.cache, index_col=0)
        iris_scores.index = adata.obs_names
    else:
        # No GPU run yet. The baseline half of the panel is still worth
        # emitting -- it is Supp. Fig. 1b on its own -- so save it and stop
        # rather than throwing the work away.
        rows = [{"signal": sig, "method": "response_gene",
                 **metrics.classification_metrics(
                     binary_labels(adata, sig)[test_mask],
                     np.asarray(rg_scores[sig])[test_mask])}
                for sig in config.signals() if sig in rg_scores]
        baseline = metrics.summarize_runs(rows)
        baseline.to_csv(outdir / "fig1c_response_gene_metrics.csv", index=False)
        print("\nResponse-gene baseline on the held-out 20%:")
        print(baseline.to_string(index=False))

        curves = {sig: {"y_true": binary_labels(adata, sig)[test_mask],
                        "y_score": np.asarray(rg_scores[sig])[test_mask]}
                  for sig in config.signals() if sig in rg_scores}
        plt = plotting.set_style()
        fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.1))
        plotting.plot_roc(curves, ax=axes[0], title="Response genes: ROC")
        plotting.plot_pr(curves, ax=axes[1], title="Response genes: PR")
        fig.tight_layout()
        plotting.save(fig, outdir, "fig1c_response_gene_only")
        plt.close(fig)

        raise SystemExit(
            f"\nWrote the response-gene baseline to {outdir}.\n"
            f"For the full IRIS comparison, run once with --retrain "
            f"(needs a GPU); predictions are then cached at {args.cache}."
        )

    # --- evaluate on the held-out 20% ------------------------------------
    rows, roc_iris, roc_rg = [], {}, {}
    for sig in config.signals():
        if sig not in rg_scores:
            continue
        y = binary_labels(adata, sig)[test_mask]
        for method, scores in (("IRIS", iris_scores), ("response_gene", rg_scores)):
            s = np.asarray(scores[sig])[test_mask]
            m = metrics.classification_metrics(y, s)
            rows.append({"signal": sig, "method": method, **m})
        roc_iris[sig] = {"y_true": y, "y_score": np.asarray(iris_scores[sig])[test_mask]}
        roc_rg[sig] = {"y_true": y, "y_score": np.asarray(rg_scores[sig])[test_mask]}

    table = metrics.summarize_runs(rows)
    table.to_csv(outdir / "fig1c_metrics.csv", index=False)
    print("\n" + table.to_string(index=False))

    # --- plots -------------------------------------------------------------
    plt = plotting.set_style()
    colors = config.palette()

    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    plotting.plot_roc(roc_iris, ax=ax)
    for sig, d in roc_rg.items():
        pts = metrics.curve_points(d["y_true"], d["y_score"])
        ax.plot(pts["fpr"], pts["tpr"], color=colors.get(sig, "k"),
                ls="--", lw=0.8, alpha=0.85)
    ax.set_title("In-sample test of IRIS accuracy")
    plotting.save(fig, outdir, "fig1c_roc")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    plotting.plot_pr(roc_iris, ax=ax, show_baseline=False)
    for sig, d in roc_rg.items():
        pts = metrics.curve_points(d["y_true"], d["y_score"])
        ax.plot(pts["recall"], pts["precision"], color=colors.get(sig, "k"),
                ls="--", lw=0.8, alpha=0.85)
    plotting.save(fig, outdir, "fig1c_precision_recall")
    plt.close(fig)

    provenance.record(
        ANALYSIS, outdir,
        inputs=[config.data_path("screens_ref_full")],
        params={"batches": MESC_BATCHES, "test_fraction": TEST_FRACTION,
                "seed": SEED, "retrained": args.retrain},
        results={"metrics": table.to_dict("records")},
    )
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
