#!/usr/bin/env python3
"""Figure 2c/2d -- cross-screen and cross-cell-type generalization.

2c  Leave-one-screen-out cross-validation. Each of the five screens
    (mP_d1, mE_d2, hM_d4, hE_d8, hM_d7) is held out in turn; batch 2
    ("mixed") is excluded because it overlaps mP_d1 and mE_d2.

2d  Endoderm-to-mesoderm transfer. Train on the endodermal screens only
    (mE_d2 + hE_d8), test on the mesodermal ones (hM_d4, hM_d7). Endoderm
    and mesoderm are developmentally distinct and non-interconvertible, so
    this is the stringent cross-cell-type test.

Both panels read the per-cell prediction CSVs written by the original
runners; ``--retrain`` regenerates them from scratch.

Outputs
    fig2c_f1_scatter.*        IRIS F1 vs random-classifier F1, per split
    fig2c_pr_curves.*         PR curves averaged over splits
    fig2c_metrics.csv
    fig2d_f1_scatter.*  fig2d_pr_curves.*  fig2d_metrics.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from iris_repro import config, metrics, plotting, provenance

ANALYSIS = "fig2cd"
POSITIVE = "Stim"

# results CSVs are named e.g. clean_splits_Bmp_results_Bmp_in_5_out_3_results.csv
PATTERN = re.compile(
    r"(?P<signal>Bmp|Fgf|RA|Wnt|TgfB)_in_(?P<in>[\d_]+)_out_(?P<out>[\d_]+)_results\.csv$"
)


def find_result_csvs(prefix: str) -> List[Path]:
    """Locate prediction CSVs for one experiment family.

    Runs wrote either to the work root or to a ``clean_splits*/``
    subdirectory. Matching is anchored to the *file* name so that, e.g.,
    ``mesc_in_endo_out_*`` files sitting inside ``clean_splits2/`` are not
    picked up by the ``clean_splits`` family.

    ``clean_splits2`` contains later reruns of splits already present in
    ``clean_splits``. Pooling both would double-count the same held-out cells
    and inflate n, so when the same (signal, in, out) split appears twice we
    keep the newer ``clean_splits2`` copy only.
    """
    root = Path(config.load_config()["roots"]["work"])
    hits = [p for p in root.glob(f"{prefix}*_results.csv")]
    hits += [p for p in root.glob(f"*/{prefix}*_results.csv")
             if p.name.startswith(prefix)]

    # De-duplicate on the split identity encoded in the filename.
    best: dict[tuple, Path] = {}
    for p in sorted(set(hits)):
        m = PATTERN.search(p.name)
        if not m:
            continue
        key = (m.group("signal"), m.group("in"), m.group("out"))
        prev = best.get(key)
        # A path under ".../clean_splits2/" supersedes ".../clean_splits/".
        if prev is None or ("2" in p.parent.name and "2" not in prev.parent.name):
            best[key] = p
    return sorted(best.values())


def _held_out_mask(df: pd.DataFrame, out_batches: List[int],
                   path: Path) -> np.ndarray | None:
    """Boolean mask selecting the held-out rows of a runner CSV.

    Two CSV generations exist. Newer ones carry a ``batch`` column and we
    filter on it directly. Older ``clean_splits`` files do not, but the
    runner concatenated ``adata_in`` then ``adata_out`` before saving, so the
    held-out cells are exactly the trailing rows -- and we can verify that by
    checking the tail length equals the known size of those batches.
    """
    if "batch" in df.columns:
        return df["batch"].isin(out_batches).to_numpy()

    sizes = config.batch_table()
    n_out = sum(sizes[b]["n_cells"] for b in out_batches if b in sizes)
    if n_out == 0 or n_out >= len(df):
        print(f"  ! {path.name}: cannot infer held-out rows without a batch "
              f"column (expected {n_out} of {len(df)}), skipping")
        return None
    mask = np.zeros(len(df), dtype=bool)
    mask[-n_out:] = True
    return mask


def load_predictions(path: Path) -> pd.DataFrame | None:
    """Read a runner CSV and keep only held-out cells.

    The runners save predictions for every cell (train and test); only the
    held-out rows are a valid measurement of generalization.
    """
    m = PATTERN.search(path.name)
    if not m:
        return None
    df = pd.read_csv(path)
    out_batches = [int(b) for b in m.group("out").split("_") if b]

    score_col = next((c for c in ("scanvi_pred_Stim", "proba", "prediction_proba")
                      if c in df.columns), None)
    truth_col = next((c for c in ("ground_truth", f"{m.group('signal')}_class")
                      if c in df.columns), None)
    if score_col is None or truth_col is None:
        return None

    mask = _held_out_mask(df, out_batches, path)
    if mask is None:
        return None
    held = df.loc[mask].copy()
    if held.empty:
        return None
    held["signal"] = m.group("signal")
    held["held_out"] = "+".join(config.batch_name(b) for b in out_batches)
    held["y_true"] = (held[truth_col] == POSITIVE).astype(int)
    held["y_score"] = held[score_col].astype(float)
    held["source"] = path.name
    return held[["signal", "held_out", "y_true", "y_score", "source"]]


def _parse_batches(group: str) -> tuple[int, ...]:
    return tuple(int(b) for b in group.split("_") if b)


def collect(prefix: str, *, train: tuple[int, ...] | None = None,
            single_holdout_only: bool = False) -> pd.DataFrame:
    """Gather held-out predictions for a specific set of splits.

    ``clean_splits*/`` mixes several experiments in one directory, so a
    filename prefix alone does not identify a panel. Selection is therefore
    made on the *split identity* encoded in the name:

      Fig. 2c -- ``single_holdout_only``: exactly one held-out screen.
      Fig. 2d -- ``train=(3, 6)``: trained on the two endodermal screens.
    """
    paths = []
    for p in find_result_csvs(prefix):
        m = PATTERN.search(p.name)
        if not m:
            continue
        if single_holdout_only and len(_parse_batches(m.group("out"))) != 1:
            continue
        if train is not None and _parse_batches(m.group("in")) != tuple(train):
            continue
        paths.append(p)

    frames = [d for p in paths if (d := load_predictions(p)) is not None]
    if not frames:
        raise SystemExit(
            f"No usable prediction CSVs found for prefix {prefix!r} "
            f"(train={train}, single_holdout={single_holdout_only}) under "
            f"{config.load_config()['roots']['work']}.\n"
            "Run with --retrain, or check config/paths.yaml."
        )
    return pd.concat(frames, ignore_index=True)


def score_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict]:
    """Per (pathway, split) metrics plus pooled curves for plotting."""
    rows, curves = [], {}
    for (sig, held), sub in df.groupby(["signal", "held_out"]):
        if sub["y_true"].nunique() < 2:
            print(f"  ! {sig} / {held}: single-class held-out set, skipping")
            continue
        m = metrics.classification_metrics(sub["y_true"], sub["y_score"])
        rows.append({"signal": sig, "held_out": held,
                     "baseline_F1": metrics.random_baseline_f1(sub["y_true"]), **m})
    for sig, sub in df.groupby("signal"):
        if sub["y_true"].nunique() >= 2:
            curves[sig] = {"y_true": sub["y_true"].values,
                           "y_score": sub["y_score"].values}
    return metrics.summarize_runs(rows), curves


def make_panel(df: pd.DataFrame, outdir: Path, tag: str, title: str):
    table, curves = score_splits(df)
    table.to_csv(outdir / f"{tag}_metrics.csv", index=False)
    print(f"\n=== {title} ===")
    print(table.to_string(index=False))
    print(table.groupby("signal")[["AUROC", "AUPRC", "F1"]].mean().round(3).to_string())

    plt = plotting.set_style()
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    plotting.plot_f1_vs_baseline(table, ax=ax)
    ax.set_title(title)
    plotting.save(fig, outdir, f"{tag}_f1_scatter")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    plotting.plot_pr(curves, ax=ax)
    ax.set_title(title)
    plotting.save(fig, outdir, f"{tag}_pr_curves")
    plt.close(fig)
    return table


def retrain(which: str, outdir: Path):
    """Refit models for a panel (GPU; hours)."""
    from iris_repro.data import load_screens
    from iris_repro.model import train_all_signals

    adata = load_screens()
    splits = config.load_config()["splits"]
    if which == "2c":
        for held in splits["cross_validation"]["held_out"]:
            print(f"--- holding out {config.batch_name(held)} ---")
            train_all_signals(adata, [held], outdir=outdir / "retrained_cv")
    else:
        s = splits["endoderm_to_mesoderm"]
        print(f"--- train {s['train']} -> test {s['test']} ---")
        train_all_signals(adata, s["test"], outdir=outdir / "retrained_endo2meso")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", choices=["2c", "2d", "both"], default="both")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    outdir = config.output_dir("fig2")
    provenance.check_environment()
    tables = {}

    if args.panel in ("2c", "both"):
        if args.retrain:
            retrain("2c", outdir)
        df = collect("clean_splits", single_holdout_only=True)
        tables["2c"] = make_panel(df, outdir, "fig2c",
                                  "Cross validation (single-screen heldout)")

    if args.panel in ("2d", "both"):
        if args.retrain:
            retrain("2d", outdir)
        # Endoderm-only training set: mE_d2 (3) + hE_d8 (6).
        endo_train = tuple(config.load_config()["splits"]["endoderm_to_mesoderm"]["train"])
        df = collect("clean_splits", train=endo_train)
        tables["2d"] = make_panel(df, outdir, "fig2d",
                                  "Cross-cell-type (endoderm to mesoderm)")

    provenance.record(
        ANALYSIS, outdir,
        inputs=[config.data_path("screens_ref_full")],
        params={"splits": config.load_config()["splits"], "retrained": args.retrain},
        results={k: v.to_dict("records") for k, v in tables.items()},
    )
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
