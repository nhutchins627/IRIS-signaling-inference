#!/usr/bin/env python3
"""Figure 2e / Supp. Fig. 10c -- IRIS vs classical ML models.

Compares IRIS against Elastic Net, two SVM parameterisations, and Random
Forest, all tuned with the same hyperparameter search (Supp. Fig. 10a).
Statistics follow Supp. Fig. 10b/c: a one-sided binomial test over all
pathway x split pairs, asking how often IRIS beats each competitor.

Two regimes, selected with ``--regime``:

  cross-validation  (default, = Supp. Fig. 10c)
      Leave-one-screen-out. Reads the ``benchmark_batch<N>_*.csv`` files
      written by ``benchmarking-linear-models-cv-parallel.py``, which loops
      over batches 1,2,3,5,6,7 holding out one at a time.

  cross-species     (= Fig. 2e)
      Train on the mouse screens, test on human (and the reverse). This is
      the data-limited regime where the paper reports IRIS keeping its
      advantage. There is no saved sklearn benchmark for this split in the
      working directory, so it must be computed with ``--compute``; the
      cached CSV is reused on later runs.

Outputs
    fig2e_benchmark_bars.*     metric by model, grouped by pathway
    fig2e_benchmark.csv        tidy per-model per-pathway metrics
    fig2e_binomial_tests.csv   IRIS vs each competitor
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from iris_repro import config, metrics, plotting, provenance

ANALYSIS = "fig2e"
MODEL_ORDER = ["EN", "SVM-1", "SVM-2", "RF", "IRIS"]
MODEL_COLORS = {"EN": "#bdbdbd", "SVM-1": "#969696", "SVM-2": "#737373",
                "RF": "#525252", "IRIS": None}  # IRIS uses the pathway colour


"""Column names in the runner CSVs look like ``f1_out-linsvc-c1`` or
``out-f1-svm-rbf``: a train/test scope, a metric, and a model tag, in either
order depending on which runner wrote the file."""
_COL = re.compile(
    r"^(?:(?P<m1>f1|auroc|auprc)_(?P<s1>in|out)|(?P<s2>in|out)-(?P<m2>f1|auroc|auprc))"
    r"[-_](?P<model>.+)$"
)


def _parse_column(col: str) -> tuple[str, str, str] | None:
    m = _COL.match(col.strip())
    if not m:
        return None
    scope = m.group("s1") or m.group("s2")
    metric = (m.group("m1") or m.group("m2")).upper()
    return scope, metric, m.group("model")


def load_benchmark_tables(scope: str = "out") -> pd.DataFrame:
    """Load per-model benchmark CSVs written by the classical-ML runners.

    Layout: rows are ``<Signal>_class``, columns encode scope/metric/model
    (e.g. ``f1_out-elasticnet``). ``scope='out'`` selects held-out
    performance, which is what the figure reports.

    Each ``benchmark_batch<N>_*.csv`` is one held-out screen; the batch number
    in the filename becomes the split label.
    """
    root = Path(config.load_config()["roots"]["work"])
    paths = sorted(root.glob("benchmark_batch*_*.csv")) or \
        sorted(root.glob("benchmark_*.csv"))
    if not paths:
        raise SystemExit(
            f"No benchmark CSVs found in {root}. Expected "
            "benchmark_batch<N>_<timestamp>.csv from "
            "benchmarking-linear-models-cv-parallel.py."
        )

    # Keep only the newest file per held-out batch.
    newest: dict[str, Path] = {}
    for p in paths:
        m = re.search(r"benchmark_batch(\d+)_", p.name)
        key = m.group(1) if m else p.stem
        if key not in newest or p.stat().st_mtime > newest[key].stat().st_mtime:
            newest[key] = p

    rows = []
    for key, p in sorted(newest.items()):
        df = pd.read_csv(p, index_col=0)
        # `fit_seconds` repeats per model and carries no metric.
        df = df.loc[:, ~df.columns.str.startswith("fit_seconds")]
        split = config.batch_name(int(key)) if key.isdigit() else key
        for col in df.columns:
            parsed = _parse_column(col)
            if parsed is None:
                continue
            col_scope, metric, model = parsed
            if col_scope != scope:
                continue
            for sig_label, val in df[col].items():
                if pd.isna(val):
                    continue
                rows.append({
                    "signal": str(sig_label).replace("_class", ""),
                    "split": split, "model": model, metric: float(val),
                })

    if not rows:
        raise SystemExit(f"No '{scope}' metric columns parsed from {paths[0].name}.")

    long = pd.DataFrame(rows)
    return (long.groupby(["signal", "split", "model"], as_index=False)
                .first()
                .reset_index(drop=True))


def compute_cross_species_baselines(cache: Path, scope: str = "out") -> pd.DataFrame:
    """Fit the classical models on the mouse/human cross-species splits.

    Mirrors ``benchmarking-linear-models-cv-parallel.py``: the same estimators
    and the same design matrix (expression hstacked with one-hot batch,
    celltype and species), but split by species instead of by batch.
    """
    from scipy.sparse import hstack
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import Normalizer, OneHotEncoder
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    from iris_repro.data import load_screens

    estimators = {
        "SVM-1": LinearSVC(C=1, dual="auto", random_state=42),
        "SVM-2": LinearSVC(C=31.62, dual="auto", random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "EN": SGDClassifier(loss="log_loss", penalty="elasticnet",
                            l1_ratio=0.5, random_state=42),
    }
    splits = config.load_config()["splits"]["cross_species"]
    adata = load_screens()

    # Resume support: this loop takes ~1 h and each (split, signal, model) fit
    # is independent, so results are appended to a partial file as they land.
    # A kill (SIGTERM, wall-clock limit, preemption) then costs only the fit in
    # flight rather than the whole run.
    partial = cache.with_suffix(".partial.csv")
    done: set[tuple[str, str, str]] = set()
    rows = []
    if partial.exists():
        prev = pd.read_csv(partial)
        rows = prev.to_dict("records")
        done = {(r["split"], r["signal"], r["model"]) for r in rows}
        print(f"  resuming: {len(done)} fits already complete in {partial.name}")

    def _checkpoint():
        partial.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(partial, index=False)

    for split_name, spec in splits.items():
        tr = adata[np.isin(adata.obs["batch"].values, spec["train"])]
        te = adata[np.isin(adata.obs["batch"].values, spec["test"])]
        feats = [c for c in ("batch", "celltype", "species") if c in adata.obs]
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        X_tr = hstack([tr.X, enc.fit_transform(tr.obs[feats])])
        X_te = hstack([te.X, enc.transform(te.obs[feats])])

        for sig in config.signals():
            y_tr = (tr.obs[f"{sig}_class"] == "Stim").astype(int).values
            y_te = (te.obs[f"{sig}_class"] == "Stim").astype(int).values
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            for name, est in estimators.items():
                if (split_name, sig, name) in done:
                    continue
                clf = make_pipeline(Normalizer(), est)
                clf.fit(X_tr, y_tr)
                src = X_te if scope == "out" else X_tr
                truth = y_te if scope == "out" else y_tr
                score = (clf.decision_function(src)
                         if hasattr(clf[-1], "decision_function")
                         else clf.predict_proba(src)[:, 1])
                m = metrics.classification_metrics(truth, score)
                rows.append({"signal": sig, "split": split_name, "model": name,
                             "F1": m["F1"], "AUROC": m["AUROC"], "AUPRC": m["AUPRC"]})
                print(f"  {split_name:16s} {sig:5s} {name:6s} "
                      f"F1={m['F1']:.3f} AUROC={m['AUROC']:.3f}", flush=True)
                _checkpoint()

    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    partial.unlink(missing_ok=True)
    print(f"  cached -> {cache}")
    return df


def load_cross_species_iris(scope: str = "out") -> pd.DataFrame:
    """IRIS metrics on the cross-species splits, from its prediction CSVs.

    The prediction CSVs are labelled by *held-out screen* (``hM_d4``,
    ``mP_d1+mixed+mE_d2``, ...), while the sklearn baselines are labelled by
    *direction* (``mouse_to_human`` / ``human_to_mouse``). Those label spaces
    do not intersect, so the two must be reconciled before any paired
    comparison -- otherwise a join silently falls back to matching on pathway
    alone and compares IRIS on one split against a baseline on another.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fig2cd_generalization import collect, score_splits

    preds = collect("cross_species")
    table, _ = score_splits(preds)
    out = table.rename(columns={"held_out": "split"})[
        ["signal", "split", "AUROC", "AUPRC", "F1"]].copy()

    # Held-out screens are human => the model trained on mouse, and vice versa.
    batches = config.batch_table()
    species_of = {v["name"]: v["species"] for v in batches.values()}

    def _direction(label: str) -> str | None:
        species = {species_of.get(part) for part in str(label).split("+")}
        species.discard(None)
        if species == {"human"}:
            return "mouse_to_human"   # held out human => trained on mouse
        if species == {"mouse"}:
            return "human_to_mouse"
        return None                   # mixed-species holdout: not a clean split

    out["split"] = [_direction(s) for s in out["split"]]
    dropped = out["split"].isna().sum()
    if dropped:
        print(f"  ! dropped {dropped} IRIS rows whose holdout spans both species")
    out = out[out["split"].notna()].copy()

    # One row per (pathway, direction): average over the screens on that side.
    out = (out.groupby(["signal", "split"], as_index=False)[["AUROC", "AUPRC", "F1"]]
              .mean())
    out["model"] = "IRIS"
    return out


def attach_iris(df: pd.DataFrame, scope: str = "out") -> pd.DataFrame:
    """Add IRIS's own held-out scores so the comparison is like-for-like.

    IRIS results come from the SCANVI prediction CSVs, not the sklearn
    benchmark files, so they are computed here and appended.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from fig2cd_generalization import collect, score_splits
    except ImportError:
        print("  ! could not import fig2cd_generalization; IRIS rows omitted")
        return df

    try:
        preds = collect("clean_splits", single_holdout_only=True)
    except SystemExit as exc:
        print(f"  ! IRIS predictions unavailable ({exc}); IRIS rows omitted")
        return df

    table, _ = score_splits(preds)
    iris = table.rename(columns={"held_out": "split"})[
        ["signal", "split", "AUROC", "AUPRC", "F1"]].copy()
    iris["model"] = "IRIS"
    return pd.concat([df, iris], ignore_index=True)


def normalize_model_names(df: pd.DataFrame, keep_all: bool = False) -> pd.DataFrame:
    """Map runner model tags onto the four display names used in Fig. 2e.

    The runners swept far more configurations than the figure shows. SVM1 and
    SVM2 are "two SVM models that differed in hyperparameters" (Fig. 2e
    legend), which here are the two linear-SVC regularisation strengths.
    Anything unmapped is dropped unless ``keep_all``.
    """
    lookup = {
        "elasticnet": "EN",
        "sgd-log_loss-elasticnet": "EN",
        "linsvc-c1": "SVM-1",
        "svm-linear": "SVM-1",
        "linsvc-c31.62": "SVM-2",
        "svm-rbf": "SVM-2",
        "rf-fast": "RF",
        "rf-gini": "RF",
        "iris": "IRIS",
    }
    df = df.copy()
    key = df["model"].astype(str).str.strip().str.lower()
    mapped = [lookup.get(k, m if keep_all else None)
              for k, m in zip(key, df["model"])]
    df["model"] = mapped
    return df[df["model"].notna()].reset_index(drop=True)


def binomial_vs_iris(df: pd.DataFrame, metric: str = "F1") -> pd.DataFrame:
    """How often does IRIS beat each competitor, across pathway x split?"""
    if "IRIS" not in set(df["model"]):
        print("  ! IRIS rows absent from the benchmark table; skipping tests")
        return pd.DataFrame()

    # Pair on pathway AND split. Dropping "split" here would silently compare
    # IRIS on one split against a competitor on another.
    idx = [c for c in ("signal", "split") if c in df.columns]
    if "split" not in idx:
        raise ValueError("benchmark table has no 'split' column; cannot pair")
    iris = df[df["model"] == "IRIS"].set_index(idx)[metric]
    rows = []
    for model, sub in df[df["model"] != "IRIS"].groupby("model"):
        other = sub.set_index(idx)[metric]
        common = iris.index.intersection(other.index)
        if len(common) == 0:
            print(f"  ! IRIS and {model} share no (pathway, split) pairs -- "
                  f"skipping. IRIS splits={sorted({i[1] for i in iris.index})}, "
                  f"{model} splits={sorted({i[1] for i in other.index})}")
            continue
        wins = int((iris.loc[common] > other.loc[common]).sum())
        rows.append({"comparison": f"IRIS vs {model}", "metric": metric,
                     **metrics.binomial_model_comparison(wins, len(common))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="F1", choices=["F1", "AUROC", "AUPRC"])
    ap.add_argument("--scope", default="out", choices=["out", "in"],
                    help="'out' = held-out performance (the published panel)")
    ap.add_argument("--regime", default="cross-validation",
                    choices=["cross-validation", "cross-species"],
                    help="cross-species reproduces Fig. 2e; "
                         "cross-validation reproduces Supp. Fig. 10c")
    ap.add_argument("--compute", action="store_true",
                    help="fit the sklearn baselines instead of using the cache "
                         "(required once for --regime cross-species)")
    ap.add_argument("--keep-all-models", action="store_true",
                    help="keep every swept configuration, not just the four shown")
    args = ap.parse_args()

    outdir = config.output_dir("fig2")
    provenance.check_environment()

    if args.regime == "cross-species":
        # Cache lives outside outputs/ so that clearing regenerated figures
        # never destroys an hour of CPU work.
        cache = (Path(config.load_config()["roots"]["outputs"]).parent
                 / "cache" / f"fig2e_cross_species_baselines_{args.scope}.csv")
        if args.compute or not cache.exists():
            if not args.compute:
                raise SystemExit(
                    f"No cached cross-species baselines at {cache}.\n"
                    "Run once with --compute (CPU, ~30-60 min) to generate them, "
                    "or use --regime cross-validation, which reads the saved "
                    "benchmark_batch*.csv files."
                )
            print("Fitting cross-species baselines ...")
            df = compute_cross_species_baselines(cache, args.scope)
        else:
            print(f"Loading cached cross-species baselines from {cache}")
            df = pd.read_csv(cache)
        df = pd.concat([df, load_cross_species_iris(args.scope)], ignore_index=True)
        title = "Model benchmarking (cross-species)"
    else:
        df = normalize_model_names(load_benchmark_tables(args.scope),
                                   keep_all=args.keep_all_models)
        df = attach_iris(df, args.scope)
        title = "Model benchmarking (cross-validation)"
    # Average over held-out splits for the summary bars.
    summary = (df.groupby(["model", "signal"], as_index=False)[args.metric]
                 .mean())
    df.to_csv(outdir / "fig2e_benchmark.csv", index=False)
    print(summary.pivot(index="signal", columns="model",
                        values=args.metric).round(3).to_string())

    tests = binomial_vs_iris(df, args.metric)
    if not tests.empty:
        tests.to_csv(outdir / "fig2e_binomial_tests.csv", index=False)
        print("\n" + tests.to_string(index=False))

    # --- grouped bar chart -------------------------------------------------
    plt = plotting.set_style()
    colors = config.palette()
    signals = [s for s in config.signals() if s in set(summary["signal"])]
    models = [m for m in MODEL_ORDER if m in set(summary["model"])]
    models += sorted(set(summary["model"]) - set(MODEL_ORDER))

    fig, ax = plt.subplots(figsize=(3.6, 1.9))
    width = 0.8 / max(len(models), 1)
    for j, model in enumerate(models):
        vals, xs = [], []
        for i, sig in enumerate(signals):
            row = summary[(summary["model"] == model) & (summary["signal"] == sig)]
            vals.append(float(row[args.metric].iloc[0]) if len(row) else np.nan)
            xs.append(i + j * width - 0.4 + width / 2)
        # IRIS is drawn in the pathway colour to stand out from the greys.
        cols = ([colors.get(s, "k") for s in signals] if model == "IRIS"
                else MODEL_COLORS.get(model, "#999999"))
        ax.bar(xs, vals, width=width * 0.9, color=cols,
               label=model, edgecolor="none")

    ax.set_xticks(range(len(signals)))
    ax.set_xticklabels([config.display_name(s) for s in signals])
    ax.set_ylabel(args.metric)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=len(models), fontsize=5,
              loc="upper center", bbox_to_anchor=(0.5, 1.32))
    plotting.save(fig, outdir, "fig2e_benchmark_bars")
    plt.close(fig)

    provenance.record(
        ANALYSIS, outdir,
        params={"metric": args.metric, "scope": args.scope,
                "regime": args.regime, "computed": args.compute},
        results={"benchmark": df.to_dict("records"),
                 "binomial": tests.to_dict("records") if not tests.empty else []},
    )
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
