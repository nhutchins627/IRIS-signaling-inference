"""Evaluation metrics and statistical tests used throughout the paper.

Kept deliberately close to the original scripts so regenerated numbers match
the published ones: ROC/PR from scikit-learn, F1 at the threshold that
maximises TPR-FPR separation, and the specific one/two-tailed tests named in
each figure legend.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)


def best_threshold(y_true, y_score) -> float:
    """Threshold maximising Youden's J (TPR - FPR).

    This is the rule the response-gene baseline uses to turn a continuous
    score into a binary call ("the threshold which maximally separates true
    positive and false positive according to the ROC curve").
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return float(thr[np.argmax(tpr - fpr)])


def classification_metrics(y_true, y_score, threshold: float | None = None
                           ) -> Dict[str, float]:
    """AUROC / AUPRC / F1 plus the threshold used, for one pathway."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    thr = best_threshold(y_true, y_score) if threshold is None else threshold
    y_pred = (y_score > thr).astype(int)
    return {
        "AUROC": float(roc_auc_score(y_true, y_score)),
        "AUPRC": float(average_precision_score(y_true, y_score)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(thr),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def random_baseline_f1(y_true) -> float:
    """Expected F1 of a classifier that always predicts positive.

    Used as the dashed "random classifier" reference in Fig. 2c/2d. For a
    predict-all-positive rule, precision == prevalence p and recall == 1,
    so F1 = 2p / (1 + p).
    """
    p = float(np.asarray(y_true).astype(int).mean())
    return 2 * p / (1 + p) if p > 0 else 0.0


def curve_points(y_true, y_score) -> Dict[str, np.ndarray]:
    """Raw ROC and PR curve coordinates, for plotting or source-data export."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}


def evaluate_predictions(df: pd.DataFrame, *, score_col: str = "proba",
                         truth_col: str = "ground_truth",
                         positive: str = "Stim") -> Dict[str, float]:
    """Metrics from a saved predictions table (as written by the runners)."""
    truth = df[truth_col]
    y = (truth == positive).astype(int) if truth.dtype == object else truth.astype(int)
    return classification_metrics(y, df[score_col])


def response_gene_score(adata, genes: Sequence[str], layer: str | None = None
                        ) -> np.ndarray:
    """Summed log-normalised expression of a pathway's response genes.

    This is the baseline of Supp. Fig. 1: p_s proportional to sum_i a_i * f_i,
    with unit weights. Genes absent from the object are skipped (dropout and
    cross-species mapping both cause misses).
    """
    present = [g for g in genes if g in adata.var_names]
    if not present:
        raise ValueError("None of the response genes are present in adata.var_names")
    X = adata[:, present].layers[layer] if layer else adata[:, present].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X).sum(axis=1).ravel()


# --- Statistical tests named in the figure legends -------------------------

def spearman_trend(x, y):
    """Two-tailed Spearman correlation (Fig. 3d/h/i trend significance)."""
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p": float(p)}


def mannwhitney_greater(a, b):
    """One-tailed Mann-Whitney U (alternative='greater').

    Used for marker enrichment and for ordering signal combinations along
    pseudotime (Fig. 3e/f).
    """
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(a, b, alternative="greater")
    return {"U": float(stat), "p": float(p)}


def fisher_enrichment(table, alternative: str = "greater"):
    """Fisher's exact test on a 2x2 contingency table (Fig. 4a, Supp. Fig. 19)."""
    from scipy.stats import fisher_exact
    odds, p = fisher_exact(np.asarray(table), alternative=alternative)
    return {"odds_ratio": float(odds), "p": float(p)}


def binomial_model_comparison(wins: int, n: int, p: float = 0.5):
    """One-sided binomial test that IRIS beats a competitor more often.

    Supp. Fig. 10b/c compares IRIS against each other model across all
    pathway x split pairs.
    """
    from scipy.stats import binomtest
    r = binomtest(int(wins), int(n), p, alternative="greater")
    return {"wins": int(wins), "n": int(n), "p": float(r.pvalue)}


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI (used for FPR bars in Supp. Fig. 11)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}


def summarize_runs(records: Iterable[Dict]) -> pd.DataFrame:
    """Tidy a list of per-run metric dicts into a DataFrame."""
    df = pd.DataFrame(list(records))
    front = [c for c in ("signal", "held_out", "split", "model") if c in df]
    return df[front + [c for c in df.columns if c not in front]]
