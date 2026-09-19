"""Shared plotting helpers, styled to match the published figures.

Figures are written as both SVG (vector, for assembly in Illustrator) and PNG,
which is what the original analysis produced.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import display_name, palette
from .metrics import curve_points, random_baseline_f1


def set_style():
    """Apply the figure style used throughout the paper."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,   # keep text editable in Illustrator
        "ps.fonttype": 42,
    })
    return plt


def save(fig, outdir: str | Path, name: str, formats: Sequence[str] = ("svg", "png")):
    """Save a figure in several formats; returns the paths written."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        p = outdir / f"{name}.{ext}"
        fig.savefig(p)
        written.append(p)
    return written


def plot_roc(results: Dict[str, Dict[str, np.ndarray]], ax=None,
             show_diagonal: bool = True, title: str | None = None):
    """Overlay per-pathway ROC curves.

    ``results`` maps signal -> {'y_true', 'y_score'}.
    """
    plt = set_style()
    ax = ax or plt.subplots(figsize=(2.0, 2.0))[1]
    colors = palette()
    for sig, d in results.items():
        pts = curve_points(d["y_true"], d["y_score"])
        ax.plot(pts["fpr"], pts["tpr"], color=colors.get(sig, "k"),
                label=display_name(sig))
    if show_diagonal:
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    return ax


def plot_pr(results: Dict[str, Dict[str, np.ndarray]], ax=None,
            show_baseline: bool = True, title: str | None = None):
    """Overlay per-pathway precision-recall curves.

    The dashed horizontal line per pathway is the random-classifier
    precision, i.e. the positive-class prevalence in that test set.
    """
    plt = set_style()
    ax = ax or plt.subplots(figsize=(2.0, 2.0))[1]
    colors = palette()
    for sig, d in results.items():
        pts = curve_points(d["y_true"], d["y_score"])
        c = colors.get(sig, "k")
        ax.plot(pts["recall"], pts["precision"], color=c, label=display_name(sig))
        if show_baseline:
            ax.axhline(np.mean(np.asarray(d["y_true"]).astype(int)),
                       color=c, ls="--", lw=0.5, alpha=0.7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="lower left")
    return ax


def plot_f1_vs_baseline(df: pd.DataFrame, ax=None, signal_col: str = "signal",
                        f1_col: str = "F1", baseline_col: str = "baseline_F1"):
    """IRIS F1 against the random-classifier F1, one dot per split.

    This is the scatter in Fig. 2c/2d (left); points above the diagonal
    are splits where IRIS beats chance.
    """
    plt = set_style()
    ax = ax or plt.subplots(figsize=(2.0, 2.0))[1]
    colors = palette()
    for sig, sub in df.groupby(signal_col):
        ax.scatter(sub[baseline_col], sub[f1_col], s=10,
                   color=colors.get(sig, "k"), label=display_name(sig),
                   zorder=3, edgecolors="none")
    lims = (0, 1)
    ax.plot(lims, lims, ls=":", c="grey", lw=0.6, zorder=1)
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel("Baseline F1 (random classifier)")
    ax.set_ylabel("IRIS F1")
    ax.legend(frameon=False, fontsize=5, loc="lower right")
    return ax


def plot_trend(x, y, ax=None, color: str = "k", label: str | None = None,
               n_bins: int = 35, show_sem: bool = True):
    """Mean +/- SEM of ``y`` in equal-width bins of ``x``.

    Matches the pseudotime/pseudospace panels (Fig. 3d/h/i), which the
    Methods describe as discretised into 35 bins.
    """
    plt = set_style()
    ax = ax or plt.subplots(figsize=(2.4, 1.6))[1]
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    edges = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

    centers, means, sems = [], [], []
    for b in range(n_bins):
        vals = y[idx == b]
        if vals.size == 0:
            continue
        centers.append((edges[b] + edges[b + 1]) / 2)
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0)

    centers = np.array(centers); means = np.array(means); sems = np.array(sems)
    ax.plot(centers, means, color=color, label=label)
    if show_sem:
        ax.fill_between(centers, means - sems, means + sems, color=color,
                        alpha=0.25, lw=0)
    return ax


def annotate_significance(ax, text: str, xy=(0.98, 0.95), **kw):
    """Place a p-value / rho annotation in axes coordinates."""
    ax.text(*xy, text, transform=ax.transAxes, ha="right", va="top", **kw)
    return ax


def combination_frequency_plot(codes: pd.Series, pseudotime: np.ndarray,
                               top_n: int = 8, n_bins: int = 35, ax=None):
    """Stacked frequency of signal combinations along pseudotime (Fig. 3e/f)."""
    plt = set_style()
    ax = ax or plt.subplots(figsize=(3.0, 1.4))[1]
    keep = codes.value_counts().head(top_n).index
    x = np.asarray(pseudotime, dtype=float)
    edges = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2

    freq = {}
    for code in keep:
        m = (codes == code).values
        freq[code] = np.array([
            m[idx == b].mean() if (idx == b).sum() else 0.0 for b in range(n_bins)
        ])
    ax.stackplot(centers, *freq.values(), labels=list(freq.keys()))
    ax.set_xlabel("Diffusion pseudotime")
    ax.set_ylabel("Frequency")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=5, ncol=2, loc="upper right")
    return ax
