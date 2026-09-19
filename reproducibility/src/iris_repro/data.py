"""Dataset loading and preparation shared across all figure analyses."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import batch_name, data_path, load_config, signals

COUNTS_LAYER = "counts"


def _lazy_sc():
    import scanpy as sc  # imported lazily: heavy, and not needed for CSV-only work
    return sc


def ensure_counts_layer(adata, layer: str = COUNTS_LAYER):
    """Stash raw counts in ``.layers['counts']`` as scvi-tools expects.

    The screen h5ad files hold raw integer counts in ``.X``; every training
    script in the original analysis begins with this exact copy.
    """
    if layer not in adata.layers:
        adata.layers[layer] = adata.X.copy()
    return adata


def load_screens(path: str | Path | None = None, *, add_names: bool = True):
    """Load the combined mESC+hESC perturbation reference (38,307 cells)."""
    sc = _lazy_sc()
    adata = sc.read_h5ad(path or data_path("screens_ref_full"))
    ensure_counts_layer(adata)
    if add_names:
        annotate_screen_names(adata)
    return adata


def load_gastrulation(path: str | Path | None = None):
    """Load the gastrulation atlas object (atlas cells + screen reference)."""
    sc = _lazy_sc()
    adata = sc.read_h5ad(path or data_path("gastrulation_atlas"))
    ensure_counts_layer(adata)
    return adata


def annotate_screen_names(adata):
    """Add a human-readable ``screen`` column (mP_d1, hM_d4, ...)."""
    if "batch" in adata.obs:
        adata.obs["screen"] = pd.Categorical(
            [batch_name(b) for b in adata.obs["batch"]]
        )
    return adata


def mask_labels(adata, signal: str, held_out: Sequence[int],
                unlabeled: str | None = None):
    """Create the ``{signal}_class_masked`` column SCANVI trains against.

    Cells in ``held_out`` batches get the sentinel ``unknown`` label, so the
    semi-supervised classifier never sees their ground truth. This mirrors
    ``scanvi_manual.py`` from the original analysis.
    """
    if unlabeled is None:
        unlabeled = load_config()["setup_anndata"]["unlabeled_category"]
    src = f"{signal}_class"
    col = f"{src}_masked"
    if src not in adata.obs:
        raise KeyError(f"{src!r} not in adata.obs; have {list(adata.obs.columns)}")

    labels = adata.obs[src].astype("category")
    if unlabeled not in labels.cat.categories:
        labels = labels.cat.add_categories([unlabeled])
    held = np.isin(adata.obs["batch"].values, np.asarray(held_out))
    labels = labels.copy()
    labels[held] = unlabeled
    adata.obs[col] = labels
    return col


def split_batches(adata, train: Iterable[int], test: Iterable[int]):
    """Return ``(adata_train, adata_test)`` views selected by batch code."""
    train, test = list(train), list(test)
    tr = adata[np.isin(adata.obs["batch"].values, train)].copy()
    te = adata[np.isin(adata.obs["batch"].values, test)].copy()
    return tr, te


def binary_labels(adata, signal: str, positive: str = "Stim") -> np.ndarray:
    """Ground truth as 0/1, where 1 == pathway stimulated."""
    return (adata.obs[f"{signal}_class"] == positive).astype(int).values


def combination_code(adata, order: Sequence[str] | None = None,
                     suffix: str = "_class", positive: str = "Stim") -> pd.Series:
    """Collapse per-pathway calls into a bit string, e.g. ``"11000"``.

    Bit order follows ``signal_order`` in the config (TGF-beta, WNT, FGF, BMP, RA),
    matching the figure legends.
    """
    order = list(order or signals())
    cols = []
    for s in order:
        c = f"{s}{suffix}"
        if c not in adata.obs:
            raise KeyError(f"Missing {c!r} in adata.obs")
        cols.append((adata.obs[c] == positive).astype(int).astype(str).values)
    return pd.Series(["".join(bits) for bits in zip(*cols)], index=adata.obs_names)


def to_human_genes(adata, inplace: bool = False):
    """Convert mouse gene symbols to human via ``mousipy``.

    The Methods specify human symbols whenever mouse and human data are
    integrated, so response-gene lookups work across species.
    """
    try:
        from mousipy import translate
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mousipy is required for cross-species gene mapping "
            "(`pip install mousipy==0.1.5`)."
        ) from exc
    out = translate(adata if inplace else adata.copy())
    return out


def describe(adata) -> pd.DataFrame:
    """Per-screen summary: cell counts and stimulated fraction per pathway."""
    rows = []
    for b in sorted(pd.unique(adata.obs["batch"])):
        sub = adata.obs[adata.obs["batch"] == b]
        row = {"batch": b, "screen": batch_name(b), "n_cells": len(sub)}
        if "species" in sub:
            row["species"] = sub["species"].iloc[0]
        for s in signals():
            col = f"{s}_class"
            if col in sub:
                row[f"{s}_stim_frac"] = float((sub[col] == "Stim").mean())
        rows.append(row)
    return pd.DataFrame(rows)
