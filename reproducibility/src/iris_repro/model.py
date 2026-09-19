"""IRIS model: train / load / predict.

IRIS is a CVAE-based semi-supervised classifier built on the scANVI
architecture from scvi-tools. Training is two-stage, exactly as in the
original scripts:

  1. Fit an SCVI model (ZINB likelihood) over all cells, held-out cells
     included, with no access to signaling labels -- this learns a
     batch-corrected latent space.
  2. Initialise SCANVI from that model and jointly refine the latent space
     with a feed-forward classifier, using labels only for training cells.

Held-out cells carry the ``unknown`` label, so their ground truth is never
seen -- the property that makes the held-out screen a genuine test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from .config import architecture, load_config, training_params
from .data import COUNTS_LAYER, ensure_counts_layer, mask_labels

# Sentinel label for cells whose ground truth is hidden from the classifier.
UNLABELED = load_config()["setup_anndata"]["unlabeled_category"]


def set_seed(seed: int = 42, deterministic: bool = True):
    """Seed every RNG the training path touches.

    Note: full bit-for-bit determinism is not guaranteed on GPU even so;
    the original analysis reported variance across 10 random initialisations
    (Supp. Fig. 10c).
    """
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class IRISModel:
    """Thin, reproducible wrapper around the SCVI -> SCANVI pipeline."""

    signal: str
    n_hidden: int = 0
    n_latent: int = 0
    n_layers: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    scanvae: Any = None
    label_key: str = ""

    def __post_init__(self):
        arch = architecture(self.signal)
        tp = training_params()
        self.n_hidden = self.n_hidden or arch["n_hidden"]
        self.n_latent = self.n_latent or arch["n_latent"]
        self.n_layers = self.n_layers or tp["n_layers"]
        self.params = {**tp, **self.params}

    # -- setup -------------------------------------------------------------
    def _setup(self, adata, label_key: str):
        import scvi
        cfg_keys = ["batch", "celltype", "species"]
        covariates = [k for k in cfg_keys if k in adata.obs]
        scvi.model.SCVI.setup_anndata(
            adata,
            layer=COUNTS_LAYER,
            batch_key="batch",
            labels_key=label_key,
            categorical_covariate_keys=covariates,
        )

    # -- training ----------------------------------------------------------
    def fit(self, adata, held_out: Sequence[int], *, verbose: bool = True):
        """Train on ``adata``, masking labels for cells in ``held_out``."""
        import scvi

        set_seed(self.params["random_seed"])
        ensure_counts_layer(adata)
        self.label_key = mask_labels(adata, self.signal, held_out)
        self._setup(adata, self.label_key)

        if verbose:
            print(f"[{self.signal}] SCVI  hidden={self.n_hidden} "
                  f"latent={self.n_latent} layers={self.n_layers}")
        vae = scvi.model.SCVI(
            adata,
            n_layers=self.n_layers,
            n_latent=self.n_latent,
            n_hidden=self.n_hidden,
            gene_likelihood=self.params["gene_likelihood"],
        )
        vae.train(max_epochs=self.params["scvi_epochs"])

        if verbose:
            print(f"[{self.signal}] SCANVI ({self.params['scanvi_epochs']} epochs)")
        self.scanvae = scvi.model.SCANVI.from_scvi_model(
            vae,
            labels_key=self.label_key,
            unlabeled_category=UNLABELED,
        )
        self.scanvae.train(max_epochs=self.params["scanvi_epochs"])
        return self

    # -- inference ---------------------------------------------------------
    def predict_proba(self, adata=None) -> pd.Series:
        """P(pathway active) per cell, from the classifier's sigmoid layer."""
        if self.scanvae is None:
            raise RuntimeError("Model not trained/loaded; call fit() or load().")
        soft = self.scanvae.predict(adata=adata, soft=True)
        return soft["Stim"]

    def predict(self, adata=None, threshold: float | None = None) -> pd.Series:
        """Binary call at the 0.5 threshold (Supp. Fig. 3)."""
        thr = self.params["classifier_threshold"] if threshold is None else threshold
        return (self.predict_proba(adata) > thr).astype(int)

    def latent(self, adata=None) -> np.ndarray:
        return self.scanvae.get_latent_representation(adata=adata)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path, overwrite: bool = True):
        self.scanvae.save(str(path), overwrite=overwrite)
        return Path(path)

    @classmethod
    def load(cls, path: str | Path, adata, signal: str,
             label_key: str | None = None) -> "IRISModel":
        """Load a pretrained checkpoint directory saved by scvi-tools.

        ``adata`` must be set up the same way it was at training time, so we
        re-register it before handing it to scvi-tools.
        """
        import scvi
        obj = cls(signal=signal)
        ensure_counts_layer(adata)
        obj.label_key = label_key or f"{signal}_class_masked"
        if obj.label_key not in adata.obs:
            # A checkpoint may be applied to data with no ground truth at all
            # (e.g. an in vivo atlas), so every cell is unlabeled.
            adata.obs[obj.label_key] = pd.Categorical(
                [UNLABELED] * adata.n_obs, categories=[UNLABELED]
            )
        obj._setup(adata, obj.label_key)
        obj.scanvae = scvi.model.SCANVI.load(str(path), adata=adata)
        return obj


def train_all_signals(adata, held_out: Sequence[int],
                      signal_list: Sequence[str] | None = None,
                      outdir: str | Path | None = None) -> Dict[str, pd.DataFrame]:
    """Fit one independent model per pathway and collect predictions.

    Pathways are modelled independently; a cell's combinatorial code is the
    concatenation of the per-pathway calls (see the Methods).
    """
    from .config import signals as _signals
    signal_list = list(signal_list or _signals())
    out: Dict[str, pd.DataFrame] = {}

    for sig in signal_list:
        model = IRISModel(sig).fit(adata.copy(), held_out)
        proba = model.predict_proba()
        df = pd.DataFrame({
            "batch": adata.obs["batch"].values,
            "ground_truth": adata.obs[f"{sig}_class"].values,
            "proba": proba.values,
            "prediction": (proba.values > model.params["classifier_threshold"]).astype(int),
        }, index=adata.obs_names)
        out[sig] = df
        if outdir:
            outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
            tag = "_".join(map(str, held_out))
            df.to_csv(outdir / f"{sig}_out_{tag}_predictions.csv")
            model.save(outdir / f"{sig}_out_{tag}_scanvi_model.pt")
    return out
