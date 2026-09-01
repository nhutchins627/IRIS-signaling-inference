#!/usr/bin/env python3
"""Guard the conventions that are easy to get silently wrong.

These are the assumptions the figure scripts depend on. Each one was a real
bug or a real ambiguity discovered while rebuilding the analysis, so a
regression here would quietly produce wrong numbers rather than an error.

    PYTHONPATH=src python -m pytest tests/ -v
    PYTHONPATH=src python tests/test_conventions.py     # no pytest needed
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "figures"))

from iris_repro import config, metrics


# --- configuration ---------------------------------------------------------

def test_architectures_match_publication():
    """Supp. Fig. 5c values; also encoded in every checkpoint directory name."""
    expected = {"Wnt": (64, 30), "TgfB": (1024, 30), "Fgf": (128, 70),
                "RA": (256, 70), "Bmp": (256, 70)}
    for sig, (hidden, latent) in expected.items():
        a = config.architecture(sig)
        assert a["n_hidden"] == hidden, f"{sig} hidden {a['n_hidden']} != {hidden}"
        assert a["n_latent"] == latent, f"{sig} latent {a['n_latent']} != {latent}"


def test_shallow_architecture():
    """The hyperparameter screen selected single-layer models."""
    assert config.training_params()["n_layers"] == 1


def test_signal_bit_order():
    """Combination strings read TGF-beta, WNT, FGF, BMP, RA (Fig. 3c legend)."""
    assert config.signals() == ["TgfB", "Wnt", "Fgf", "Bmp", "RA"]


def test_batch_mapping():
    """Batch codes must match the cell counts reported in the Methods."""
    expected = {1: ("mP_d1", 4751), 2: ("mixed", 175), 3: ("mE_d2", 5224),
                5: ("hM_d4", 6379), 6: ("hE_d8", 9921), 7: ("hM_d7", 11857)}
    table = config.batch_table()
    for b, (name, n) in expected.items():
        assert table[b]["name"] == name
        assert table[b]["n_cells"] == n
    assert 4 not in table, "there is no batch 4"


def test_batch_two_excluded_from_cv():
    """Batch 2 overlaps mP_d1 and mE_d2, so it cannot be a clean holdout."""
    assert 2 not in config.load_config()["splits"]["cross_validation"]["held_out"]


def test_endoderm_to_mesoderm_split_is_disjoint():
    """Fig. 2d is only meaningful if train and test share no batch."""
    s = config.load_config()["splits"]["endoderm_to_mesoderm"]
    assert set(s["train"]).isdisjoint(s["test"])
    assert all(config.batch_table()[b]["lineage"] == "endoderm" for b in s["train"])
    assert all(config.batch_table()[b]["lineage"] == "mesoderm" for b in s["test"])


def test_legacy_path_rewriting():
    """The original working directory has been archived; old paths must remap."""
    legacy = "/lab/tambora_li/Nicholas/figures-paper/fig5v2/fig5v2_data.h5ad"
    assert config.resolve(legacy) != Path(legacy)
    assert str(config.resolve(legacy)).startswith(
        config.load_config()["roots"]["archive"])


# --- metrics ---------------------------------------------------------------

def test_random_baseline_f1():
    """Always-predict-positive gives F1 = 2p/(1+p)."""
    for p in (0.1, 0.5, 0.8):
        y = np.zeros(1000, dtype=int)
        y[: int(p * 1000)] = 1
        assert abs(metrics.random_baseline_f1(y) - 2 * p / (1 + p)) < 1e-6


def test_perfect_and_random_classifiers():
    y = np.array([0, 0, 1, 1])
    assert metrics.classification_metrics(y, np.array([0., .1, .9, 1.]))["AUROC"] == 1.0
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    auroc = metrics.classification_metrics(y, rng.random(4000))["AUROC"]
    assert 0.45 < auroc < 0.55


def test_threshold_is_agnostic():
    """Binary calls use 0.5 (Supp. Fig. 3)."""
    assert config.training_params()["classifier_threshold"] == 0.5


# --- ablation --------------------------------------------------------------

def test_ablation_block_size():
    """Filenames count 73-gene blocks, not genes.

    The runners walk `variable_genes[73*i : 73*j]`. Forgetting this makes the
    published table look ~73x too small.
    """
    from fig2.fig2f_gene_ablation import GENES_PER_BLOCK
    assert GENES_PER_BLOCK == 73
    # 40 blocks x 73 = 2920, the mP_d1 TGF-beta entry in the Fig. 2f table.
    assert 40 * GENES_PER_BLOCK == 2920


def test_exponential_fit_recovers_half_point():
    """genes_to_half locates the 50% point of the normalized 3-param fit.

    The curve is ``c + A*exp(-k x)`` with a floor ``c``: ablating every gene
    leaves the model at chance, not at zero. Min-max normalizing that fit puts
    the half point at ``ln(2)/k``, independent of A and c.
    """
    from fig2.fig2f_gene_ablation import fit_exponential, genes_to_half

    k = 1e-3
    x = np.linspace(0, 12000, 80)
    df = pd.DataFrame({"n_ablated": x, "AUROC": 0.5 + 0.45 * np.exp(-k * x)})

    (A, k_fit, c), ok = fit_exponential(df)
    assert ok, "fit did not converge on a clean synthetic curve"
    assert abs(k_fit - k) / k < 0.05, f"recovered k={k_fit:.2e}, expected {k:.2e}"
    assert abs(c - 0.5) < 0.02, f"recovered floor c={c:.3f}, expected 0.5"

    # Normalization removes c, so the half point is ln(2)/k.
    expected = np.log(2) / k
    got = genes_to_half(df, "fit")
    assert abs(got - expected) / expected < 0.02, f"{got:.0f} vs {expected:.0f}"


def test_ablation_prefers_published_source_data():
    """The published workbook is the authority for the genes-to-50% table."""
    from fig2.fig2f_gene_ablation import SOURCE_DATA_XLSX, load_published_fits

    assert SOURCE_DATA_XLSX.endswith("source_data_ablation_expfit.xlsx")
    published = load_published_fits()
    if published is None:
        return  # workbook not present in this checkout; nothing to verify
    assert {"signal", "assay", "held_out", "genes_to_50pct"} <= set(published.columns)
    # Values are in genes, not 73-gene blocks.
    assert published["genes_to_50pct"].max() > 500


# --- held-out row selection ------------------------------------------------

def test_held_out_mask_uses_batch_column_when_present():
    from fig2.fig2cd_generalization import _held_out_mask
    df = pd.DataFrame({"batch": [1, 1, 5, 5, 5]})
    assert _held_out_mask(df, [5], Path("x.csv")).tolist() == [False, False, True, True, True]


def test_held_out_mask_falls_back_to_trailing_rows():
    """Older CSVs lack `batch`; the runner wrote train rows then test rows."""
    from fig2.fig2cd_generalization import _held_out_mask
    n_out = config.batch_table()[3]["n_cells"]          # mE_d2 = 5224
    df = pd.DataFrame({"scanvi_pred_Stim": np.zeros(6379 + n_out)})
    mask = _held_out_mask(df, [3], Path("clean_splits_Bmp_in_5_out_3_results.csv"))
    assert mask.sum() == n_out
    assert mask[-n_out:].all() and not mask[:-n_out].any()


def test_split_parsing_distinguishes_panels():
    """Fig. 2c is single-holdout; Fig. 2d is trained on batches 3+6."""
    from fig2.fig2cd_generalization import PATTERN, _parse_batches
    m = PATTERN.search("clean_splits_Bmp_results_Bmp_in_3_6_out_5_7_results.csv")
    assert _parse_batches(m.group("in")) == (3, 6)
    assert _parse_batches(m.group("out")) == (5, 7)
    m = PATTERN.search("clean_splits_RA_in_5_out_3_results.csv")
    assert _parse_batches(m.group("out")) == (3,)


# --- fig 3 / fig 4 label handling ------------------------------------------

def test_code_string_parsing_normalises_case():
    """Saved codes spell BMP in caps; config uses 'Bmp'."""
    from fig3.fig3_lineage_dynamics import _parse_code_string
    calls = _parse_code_string("BMP+TgfB+Fgf-Wnt+RA-")
    assert calls == {"Bmp_pred": 1, "TgfB_pred": 1, "Fgf_pred": 0,
                     "Wnt_pred": 1, "RA_pred": 0}


def test_mesenchyme_population_canonicalisation():
    """Sub-states collapse onto the four Fig. 4a fates; others are dropped."""
    from fig4.fig4a_mesenchyme_enrichment import _canonical
    assert _canonical("respiratory-lung") == "respiratory"
    assert _canonical("respiratory-trachea") == "respiratory"
    assert _canonical("esophagus-2") == "esophageal"
    assert _canonical("dorsal lateral foregut") == "dorsal_lateral_foregut"
    # Not part of the Fig. 4a comparison.
    assert _canonical("Erythroid2") is None
    assert _canonical("Pharyngeal mesoderm") is None


def test_response_genes_are_human_symbols():
    """Mouse data is mapped to human symbols before analysis (Methods)."""
    for sig in config.signals():
        genes = config.response_genes(sig)
        assert genes, f"{sig} has no response genes"
        assert all(g == g.upper() for g in genes), f"{sig} has non-human casing"


def _main() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
