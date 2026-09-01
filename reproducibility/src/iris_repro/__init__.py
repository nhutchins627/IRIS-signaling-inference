"""IRIS reproducibility package.

Regenerates every analysis in Hutchins et al., organized by figure number.

    from iris_repro import config, data, model, metrics, plotting

Figure scripts live in ``figures/fig<N>/`` and the matching interactive
notebooks in ``notebooks/``. Both import this package, so the two always run
identical code paths.
"""
__version__ = "1.0.0"

from . import config, data, metrics, plotting, provenance  # noqa: F401

__all__ = ["config", "data", "metrics", "plotting", "provenance", "model"]


def __getattr__(name):
    # `model` pulls in torch/scvi, which is slow and unnecessary for
    # CSV-only figure regeneration -- so import it only when asked for.
    if name == "model":
        from . import model as _model
        return _model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
