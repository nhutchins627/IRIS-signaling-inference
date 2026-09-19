"""Configuration loading and path resolution for the IRIS reproducibility package.

Everything that touches the filesystem goes through here, so relocating the
analysis means editing ``config/paths.yaml`` and nothing else.

Also handles a historical wrinkle: the original analysis ran out of
``/lab/tambora_li/Nicholas/figures-paper/``, which has since been moved to
``/archive/li/Nicholas-IRIS-figures-paper-archive/``. Legacy scripts still
contain the old absolute paths, so :func:`resolve` rewrites them on the fly.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# config/ lives two levels up from src/iris_repro/
_PKG_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = _PKG_ROOT / "config"

_VAR = re.compile(r"\$\{([a-zA-Z0-9_]+)\}")


def _interpolate(tree: Any, scope: Dict[str, str]) -> Any:
    """Recursively expand ``${name}`` references using ``scope``.

    Resolution is iterative so that ``outputs: ${work}/...`` works even when
    ``work`` is itself defined in the same mapping.
    """
    if isinstance(tree, dict):
        return {k: _interpolate(v, scope) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_interpolate(v, scope) for v in tree]
    if isinstance(tree, str):
        prev = None
        cur = tree
        # Bounded loop: guards against a self-referential ${a} -> ${a} cycle.
        for _ in range(10):
            if cur == prev or "${" not in cur:
                break
            prev = cur
            cur = _VAR.sub(lambda m: scope.get(m.group(1), m.group(0)), cur)
        return cur
    return tree


@lru_cache(maxsize=None)
def load_config() -> Dict[str, Any]:
    """Load and interpolate ``paths.yaml`` + ``model_params.yaml`` (cached)."""
    with open(CONFIG_DIR / "paths.yaml") as fh:
        paths = yaml.safe_load(fh)
    with open(CONFIG_DIR / "model_params.yaml") as fh:
        params = yaml.safe_load(fh)

    scope = dict(paths.get("roots", {}))
    # Let roots reference each other (outputs is defined in terms of work).
    scope = _interpolate(scope, scope)
    paths = _interpolate(paths, scope)
    paths["roots"] = scope

    cfg = {**paths, **params}

    # Environment overrides make the package portable without editing YAML.
    for key, env in (("work", "IRIS_WORK_DIR"), ("archive", "IRIS_ARCHIVE_DIR"),
                     ("outputs", "IRIS_OUTPUT_DIR")):
        if os.environ.get(env):
            cfg["roots"][key] = os.environ[env]
    return cfg


def resolve(path: str | Path) -> Path:
    """Return ``path`` with any legacy prefix rewritten to its current location.

    >>> resolve("/lab/tambora_li/Nicholas/figures-paper/fig5v2/fig5v2_data.h5ad")
    PosixPath('/archive/li/.../fig5v2/fig5v2_data.h5ad')
    """
    cfg = load_config()
    s = str(path)
    for old, new in (cfg.get("legacy_rewrites") or {}).items():
        if s.startswith(old):
            s = new + s[len(old):]
            break
    return Path(s)


def data_path(key: str, *, require: bool = True) -> Path:
    """Path to a named dataset from ``paths.yaml`` under ``data:``."""
    cfg = load_config()
    if key not in cfg["data"]:
        raise KeyError(
            f"Unknown dataset {key!r}. Available: {sorted(cfg['data'])}"
        )
    p = resolve(cfg["data"][key])
    if require and not p.exists():
        raise FileNotFoundError(
            f"Dataset {key!r} not found at {p}.\n"
            "If the archive is mounted elsewhere, set IRIS_ARCHIVE_DIR or edit "
            "config/paths.yaml."
        )
    return p


def results_dir(key: str) -> Path:
    return resolve(load_config()["results"][key])


def output_dir(*parts: str) -> Path:
    """Output directory, created on demand."""
    p = Path(load_config()["roots"]["outputs"]).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- Convenience accessors -------------------------------------------------

def architecture(signal: str) -> Dict[str, int]:
    """``{'n_hidden': ..., 'n_latent': ...}`` for a pathway."""
    arch = load_config()["architectures"]
    if signal not in arch:
        raise KeyError(f"No architecture for {signal!r}; have {sorted(arch)}")
    return dict(arch[signal])


def training_params() -> Dict[str, Any]:
    return dict(load_config()["training"])


def batch_name(batch: int) -> str:
    """Integer batch code -> screen name, e.g. ``5 -> 'hM_d4'``."""
    return load_config()["batches"][int(batch)]["name"]


def batch_table() -> Dict[int, Dict[str, Any]]:
    return {int(k): v for k, v in load_config()["batches"].items()}


def signals(include_shh: bool = False) -> list[str]:
    """The five benchmarked pathways, in display (bit-string) order."""
    order = list(load_config()["signal_order"])
    return order + ["Shh"] if include_shh else order


def response_genes(signal: str) -> list[str]:
    return list(load_config()["response_genes"][signal])


def palette() -> Dict[str, str]:
    return dict(load_config()["palette"])


def display_name(signal: str) -> str:
    return load_config()["display_names"].get(signal, signal)
