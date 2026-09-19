#!/usr/bin/env python3
"""Preflight check: environment, config and data availability.

Run this first. It answers "will anything actually work here?" without
loading a single cell of data.

    PYTHONPATH=src python scripts/check_setup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OK, WARN, FAIL = "  OK  ", " WARN ", " FAIL "


def main() -> int:
    problems = 0

    print("=" * 68)
    print("ENVIRONMENT")
    print("=" * 68)
    from iris_repro import provenance
    report = provenance.check_environment()
    for pkg, ver in sorted(report["actual"].items()):
        tag = FAIL if ver == "not installed" else (
            WARN if pkg in report["mismatches"] else OK)
        if tag is not OK:
            problems += (tag is FAIL)
        print(f"[{tag}] {pkg:12s} {ver}")

    print()
    print("=" * 68)
    print("CONFIG")
    print("=" * 68)
    from iris_repro import config
    cfg = config.load_config()
    for k, v in cfg["roots"].items():
        exists = Path(v).exists()
        print(f"[{OK if exists else FAIL}] root {k:8s} {v}")
        problems += (not exists)

    # The legacy rewrite is what keeps old absolute paths working.
    legacy = "/lab/tambora_li/Nicholas/figures-paper/fig5v2/fig5v2_data.h5ad"
    rewritten = config.resolve(legacy)
    print(f"[{OK if rewritten != Path(legacy) else WARN}] "
          f"legacy path rewriting active")

    print()
    print("=" * 68)
    print("DATASETS")
    print("=" * 68)
    for key in cfg["data"]:
        try:
            p = config.data_path(key)
            print(f"[{OK}] {key:20s} {p.stat().st_size / 1e9:6.2f} GB")
        except FileNotFoundError:
            print(f"[{WARN}] {key:20s} not found (only needed by some figures)")
        except Exception as exc:
            print(f"[{FAIL}] {key:20s} {exc}")
            problems += 1

    print()
    print("=" * 68)
    print("RESULT DIRECTORIES")
    print("=" * 68)
    for key in cfg["results"]:
        p = config.results_dir(key)
        n = len(list(p.glob("*"))) if p.exists() else 0
        tag = OK if p.exists() and n else WARN
        print(f"[{tag}] {key:18s} {n:6d} entries  {p}")

    print()
    print("=" * 68)
    print("HYPERPARAMETERS")
    print("=" * 68)
    for sig in config.signals(include_shh=True):
        a = config.architecture(sig)
        print(f"[{OK}] {sig:5s} hidden={a['n_hidden']:5d} latent={a['n_latent']:3d}")

    print()
    if problems:
        print(f"{problems} blocking problem(s). Fix config/paths.yaml or the "
              f"environment before running the pipeline.")
    else:
        print("All checks passed. Run: python scripts/run_all.py")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
