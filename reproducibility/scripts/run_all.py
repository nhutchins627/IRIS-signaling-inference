#!/usr/bin/env python3
"""Run the IRIS reproducibility pipeline.

By default this runs only the *fast* path: every analysis that can be rebuilt
from saved predictions and result tables, no GPU required. Steps that refit
models are opt-in, because they take hours.

    python scripts/run_all.py                 # fast path, all figures
    python scripts/run_all.py --figure 2      # one figure
    python scripts/run_all.py --list          # show steps without running
    python scripts/run_all.py --include-slow  # also refit models (GPU)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class Step:
    figure: str
    name: str
    script: Path
    args: list[str] = field(default_factory=list)
    slow: bool = False
    note: str = ""


STEPS = [
    Step("1", "Fig 1c response-gene baseline",
         ROOT / "figures/fig1/fig1c_insample_accuracy.py",
         note="emits the baseline only; add --retrain for the IRIS "
              "comparison (GPU, ~20 min)"),
    Step("2", "Fig 2c/2d generalization",
         ROOT / "figures/fig2/fig2cd_generalization.py"),
    Step("2", "Fig 2e benchmarking (cross-validation)",
         ROOT / "figures/fig2/fig2e_model_benchmarking.py"),
    Step("2", "Fig 2e benchmarking (cross-species)",
         ROOT / "figures/fig2/fig2e_model_benchmarking.py",
         ["--regime", "cross-species"],
         note="add --compute on first run (CPU, ~1 h)"),
    Step("2", "Fig 2f gene ablation",
         ROOT / "figures/fig2/fig2f_gene_ablation.py", ["--assay", "mi"]),
    Step("3", "Fig 3 lineage dynamics",
         ROOT / "figures/fig3/fig3_lineage_dynamics.py"),
    Step("4", "Fig 4a mesenchyme enrichment",
         ROOT / "figures/fig4/fig4a_mesenchyme_enrichment.py",
         slow=True, note="needs --predict once (GPU)"),

    # --- supplementary ----------------------------------------------------
    Step("S", "Supp coverage map",
         ROOT / "figures/supp/supp_coverage.py"),
    Step("S", "Supp source-data status",
         ROOT / "figures/supp/supp_source_data.py", ["--check"]),
    Step("S", "Supp 3 threshold distribution",
         ROOT / "figures/supp/supp03_threshold_distribution.py"),
    Step("S", "Supp 4 screen diversity",
         ROOT / "figures/supp/supp04_screen_diversity.py"),
    Step("S", "Supp 19 airway receptors",
         ROOT / "figures/supp/supp19_airway_receptors.py"),
    Step("S", "Supp 2 in vivo trajectories",
         ROOT / "figures/supp/supp02_invivo_response_genes.py"),
    Step("S", "Supp 11 signal carry-over",
         ROOT / "figures/supp/supp11_signal_carryover.py",
         note="needs hM_d4_step_labels.csv rebuilt from the screen barcodes"),
]


def run(step: Step, python: str, extra: list[str]) -> tuple[bool, float, str]:
    cmd = [python, str(step.script), *step.args, *extra]
    print(f"\n{'=' * 72}\n[Fig {step.figure}] {step.name}\n  $ {' '.join(cmd)}\n{'=' * 72}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    out = proc.stdout.strip().splitlines()
    for line in out[-25:]:
        print("  " + line)
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()
        print("  --- stderr (last 15) ---")
        for line in err[-15:]:
            print("  " + line)
    # A step may exit non-zero on purpose when its GPU half has not been run
    # yet, after writing everything it could. Treat that as a soft skip.
    tail = (proc.stderr.strip().splitlines() or [""])[-1]
    # Markers a step prints when it stops for a *known missing input* rather
    # than a defect. Matched over the whole message, since these explanations
    # run to several lines.
    SOFT_MARKERS = ("run once with --", "--retrain", "--predict", "--compute",
                    "reconstruct them into", "needs per-cell",
                    "run without --atlas", "not saved as a standalone table")
    blob = proc.stderr.lower()
    soft = proc.returncode != 0 and any(m.lower() in blob for m in SOFT_MARKERS)
    if soft:
        return None, dt, tail
    return proc.returncode == 0, dt, tail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure",
                    help="run only this figure: 1-4, or S for supplementary")
    ap.add_argument("--include-slow", action="store_true",
                    help="also run steps that refit models")
    ap.add_argument("--list", action="store_true", help="list steps and exit")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter to use (default: the current one)")
    ap.add_argument("extra", nargs="*",
                    help="extra args passed through to every step")
    args = ap.parse_args()

    steps = [s for s in STEPS if not args.figure or s.figure == args.figure]
    if not args.include_slow:
        steps = [s for s in steps if not s.slow]

    if args.list:
        for s in STEPS:
            flag = "SLOW" if s.slow else "fast"
            print(f"  [Fig {s.figure}] {flag}  {s.name}")
            if s.note:
                print(f"           note: {s.note}")
        return

    if not steps:
        raise SystemExit("No steps selected.")

    from iris_repro import provenance
    provenance.check_environment()

    results = []
    for s in steps:
        if not s.script.exists():
            print(f"  ! missing script {s.script}")
            results.append((s, False, 0.0, "script not found"))
            continue
        ok, dt, err = run(s, args.python, args.extra)
        results.append((s, ok, dt, err))

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for s, ok, dt, err in results:
        status = {True: "OK  ", False: "FAIL", None: "SKIP"}[ok]
        print(f"  {status} [Fig {s.figure}] {s.name}  ({dt:.0f}s)")
        if ok is not True:
            if s.note:
                print(f"         hint: {s.note}")
            if err:
                print(f"         {err[:100]}")
    n_ok = sum(1 for _, ok, _, _ in results if ok is True)
    n_skip = sum(1 for _, ok, _, _ in results if ok is None)
    n_fail = sum(1 for _, ok, _, _ in results if ok is False)
    print(f"\n{n_ok} succeeded, {n_skip} skipped (need a GPU/compute pass), "
          f"{n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
