#!/usr/bin/env python3
"""Supplementary figures -- run the source-data extraction scripts.

Twelve numbered scripts under ``in_vivo_results/source_data_scripts/`` already
produce the source data behind most supplementary panels, each writing an
``.xlsx`` workbook. Rather than reimplement them (and risk drifting from the
published numbers) this wraps them: it maps each script to the supplementary
figure it supports, runs them on request, and summarises the workbooks.

    python supp_source_data.py --list          # what maps to what
    python supp_source_data.py --check         # which outputs already exist
    python supp_source_data.py --run 12        # run one script
    python supp_source_data.py --run all       # run everything (slow)
    python supp_source_data.py --summarize     # sheet inventory per workbook

Note these scripts hold absolute paths and expect the working directory's
result files; several need a GPU. ``--check`` tells you what is already done
without running anything.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from iris_repro import config, provenance

ANALYSIS = "supp_source_data"
SCRIPT_DIR = "in_vivo_results/source_data_scripts"

# script -> (supplementary figures it supports, output workbook, notes)
SCRIPTS = {
    "01_meso_endo_diffmap_pseudotime.py": (
        ["Fig. 3c-f", "Supp. 15"], "source_data_meso_endo_diffmap.xlsx",
        "endoderm + cardiac diffusion map and pseudotime"),
    "02_neuroectoderm_diffmap_pseudotime.py": (
        ["Supp. 18"], "source_data_neuroectoderm_diffmap.xlsx",
        "early neural differentiation; cell types outside the training data"),
    "03_nmp_diffmap_multiroot_pseudotime.py": (
        ["Supp. 18"], "source_data_nmp_diffmap.xlsx",
        "neuromesodermal progenitor trajectory"),
    "04_somitic_mesoderm_diffmap.py": (
        ["Fig. 3g-i", "Supp. 17"], "source_data_somitic_meso_diffmap.xlsx",
        "somitic mesoderm pseudospace and morphogen gradients"),
    "05_signal_combos_dotplot_stats.py": (
        ["Fig. 3b", "Supp. 14"], "source_data_signal_combos_stats.xlsx",
        "combination usage across stages; Mann-Whitney enrichment"),
    "06_subset_leiden_enrichment.py": (
        ["Supp. 17d-e"], "source_data_leiden_enrichment.xlsx",
        "re-clustering of the RA-WNT+FGF- population"),
    "07_fig6v2_resp_meso_diffmap.py": (
        ["Fig. 4a", "Supp. 20"], "source_data_fig6v2_resp_meso_diffmap.xlsx",
        "respiratory mesenchyme diffusion map (archive dir fig6v2)"),
    "08_endoderm_quant.py": (
        ["Supp. 16c-d"], "source_data_endoderm_quant.xlsx",
        "gut heterogeneity; foregut vs mid/hindgut markers"),
    "09_cardiomyocytes_analysis_quant.py": (
        ["Supp. 16b"], "source_data_cardiomyocytes_analysis_quant.xlsx",
        "RA+ vs RA- cardiomyocytes; atrial markers"),
    "10_cardio.py": (
        ["Fig. 3d", "Supp. 16a"], "source_data_cardio.xlsx",
        "cardiac lineage marker expression"),
    "11_saliency_maps.py": (
        ["Supp. 13"], "source_data_saliency_maps.xlsx",
        "gradient saliency + GSEA over GO Biological Process 2021"),
    "12_ablation_expfit.py": (
        ["Fig. 2f", "Supp. 12"], "source_data_ablation_expfit.xlsx",
        "exponential fits; produces the published genes-to-50% table"),
}


def script_dir() -> Path:
    return Path(config.load_config()["roots"]["work"]) / SCRIPT_DIR


def status() -> pd.DataFrame:
    d = script_dir()
    rows = []
    for name, (figs, out, note) in sorted(SCRIPTS.items()):
        wb = d / out
        rows.append({
            "script": name,
            "figures": ", ".join(figs),
            "output_exists": wb.exists(),
            "size_kb": round(wb.stat().st_size / 1024, 1) if wb.exists() else None,
            "note": note,
        })
    return pd.DataFrame(rows)


def summarize() -> pd.DataFrame:
    """List the sheets inside each existing workbook."""
    d = script_dir()
    rows = []
    for name, (figs, out, _) in sorted(SCRIPTS.items()):
        wb = d / out
        if not wb.exists():
            continue
        try:
            sheets = pd.ExcelFile(wb).sheet_names
        except Exception as exc:
            rows.append({"workbook": out, "sheet": f"<unreadable: {exc}>",
                         "figures": ", ".join(figs)})
            continue
        for sh in sheets:
            rows.append({"workbook": out, "sheet": sh,
                         "figures": ", ".join(figs)})
    return pd.DataFrame(rows)


def run(name: str, python: str) -> bool:
    d = script_dir()
    path = d / name
    if not path.exists():
        print(f"  ! missing {path}")
        return False
    print(f"\n=== {name} -> {SCRIPTS[name][1]} ===")
    # These scripts write relative to their own directory.
    proc = subprocess.run([python, str(path)], cwd=d,
                          capture_output=True, text=True)
    for line in proc.stdout.strip().splitlines()[-12:]:
        print("  " + line)
    if proc.returncode != 0:
        for line in proc.stderr.strip().splitlines()[-12:]:
            print("  " + line)
        print(f"  FAILED (exit {proc.returncode})")
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="script -> figure map")
    ap.add_argument("--check", action="store_true", help="which outputs exist")
    ap.add_argument("--summarize", action="store_true", help="sheets per workbook")
    ap.add_argument("--run", metavar="N", help="script number (e.g. 12) or 'all'")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    outdir = config.output_dir("supp")

    if args.list or not any([args.check, args.summarize, args.run]):
        print("Supplementary source-data scripts\n")
        for name, (figs, out, note) in sorted(SCRIPTS.items()):
            print(f"  {name}")
            print(f"      supports : {', '.join(figs)}")
            print(f"      output   : {out}")
            print(f"      {note}")
        print("\nRun one with --run <number>, or --check to see what already exists.")
        return

    if args.check:
        table = status()
        table.to_csv(outdir / "supp_source_data_status.csv", index=False)
        print(table[["script", "figures", "output_exists", "size_kb"]].to_string(index=False))
        missing = table[~table["output_exists"]]
        print(f"\n{len(table) - len(missing)}/{len(table)} workbooks present")
        if len(missing):
            print("Missing:", ", ".join(missing["script"]))

    if args.summarize:
        table = summarize()
        table.to_csv(outdir / "supp_source_data_sheets.csv", index=False)
        print(table.to_string(index=False))
        print(f"\n{table['workbook'].nunique()} workbooks, {len(table)} sheets")

    if args.run:
        targets = (sorted(SCRIPTS) if args.run == "all"
                   else [n for n in SCRIPTS if n.startswith(args.run.zfill(2))])
        if not targets:
            raise SystemExit(f"No script matches {args.run!r}. Use --list.")
        results = {n: run(n, args.python) for n in targets}
        print("\nSUMMARY")
        for n, ok in results.items():
            print(f"  {'OK  ' if ok else 'FAIL'} {n}")
        provenance.record(ANALYSIS, outdir,
                          params={"ran": list(results)},
                          results={"succeeded": [n for n, k in results.items() if k]})


if __name__ == "__main__":
    main()
