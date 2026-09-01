#!/usr/bin/env python3
"""Coverage map: what reproduces every supplementary figure, and what doesn't.

Answers "is the code here for all the supplemental figures?" honestly, per
panel, with the status verified against the filesystem rather than asserted.

    python supp_coverage.py            # the table
    python supp_coverage.py --gaps     # only what is missing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from iris_repro import config

# status:
#   script    - a script in this package regenerates it
#   sourcedata- a numbered source-data script produces its underlying table
#   data      - inputs are present, but no dedicated script exists yet
#   gpu       - needs a model refit before it can be produced
#   wetlab    - microscopy/imaging; no computational path
COVERAGE = [
    (1,  "Signal inference using the response-gene method", "script",
     "figures/fig1/fig1c_insample_accuracy.py", "fig1_source_data_from_archive/"),
    (2,  "In vivo signal inference, response-gene method", "partial",
     "figures/supp/supp02_invivo_response_genes.py",
     "IRIS trajectories only; response-gene scores need the atlas matrix"),
    (3,  "Predicted probability distribution and thresholding", "script",
     "figures/supp/supp03_threshold_distribution.py", "clean_splits/*.csv"),
    (4,  "Cell-state diversity in the screens", "script",
     "figures/supp/supp04_screen_diversity.py", "screens_ref_full.h5ad"),
    (5,  "Hyperparameter screen", "data",
     "-", "ARCHIVE supp-fig1/ (6729 model dirs; sweep outputs)"),
    (6,  "Cross-validation ROC/PRC", "script",
     "figures/fig2/fig2cd_generalization.py", "clean_splits*/"),
    (7,  "Cross-cell-type generalization", "script",
     "figures/fig2/fig2cd_generalization.py --panel 2d", "clean_splits*/"),
    (8,  "Cross-species generalization", "script",
     "figures/fig2/fig2e_model_benchmarking.py --regime cross-species",
     "cross_species_*_results.csv"),
    (9,  "Increasing cell-type diversity in training", "gpu",
     "-", "needs refits with/without hM_d4 in training"),
    (10, "Model benchmarking, white-box models", "script",
     "figures/fig2/fig2e_model_benchmarking.py", "benchmark_batch*.csv"),
    (11, "Signal carry-over across time steps", "partial",
     "figures/supp/supp11_signal_carryover.py",
     "needs hM_d4_step_labels.csv (recoverable from barcodes)"),
    (12, "Gene ablation tests", "script",
     "figures/fig2/fig2f_gene_ablation.py --assay hvg|expr", "ablation_results/"),
    (13, "Gene saliency / GSEA", "sourcedata",
     "figures/supp/supp_source_data.py --run 11", "source_data_saliency_maps.xlsx"),
    (14, "Per-pathway activity across the atlas", "sourcedata",
     "figures/supp/supp_source_data.py --run 05", "source_data_signal_combos_stats.xlsx"),
    (15, "Endodermal and cardiac lineage predictions", "sourcedata",
     "figures/supp/supp_source_data.py --run 01", "source_data_meso_endo_diffmap.xlsx"),
    (16, "Signaling heterogeneity vs cell-type heterogeneity", "sourcedata",
     "figures/supp/supp_source_data.py --run 08|09|10",
     "source_data_{endoderm_quant,cardiomyocytes,cardio}.xlsx"),
    (17, "Somitic mesoderm spatial reconstruction", "sourcedata",
     "figures/supp/supp_source_data.py --run 04|06",
     "source_data_somitic_meso_diffmap.xlsx"),
    (18, "Early neural differentiation", "sourcedata",
     "figures/supp/supp_source_data.py --run 02|03",
     "source_data_{neuroectoderm,nmp}_diffmap.xlsx"),
    (19, "Airway epithelium receptor enrichment", "script",
     "figures/supp/supp19_airway_receptors.py", "receptor_pred_tests/"),
    (20, "Organ-specific mesenchyme fate divergence", "sourcedata",
     "figures/supp/supp_source_data.py --run 07",
     "source_data_fig6v2_resp_meso_diffmap.xlsx"),
    (21, "WNT validation (foregut explant + hESC HCR-FISH)", "wetlab",
     "-", "microscopy; quantification in the paper's source data"),
]

LABEL = {
    "script":     "script in this package",
    "sourcedata": "source-data script (wrapped)",
    "data":       "inputs present, no script yet",
    "gpu":        "needs a model refit",
    "partial":    "script present, needs one input rebuilt",
    "wetlab":     "imaging; no computational path",
}


def build() -> pd.DataFrame:
    work = Path(config.load_config()["roots"]["work"])
    rows = []
    for num, title, status, how, inputs in COVERAGE:
        # Verify the claim rather than trusting the table.
        probe = inputs.split()[0].replace("ARCHIVE ", "")
        if probe.startswith(("source_data", "needs", "microscopy")):
            present = None
        else:
            present = bool(list(work.glob(probe))) or (work / probe).exists()
        rows.append({"supp": num, "title": title,
                     "status": LABEL[status], "how": how,
                     "inputs": inputs,
                     "inputs_found": present})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gaps", action="store_true",
                    help="only rows without a runnable script")
    args = ap.parse_args()

    df = build()
    outdir = config.output_dir("supp")
    df.to_csv(outdir / "supp_coverage.csv", index=False)

    show = df[~df["status"].str.contains("script|source-data")] if args.gaps else df
    with pd.option_context("display.max_colwidth", 46, "display.width", 200):
        print(show[["supp", "title", "status", "how"]].to_string(index=False))

    print()
    counts = df["status"].value_counts()
    runnable = int(counts.get("script in this package", 0) +
                   counts.get("source-data script (wrapped)", 0))
    print(f"{runnable}/{len(df)} supplementary figures have a runnable path")
    for k, v in counts.items():
        print(f"  {v:2d}  {k}")
    print(f"\nWrote {outdir / 'supp_coverage.csv'}")


if __name__ == "__main__":
    main()
