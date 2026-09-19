# Package structure

```
IRIS_reproducibility/
├── README.md               start here
├── requirements.txt        pinned to the versions used for the paper
│
├── config/
│   ├── paths.yaml          every filesystem path (edit this to relocate)
│   └── model_params.yaml   architectures, batch map, splits, response genes
│
├── src/iris_repro/
│   ├── config.py           YAML loading, ${var} interpolation, legacy rewrites
│   ├── data.py             loading, label masking, batch splits, code strings
│   ├── model.py            SCVI -> SCANVI wrapper (train / load / predict)
│   ├── metrics.py          ROC/PR/F1 and the tests named in each legend
│   ├── plotting.py         figure style and shared panel builders
│   └── provenance.py       environment checks + per-run JSON sidecars
│
├── figures/
│   ├── fig1/fig1c_insample_accuracy.py
│   ├── fig2/fig2cd_generalization.py       Fig. 2c, 2d
│   ├── fig2/fig2e_model_benchmarking.py    Fig. 2e, Supp. 10c
│   ├── fig2/fig2f_gene_ablation.py         Fig. 2f
│   ├── fig3/fig3_lineage_dynamics.py       Fig. 3d-f
│   ├── fig4/fig4a_mesenchyme_enrichment.py Fig. 4a
│   └── supp/
│       ├── supp_coverage.py                per-figure status for Supp. 1-21
│       ├── supp_source_data.py             wraps the 12 source-data scripts
│       ├── supp02_invivo_response_genes.py Supp. 2 (partial)
│       ├── supp03_threshold_distribution.py Supp. 3
│       ├── supp04_screen_diversity.py      Supp. 4
│       ├── supp11_signal_carryover.py      Supp. 11 (needs one input)
│       └── supp19_airway_receptors.py      Supp. 19
│
├── notebooks/              interactive counterparts (same library calls)
├── scripts/
│   ├── check_setup.py      preflight: environment, config, data
│   └── run_all.py          orchestrator
├── tests/test_conventions.py   guards the easy-to-get-wrong conventions
├── docs/                   DATA.md, METHODS.md, STRUCTURE.md,
│                           FIGURE_MAPPING.md
├── cache/                  expensive intermediates (gitignored)
└── outputs/fig<N>/         generated SVG/PNG/CSV + provenance (gitignored)
```

## Design rules

**One source of truth for paths.** Nothing hard-codes a filesystem location;
everything resolves through `config/paths.yaml`. The original working
directory was archived mid-project, so `config.resolve()` rewrites the stale
`/lab/.../figures-paper/` prefix automatically.

**Scripts and notebooks share code.** Both import `iris_repro`, so an
interactive exploration and a batch rerun cannot silently disagree.

**Fast by default.** Panels rebuild from saved predictions in seconds; every
step that refits a model is opt-in behind a flag and prints what it will cost.

**Provenance on every run.** Each script writes
`<analysis>.provenance.json` recording package versions, git commit, host,
input fingerprints and parameters.

**Supplementary coverage is mapped, not claimed.** `supp_coverage.py` reports
per figure whether a script exists here, a source-data script is wrapped, or an
input is still missing -- and verifies those claims against the filesystem
rather than trusting the table. Where a panel cannot be produced, the script
says exactly what is missing instead of emitting something wrong.

**Conventions are tested.** `tests/test_conventions.py` pins the things that
would otherwise fail silently: the 73-gene ablation block size, the batch
code map, the bit order of combination strings, and how held-out rows are
identified in each CSV generation.
