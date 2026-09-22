# Numerical artifact package

This package contains the code, retained records, and explicitly identified static figures supporting the numerical figures and tables in *Bipartite Gaussian Boson Sampling for Hamiltonian Cycles in Directed Graphs* (arXiv:2606.28775). The files described below are located directly in the repository root. The public entry points are `Figure-01.py` through `Figure-09.py` and `Table-01.py` through `Table-08.py`; their numbers match the displayed artifact numbers in the manuscript.

## Reproduce all reported artifacts

The verified replay uses Python 3.13.14 on Windows with the package versions pinned in `requirements.txt`. Other compatible environments may support saved-data analysis, but exact seeded trajectories are tied to the recorded runtime:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python reproduce_all.py
```

The command validates the archived records and writes all figures, table bodies, and validation reports to `generated/`. It does not rerun the expensive optical sampling or genetic algorithm experiments.

To reproduce one artifact, run its numbered entry point:

```bash
python Figure-04.py
python Table-03.py
```

Every numbered script accepts `--output-dir PATH`.

## Artifact map

| Manuscript artifact | Public entry point | Retained source | Reproduction |
|---|---|---|---|
| Fig. 1, optical architectures | `Figure-01.py` | `static/Figure01-source.pdf` | Stages the accepted vector schematic |
| Fig. 2, edge count and matrix functions | `Figure-02.py` | `static/Figure02-source.pdf` | Stages the accepted vector plot |
| Fig. 3, algorithm workflows | `Figure-03.py` | `static/Figure03-source.pdf` | Stages the accepted vector schematic |
| Fig. 4, main benchmark | `Figure-04.py` | `experiment_results.xlsx`, `sr_control_results.csv` | Recomputes the plot from trial records |
| Fig. 5, ablation study | `Figure-05.py` | `experiment_results.xlsx` | Recomputes the plot from trial records |
| Fig. 6, fitness weight scan | `Figure-06.py` | `alpha_sensitivity_results.xlsx` | Recomputes the plot from trial records |
| Fig. 7, output loss scan | `Figure-07.py` | `loss_scan_results.csv` | Recomputes the plot from graph summaries |
| Fig. 8, component sensitivity | `Figure-08.py` | `static/Figure08-source.pdf` | Stages the accepted vector plot |
| Fig. 9, multistage sensitivity | `Figure-09.py` | `static/Figure09-source.pdf` | Stages the accepted vector plot |
| Tab. I, postselection overhead | `Table-01.py` | Analytic expression in the manuscript | Recomputes the values analytically |
| Tab. II, accepted samples | `Table-02.py` | `gbs_sampling_data.xlsx` | Recomputes the summary from accepted banks |
| Tab. III, compact paired tests | `Table-03.py` | Main and source-resolved trial records | Recomputes all paired statistics |
| Tab. IV, ablation variants | `Table-04.py` | Algorithm specification | Emits the deterministic specification |
| Tab. V, matched subset controls | `Table-05.py` | `matched_control_data/` | Recomputes density, initialization, and success summaries |
| Tab. VI, fitness weight tests | `Table-06.py` | `alpha_sensitivity_results.xlsx` | Recomputes the paired tests |
| Tab. VII, default parameters | `Table-07.py` | Algorithm specification | Emits the deterministic specification |
| Tab. VIII, full paired tests | `Table-08.py` | Main and source-resolved trial records | Recomputes all detailed statistics |

The manuscript source retains two historical PDF filenames: displayed Fig. 5 includes `Figure06.pdf`, and displayed Fig. 6 includes `Figure05.pdf`. The public scripts follow the displayed figure numbers.

## Data-to-figure consistency

`validate_saved_data.py` checks the data before artifact generation. It verifies:

- 60 shared graphs and 10 trials per graph, algorithm, and size in the main benchmark, with verified replay source, bank, and output hashes;
- 500 raw shots per graph and the accepted subset banks used in Tab. II;
- the eight paired comparisons, bootstrap intervals, Wilcoxon tests, and Holm corrections used in Tabs. III, VI, and VIII;
- the raw matched control banks and trials used in Tab. V, including the outcome-independent selection of density dose `m=4`;
- all 960 saved output loss records and the covariance construction used by the loss runner;
- Python syntax, the complete set of numbered entry points, and explicit failure when a quantum backend is unavailable.

The generated validation report includes SHA-256 hashes for every primary input. The workflow writes derived statistical CSV and LaTeX files to `generated/`. The final-success matched-control and initial-population analyses are exported separately. `loss_tail_check.py` writes the ideal-state truncation bounds. All are regenerated from the retained records or specified analytic model.

Table V is rounded directly from the archived observations; there are no manual value overrides. Paired final-success ranks use integer success-count differences, preserving ties, as recommended by the [SciPy Wilcoxon numerical-precision documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html). Table II is rounded once from the unrounded summary statistics.

Figs. 1 and 3 are author-designed schematics. The point-level ensembles for Fig. 2 and the plotted records for Figs. 8 and 9 were not retained. Their numbered entry points therefore copy the accepted vector PDFs from `static/`; the repository does not claim that those three plots can be regenerated point for point. The experiment implementations for the component and multistage sensitivity scans remain available for independent reruns.

## Documented replacement main benchmark

The main and ablation outcomes were replaced by a replay of all 11 displayed
algorithms on the original 360 graphs, with 10 trials each (39,600 searches).
The historical quantum accepted subsets are reused without new quantum photon
sampling. The source-resolved control is regenerated from its graph seeds in
the recorded environment, and its raw counts and accepted subsets are retained.
All searches use the unchanged classes in `main_benchmark.py` and zero defaults.
The replay canonicalizes subset construction from sorted vertex lists before
building guidance, so input-set serialization does not affect seeded trajectories.
This preserves subset contents and sampling choices; it does not change the
fitness, mutation, or population/generation budgets.

`replay/manifest.json` records source hashes, input-bank hash, runtime versions,
configuration, and graph/trial seed formulas. `replay/completed.json` records the
output hashes. Per-graph JSON checkpoints retain graph edges, source subsets,
trial outcomes, returned cycles/paths, and the actual multistage insertion order.
Compressed NumPy files retain the 500 raw source-resolved shots per graph.
`replay/trial_records.csv.gz` is the combined trial archive. The canonical main
workbook contains the nine original displayed arms (32,400 rows), and
`sr_control_results.csv` contains the two source-control arms (7,200 rows).
The unused historical distinguishable-particle arm is not part of this replay.

Run the full search replay with:

```bash
python replay_benchmark.py --workers 32
python reproduce_all.py
python verify_replay.py
```

Matching checkpoints resume a run; a changed configuration or source hash is
rejected. Keep an archival copy before a completely fresh full run and set aside
its `replay/` folder. The full runner replaces canonical outcome files only when
all graph tasks finish. For a small independent pilot that leaves canonical files
alone, use `--sizes 20 --graphs 1 --trials 1 --output-dir pilot`.

`verify_replay.py` validates all saved Hamiltonian-cycle certificates and source
postselection records, then executes selected full-budget searches and selected
matched-control trials. It verifies stored-list reinforcement behavior and the
integer-rank correction. It also cross-checks all 39,600 checkpoint outcomes against the analysis inputs, verifies seed formulas and recorded insertion orders, and writes ordered per-graph bank hashes to `generated/provenance_audit.json`. This does not certify the historical quantum sampling
distribution. In particular, the original quantum raw mode counts and optical
environment remain unavailable.

The matched-density experiment is separate: all guidance sources, including
BipartiteGBS-InitOnly, were searched under its own shared 30-seed policy. Its trial
records are retained, with corrected rank tests; they are not historical main
outcomes relabeled as controls. The historical fitness-weight scan and loss study
also remain separate experiments, with their recorded limitations.

## Multistage ordering and loss-tail scope

The stored top-edge list initially follows `Counter.most_common`, with ties in
first-occurrence bank/vertex-pair traversal order. Reinforcement raises scores to
at least 0.8, leaves existing list positions unchanged, and prepends only absent
edges in the actual Python set iteration order. The replay records that order;
repeated prepending reverses the visit order of inserted edges. Initialization
uses only the first n entries and does not re-sort. The manuscript pseudocode
now describes this behavior.

`loss_tail_check.py` computes the ideal lossless total-photon tail above 30 from
the convolution of geometric source-pair distributions, and a thermal-marginal
union bound for any output mode reaching occupation eight. Its CSV lists all 60
loss-study graphs. These bounds do not validate an unknown historical backend or
bound errors after conditioning on postselection. Historical raw loss banks were
not retained; the retained graph summaries support the loss reanalysis.

## Archive identity and public status

The historical public snapshot is commit
`97f224e84ad875482c8cbec7938ce9a79cbd1b8a` of the linked GitHub repository.
This in-place revised package contains newer replay and correction files; that
historical commit is not a claim that the new package has already been uploaded.
`revision-manifest.json` identifies this package's source and numerical payload
by SHA-256. The manuscript distinguishes the historical public snapshot from
the revised package accompanying the submission.

## Repository layout

```text
./
|-- Figure-01.py ... Figure-09.py   canonical figure entry points
|-- Table-01.py ... Table-08.py     canonical table entry points
|-- artifact_builders.py            shared plotting and table logic
|-- validate_saved_data.py          data integrity and statistical checks
|-- reproduce_all.py                complete saved-data workflow
|-- main_benchmark.py               main quantum and GA implementation
|-- fitness_weight_scan.py          experiment underlying Fig. 6
|-- source_resolved.py              classical source-resolved sampler
|-- source_resolved_control.py      source-resolved benchmark runner
|-- matched_control_utils.py        shared matched-control functions
|-- matched_density_controls.py     matched density-dose experiment
|-- output_loss_scan.py             output loss experiment
|-- component_sensitivity.py        experiment underlying Fig. 8
|-- multistage_sensitivity.py       experiment underlying Fig. 9
|-- unobserved_edge_check.py        unobserved-edge convention check
|-- matched_control_data/           compressed raw records and metadata for Tab. V
|-- static/                         accepted vector artifacts without raw records
|-- *.xlsx, *.csv                   primary archived numerical records
`-- generated/                      reproducible output; ignored by Git
```

## Rerun the experiments

The saved-data workflow above is the route for reproducing the manuscript artifacts. Full experiment reruns are substantially more expensive. Install the optical backends in a separate compatible environment:

```bash
python -m pip install -r requirements-quantum.txt
```

The experiment implementations are:

| Experiment | Script | Primary output |
|---|---|---|
| New optical experiment (not the revised benchmark replay) | `main_benchmark.py` | `experiment_results.xlsx`, `gbs_sampling_data.xlsx` |
| Revised main/ablation search replay | `replay_benchmark.py` | main workbook, source-control CSVs, `replay/` |
| Fitness weight scan | `fitness_weight_scan.py` | `alpha_sensitivity_results.xlsx` |
| Source-resolved controls | `source_resolved_control.py` | `sr_control_results.csv`, `sr_control_sampling.csv` |
| Matched density-dose controls | `matched_density_controls.py` | `matched_control_data/` |
| Uniform output loss scan | `output_loss_scan.py` | `loss_scan_results.csv` |
| Initialization and mutation sensitivity | `component_sensitivity.py` | sensitivity plot |
| Multistage sensitivity | `multistage_sensitivity.py` | sensitivity plot |
| Unobserved-edge convention | `unobserved_edge_check.py` | `default_check_results.csv` |

The original optical package versions used for the historical simulations were not recorded. They are not reconstructed by the replacement classical search replay. A fresh experiment rerun is scientifically comparable but is not guaranteed to reproduce identical random streams across third-party sampler versions. Keep rerun outputs separate from the archived inputs until they have been checked.

## License

The scripts are licensed under the Apache License 2.0. Each Python file carries the SPDX header, and the full license is in `LICENSE`.
