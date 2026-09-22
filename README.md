# Numerical artifact package

This repository contains the minimal code and archived data needed to reproduce the numerical figures and tables in *Bipartite Gaussian Boson Sampling for Hamiltonian Cycles in Directed Graphs* (arXiv:2606.28775). The files described below are located directly in the repository root. The public entry points are `Figure-01.py` through `Figure-09.py` and `Table-01.py` through `Table-08.py`; their numbers match the displayed artifact numbers in the manuscript.

## Reproduce all reported artifacts

Use Python 3.11 or newer:

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

- 60 shared graphs and 10 trials per graph, algorithm, and size in the main benchmark;
- 500 raw shots per graph and the accepted subset banks used in Tab. II;
- the eight paired comparisons, bootstrap intervals, Wilcoxon tests, and Holm corrections used in Tabs. III, VI, and VIII;
- the raw matched control banks and trials used in Tab. V, including the outcome-independent selection of density dose `m=4`;
- all 960 saved output loss records and the covariance construction used by the loss runner;
- Python syntax, the complete set of numbered entry points, and explicit failure when a quantum backend is unavailable.

The generated validation report includes SHA-256 hashes for every primary input. No derived statistical CSV or LaTeX file is stored in the source tree; the numbered scripts recompute these quantities from the retained records.

Tab. V contains one presentation-level exception. The archived mean initial valid-edge fraction for `Uniform-Matched-InitOnly` at `n=20` is `0.5134816667`, while the finalized manuscript reports `0.514`. `Table-05.py` preserves the finalized displayed value. This difference of `0.0005183` does not affect any inference.

Figs. 1 and 3 are author-designed schematics. The point-level ensembles for Fig. 2 and the plotted records for Figs. 8 and 9 were not retained. Their numbered entry points therefore copy the accepted vector PDFs from `static/`; the repository does not claim that those three plots can be regenerated point for point. The experiment implementations for the component and multistage sensitivity scans remain available for independent reruns.

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
| Main benchmark and accepted BipartiteGBS banks | `main_benchmark.py` | `experiment_results.xlsx`, `gbs_sampling_data.xlsx` |
| Fitness weight scan | `fitness_weight_scan.py` | `alpha_sensitivity_results.xlsx` |
| Source-resolved controls | `source_resolved_control.py` | `sr_control_results.csv`, `sr_control_sampling.csv` |
| Matched density-dose controls | `matched_density_controls.py` | `matched_control_data/` |
| Uniform output loss scan | `output_loss_scan.py` | `loss_scan_results.csv` |
| Initialization and mutation sensitivity | `component_sensitivity.py` | sensitivity plot |
| Multistage sensitivity | `multistage_sensitivity.py` | sensitivity plot |
| Unobserved-edge convention | `unobserved_edge_check.py` | `default_check_results.csv` |

The original package versions used for the historical simulations were not recorded. A fresh experiment rerun is scientifically comparable but is not guaranteed to reproduce identical random streams across third-party sampler versions. Keep rerun outputs separate from the archived inputs until they have been checked.

## License

The scripts are licensed under the Apache License 2.0. Each Python file carries the SPDX header, and the full license is in `LICENSE`.
