# Copyright 2026 Kun Wang
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the matched subset-source and density-dose experiment in Table V.

For each saved BipartiteGBS accepted bank, this experiment keeps the exact
number and ordered sizes of its vertex subsets.  A classical bank at dose m is
constructed by drawing m uniform subsets of each required size and retaining
the candidate with the largest directed induced-edge density.  Thus m=1 is the
uniform matched control and m>1 introduces a monotone, permanent-free density
bias without changing the downstream InitOnly genetic algorithm.

The experiment uses several independent classical-bank realizations per graph,
paired GA random seeds across all methods, and records both initial-population
quality and final search success.  The density dose whose pooled mean bank
density is closest to the saved BipartiteGBS banks is identified without using
any optimization outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import matched_control_utils as base  # noqa: E402


DEFAULT_DOSES = [1, 2, 4, 8, 16]
DEFAULT_BANK_REPLICATES = 3
DOSE_SEED_OFFSET = 51_000_000
TRIAL_SEED_OFFSET = 61_000_000
BOOTSTRAP_SEED = 20_260_921
METHOD_BGBS = "BipartiteGBS-InitOnly"
METHOD_GA = "Standard-GA"


def method_for_dose(dose: int) -> str:
    return "Uniform-Matched-InitOnly" if dose == 1 else f"DensityDose-m{dose}-InitOnly"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=base.DEFAULT_SIZES)
    parser.add_argument("--graphs-per-size", type=int, default=base.DEFAULT_GRAPHS_PER_SIZE)
    parser.add_argument("--trials", type=int, default=base.DEFAULT_TRIALS)
    parser.add_argument("--bank-replicates", type=int, default=DEFAULT_BANK_REPLICATES)
    parser.add_argument("--doses", nargs="+", type=int, default=DEFAULT_DOSES)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument(
        "--sampling-workbook",
        type=Path,
        default=REPO_ROOT / "gbs_sampling_data.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "matched_control_data",
    )
    return parser.parse_args()


def density_dose_bank(
    graph: object,
    target_sizes: Sequence[int],
    dose: int,
    rng: np.random.Generator,
) -> List[Set[int]]:
    """Choose the densest of ``dose`` uniform candidates for every subset."""
    if dose < 1:
        raise ValueError("Density dose must be positive")
    bank: List[Set[int]] = []
    for size in target_sizes:
        candidates = [
            set(int(v) for v in rng.choice(graph.n, size=size, replace=False))
            for _ in range(dose)
        ]
        edge_counts = np.array(
            [base.induced_edge_count(graph, subset) for subset in candidates], dtype=int
        )
        best = np.flatnonzero(edge_counts == edge_counts.max())
        bank.append(candidates[int(rng.choice(best))])
    return bank


def seeded_initial_metrics(
    graph: object, info: object, config: Dict[str, object], seed: int
) -> Dict[str, float]:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    algorithm = base.make_init_only(config)
    population = algorithm._gbs_guided_init_population(graph, info)
    fitness = np.asarray([algorithm._fitness(graph, ind) for ind in population], dtype=float)
    cycles = sum(int(algorithm.validator.verify_cycle(graph, ind)) for ind in population)
    return {
        "Initial Mean Fitness": float(fitness.mean()),
        "Initial Best Fitness": float(fitness.max()),
        "Initial Hamiltonian Cycles": int(cycles),
    }


def seeded_standard_initial_metrics(
    graph: object, config: Dict[str, object], seed: int
) -> Dict[str, float]:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    algorithm = base.framework.GeneticAlgorithm(
        pop_size=config["population"],
        max_generations=config["generations"],
        crossover_rate=config["crossover_rate"],
        mutation_rate=config["mutation_rate"],
    )
    population = algorithm._init_population(graph)
    fitness = np.asarray([algorithm._fitness(graph, ind) for ind in population], dtype=float)
    cycles = sum(int(algorithm.validator.verify_cycle(graph, ind)) for ind in population)
    return {
        "Initial Mean Fitness": float(fitness.mean()),
        "Initial Best Fitness": float(fitness.max()),
        "Initial Hamiltonian Cycles": int(cycles),
    }


def rows_for_bank(
    graph: object,
    sampler: str,
    bank: Sequence[Set[int]],
    seed: int | None,
    replicate: int,
    dose: int | None,
) -> List[dict]:
    rows = base.bank_rows(graph, sampler, bank, seed)
    for row in rows:
        row["Bank Replicate"] = replicate
        row["Density Dose"] = dose
    return rows


def run_one_graph(task: tuple) -> Tuple[List[dict], List[dict]]:
    n, index, bgbs_bank, config = task
    graph_seed = base.graph_seed(n, index)
    graph = base.framework.GraphGenerator.generate_random_digraph(n, 0.3, graph_seed)
    graph.name = f"Random_n{n}_{index + 1}"
    target_sizes = [len(sample) for sample in bgbs_bank]
    methods = [METHOD_GA, METHOD_BGBS] + [method_for_dose(d) for d in config["doses"]]
    bgbs_info = base.guidance_from_bank(graph, bgbs_bank)

    trial_rows: List[dict] = []
    bank_rows: List[dict] = []
    for replicate in range(config["bank_replicates"]):
        banks: Dict[str, Sequence[Set[int]]] = {METHOD_BGBS: bgbs_bank}
        infos: Dict[str, object] = {METHOD_BGBS: bgbs_info}
        bank_rows.extend(
            rows_for_bank(graph, "BipartiteGBS", bgbs_bank, None, replicate, None)
        )
        for dose in config["doses"]:
            bank_seed = DOSE_SEED_OFFSET + graph_seed * 100 + replicate * 10_000 + dose
            bank = density_dose_bank(
                graph, target_sizes, dose, np.random.default_rng(bank_seed)
            )
            if [len(sample) for sample in bank] != target_sizes:
                raise AssertionError(f"Size matching failed for {graph.name}, m={dose}")
            method = method_for_dose(dose)
            banks[method] = bank
            infos[method] = base.guidance_from_bank(graph, bank)
            bank_rows.extend(
                rows_for_bank(
                    graph,
                    "Uniform-Matched" if dose == 1 else f"DensityDose-m{dose}",
                    bank,
                    bank_seed,
                    replicate,
                    dose,
                )
            )

        for trial in range(config["trials"]):
            trial_seed = (
                TRIAL_SEED_OFFSET
                + n * 1_000_000
                + index * 10_000
                + replicate * 100
                + trial
            )
            for method in methods:
                random.seed(trial_seed)
                np.random.seed(trial_seed % (2**32 - 1))
                dose = None
                if method == METHOD_GA:
                    metrics = seeded_standard_initial_metrics(graph, config, trial_seed)
                    random.seed(trial_seed)
                    np.random.seed(trial_seed % (2**32 - 1))
                    algorithm = base.framework.GeneticAlgorithm(
                        pop_size=config["population"],
                        max_generations=config["generations"],
                        crossover_rate=config["crossover_rate"],
                        mutation_rate=config["mutation_rate"],
                    )
                    result = algorithm.search(graph, None)
                else:
                    metrics = seeded_initial_metrics(graph, infos[method], config, trial_seed)
                    random.seed(trial_seed)
                    np.random.seed(trial_seed % (2**32 - 1))
                    algorithm = base.make_init_only(config)
                    result = algorithm.search(graph, infos[method])
                if method not in {METHOD_BGBS, METHOD_GA}:
                    dose = 1 if method.startswith("Uniform") else int(method.split("m")[1].split("-")[0])
                trial_rows.append(
                    {
                        "Graph Name": graph.name,
                        "Graph Size": n,
                        "Graph Seed": graph_seed,
                        "Bank Replicate": replicate,
                        "Trial": trial,
                        "Trial Seed": trial_seed,
                        "Algorithm": method,
                        "Density Dose": dose,
                        "Success": int(result.success),
                        "Generations": result.generations,
                        "Longest Path Length": len(result.longest_path) if result.longest_path else 0,
                        **metrics,
                    }
                )
    return trial_rows, bank_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def per_graph_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["Graph Size", "Graph Name", "Algorithm"], as_index=False)
        .agg(
            Success_Rate=("Success", "mean"),
            Success_Count=("Success", "sum"),
            Trial_Count=("Success", "size"),
            Initial_Mean_Fitness=("Initial Mean Fitness", "mean"),
            Initial_Best_Fitness=("Initial Best Fitness", "mean"),
            Initial_Cycles=("Initial Hamiltonian Cycles", "mean"),
            Mean_Generations=("Generations", "mean"),
        )
    )


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    per_graph = per_graph_results(results)
    summary = (
        per_graph.groupby(["Graph Size", "Algorithm"], as_index=False)
        .agg(
            Graphs=("Graph Name", "size"),
            Mean_Success_Rate=("Success_Rate", "mean"),
            SD_Success_Rate=("Success_Rate", "std"),
            Mean_Initial_Fitness=("Initial_Mean_Fitness", "mean"),
            Mean_Initial_Best_Fitness=("Initial_Best_Fitness", "mean"),
            Mean_Initial_Cycles=("Initial_Cycles", "mean"),
            Mean_Generations=("Mean_Generations", "mean"),
        )
    )
    summary["SEM_Success_Rate"] = summary["SD_Success_Rate"] / np.sqrt(summary["Graphs"])
    for column in ["Mean_Success_Rate", "SD_Success_Rate", "SEM_Success_Rate"]:
        summary[column + " (%)"] = 100.0 * summary[column]
    return summary


def density_summary(banks: pd.DataFrame) -> pd.DataFrame:
    per_graph = (
        banks.groupby(["Graph Size", "Graph Name", "Sampler"], as_index=False)
        .agg(Mean_Bank_Density=("Induced Density", "mean"))
    )
    return (
        per_graph.groupby(["Graph Size", "Sampler"], as_index=False)
        .agg(
            Graphs=("Graph Name", "size"),
            Mean_Bank_Density=("Mean_Bank_Density", "mean"),
            SD_Bank_Density=("Mean_Bank_Density", "std"),
        )
    )


def calibrated_method(banks: pd.DataFrame, doses: Sequence[int]) -> Tuple[int, pd.DataFrame]:
    pooled = banks.groupby("Sampler", as_index=False).agg(
        Mean_Density=("Induced Density", "mean")
    )
    bgbs_density = float(
        pooled.loc[pooled["Sampler"] == "BipartiteGBS", "Mean_Density"].iloc[0]
    )
    candidates = pooled[pooled["Sampler"].isin([f"DensityDose-m{d}" for d in doses if d > 1])].copy()
    candidates["Absolute_Distance_to_BipartiteGBS"] = (
        candidates["Mean_Density"] - bgbs_density
    ).abs()
    selected_sampler = str(
        candidates.sort_values(["Absolute_Distance_to_BipartiteGBS", "Sampler"]).iloc[0]["Sampler"]
    )
    selected_dose = int(selected_sampler.split("m")[-1])
    pooled["BipartiteGBS_Mean_Density"] = bgbs_density
    pooled["Selected_Calibrated_Dose"] = selected_dose
    return selected_dose, pooled


def paired_rows(
    per_graph: pd.DataFrame,
    first: str,
    second: str,
    sizes: Sequence[int],
    contrast_index: int,
    resamples: int,
) -> List[dict]:
    rows: List[dict] = []
    for n in sizes:
        selected = per_graph[per_graph["Graph Size"] == n]
        pivot = selected.pivot(index="Graph Name", columns="Algorithm", values="Success_Rate")
        counts = selected.pivot(index="Graph Name", columns="Algorithm", values="Success_Count")
        trials = selected.pivot(index="Graph Name", columns="Algorithm", values="Trial_Count")
        if not (trials[first].eq(trials[second]).all() and trials[first].nunique() == 1):
            raise ValueError("Count-based paired ranks require equal trial counts")
        count_differences = (counts[first] - counts[second]).to_numpy(dtype=int)
        differences = 100.0 * (pivot[first] - pivot[second]).to_numpy(dtype=float)
        low, high = base.bootstrap_ci(
            differences, resamples, BOOTSTRAP_SEED + contrast_index * 1_000 + n
        )
        rows.append(
            {
                "Contrast": f"{first} vs {second}",
                "Graph Size": n,
                "Graphs": len(differences),
                "First Rate (%)": 100.0 * float(pivot[first].mean()),
                "Second Rate (%)": 100.0 * float(pivot[second].mean()),
                "Mean Difference (pp)": float(differences.mean()),
                "CI95 Low (pp)": low,
                "CI95 High (pp)": high,
                "Wilcoxon p raw": base.paired_wilcoxon(count_differences),
            }
        )
    adjusted = base.holm_adjust([row["Wilcoxon p raw"] for row in rows])
    for row, p_value in zip(rows, adjusted):
        row["Wilcoxon p Holm"] = p_value
    return rows


def paired_statistics(
    results: pd.DataFrame,
    sizes: Sequence[int],
    doses: Sequence[int],
    selected_dose: int,
    resamples: int,
) -> pd.DataFrame:
    per_graph = per_graph_results(results)
    uniform = method_for_dose(1)
    calibrated = method_for_dose(selected_dose)
    contrasts = [
        (METHOD_BGBS, uniform, "Primary"),
        (calibrated, uniform, "Primary"),
        (METHOD_BGBS, calibrated, "Primary"),
        (uniform, METHOD_GA, "Primary"),
        (METHOD_BGBS, METHOD_GA, "Primary"),
        (calibrated, METHOD_GA, "Primary"),
    ]
    for dose in doses:
        if dose > 1 and dose != selected_dose:
            contrasts.append((method_for_dose(dose), uniform, "Exploratory dose"))
    rows: List[dict] = []
    for index, (first, second, family) in enumerate(contrasts):
        contrast_rows = paired_rows(per_graph, first, second, sizes, index, resamples)
        for row in contrast_rows:
            row["Comparison Family"] = family
            row["Calibrated Density Dose"] = selected_dose
        rows.extend(contrast_rows)
    return pd.DataFrame(rows)


def initial_metric_statistics(
    results: pd.DataFrame,
    sizes: Sequence[int],
    selected_dose: int,
    resamples: int,
) -> pd.DataFrame:
    """Paired diagnostics for how the bank changes the generated population."""
    per_graph = per_graph_results(results)
    uniform = method_for_dose(1)
    calibrated = method_for_dose(selected_dose)
    contrasts = [
        (METHOD_BGBS, uniform),
        (calibrated, uniform),
        (METHOD_BGBS, calibrated),
    ]
    metrics = [
        ("Initial_Mean_Fitness", "Initial mean fitness"),
        ("Initial_Best_Fitness", "Initial best fitness"),
        ("Initial_Cycles", "Initial Hamiltonian cycles per population"),
    ]
    rows: List[dict] = []
    for metric_index, (column, label) in enumerate(metrics):
        for contrast_index, (first, second) in enumerate(contrasts):
            contrast_rows: List[dict] = []
            for n in sizes:
                selected = per_graph[per_graph["Graph Size"] == n]
                pivot = selected.pivot(index="Graph Name", columns="Algorithm", values=column)
                differences = (pivot[first] - pivot[second]).to_numpy(dtype=float)
                low, high = base.bootstrap_ci(
                    differences,
                    resamples,
                    BOOTSTRAP_SEED + 100_000 + metric_index * 10_000 + contrast_index * 1_000 + n,
                )
                contrast_rows.append(
                    {
                        "Metric": label,
                        "Contrast": f"{first} vs {second}",
                        "Graph Size": n,
                        "Graphs": len(differences),
                        "First Mean": float(pivot[first].mean()),
                        "Second Mean": float(pivot[second].mean()),
                        "Mean Difference": float(differences.mean()),
                        "CI95 Low": low,
                        "CI95 High": high,
                        "Wilcoxon p raw": base.paired_wilcoxon(differences),
                    }
                )
            adjusted = base.holm_adjust([row["Wilcoxon p raw"] for row in contrast_rows])
            for row, p_value in zip(contrast_rows, adjusted):
                row["Wilcoxon p Holm"] = p_value
            rows.extend(contrast_rows)
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    results: pd.DataFrame,
    banks: pd.DataFrame,
    config: Dict[str, object],
    workbook: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.sort_values(
        ["Graph Size", "Graph Name", "Bank Replicate", "Algorithm", "Trial"]
    ).to_csv(output_dir / "density_dose_results.csv.gz", index=False)
    banks.sort_values(
        ["Graph Size", "Graph Name", "Bank Replicate", "Sampler", "Sample Index"]
    ).to_csv(output_dir / "density_dose_banks.csv.gz", index=False)
    summarize(results).to_csv(output_dir / "density_dose_summary.csv", index=False)
    density_summary(banks).to_csv(output_dir / "density_dose_bank_density.csv", index=False)
    selected_dose, calibration = calibrated_method(banks, config["doses"])
    calibration.to_csv(output_dir / "density_dose_calibration.csv", index=False)
    stats = paired_statistics(
        results,
        config["sizes"],
        config["doses"],
        selected_dose,
        config["bootstrap_resamples"],
    )
    stats.to_csv(output_dir / "density_dose_stats.csv", index=False)
    initial_stats = initial_metric_statistics(
        results,
        config["sizes"],
        selected_dose,
        config["bootstrap_resamples"],
    )
    initial_stats.to_csv(output_dir / "density_dose_initial_stats.csv", index=False)
    metadata = {
        "description": "Outcome-blind matched density-dose controls",
        "config": config,
        "selected_calibrated_dose": selected_dose,
        "selection_rule": "pooled bank density closest to BipartiteGBS; outcomes unused",
        "sampling_workbook": workbook.name,
        "sampling_workbook_sha256": sha256(workbook),
        "runner": Path(__file__).name,
        "rows": {
            "results": len(results),
            "banks": len(banks),
            "stats": len(stats),
            "initial_stats": len(initial_stats),
        },
    }
    (output_dir / "density_dose_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    display = summarize(results).pivot(
        index="Graph Size", columns="Algorithm", values="Mean_Success_Rate (%)"
    )
    print("\nSuccess rates (%):")
    print(display.round(2).to_string())
    print("\nOutcome-blind density calibration:")
    print(calibration.to_string(index=False))
    print("\nPrimary paired comparisons:")
    print(
        stats[stats["Comparison Family"] == "Primary"][
            [
                "Contrast",
                "Graph Size",
                "Mean Difference (pp)",
                "CI95 Low (pp)",
                "CI95 High (pp)",
                "Wilcoxon p Holm",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    sizes = sorted(set(args.sizes))
    doses = sorted(set(args.doses))
    if not doses or doses[0] != 1 or any(dose < 1 for dose in doses):
        raise ValueError("Density doses must be positive and include m=1")
    if len(doses) < 2:
        raise ValueError("At least one positive density dose m>1 is required")
    if args.bank_replicates < 1 or args.trials < 1:
        raise ValueError("Bank replicates and trials must be positive")
    saved = base.load_saved_banks(args.sampling_workbook, sizes, args.graphs_per_size)
    config: Dict[str, object] = {
        "sizes": sizes,
        "graphs_per_size": args.graphs_per_size,
        "trials": args.trials,
        "bank_replicates": args.bank_replicates,
        "doses": doses,
        "population": args.population,
        "generations": args.generations,
        "crossover_rate": args.crossover_rate,
        "mutation_rate": args.mutation_rate,
        "beta": args.beta,
        "bootstrap_resamples": args.bootstrap_resamples,
    }
    tasks = [
        (n, index, saved[f"Random_n{n}_{index + 1}"], config)
        for n in sizes
        for index in range(args.graphs_per_size)
    ]
    method_count = 2 + len(doses)
    print(
        f"Running {len(tasks)} graphs x {method_count} methods x "
        f"{args.bank_replicates} banks x {args.trials} trials with "
        f"{args.workers} worker(s)...",
        flush=True,
    )
    started = time.time()
    trial_rows: List[dict] = []
    bank_rows: List[dict] = []
    if args.workers == 1:
        iterator: Iterable = map(run_one_graph, tasks)
        pool = None
    else:
        pool = mp.Pool(processes=args.workers)
        iterator = pool.imap_unordered(run_one_graph, tasks, chunksize=1)
    try:
        for completed, (trials, banks) in enumerate(iterator, start=1):
            trial_rows.extend(trials)
            bank_rows.extend(banks)
            if completed % 10 == 0 or completed == len(tasks):
                print(
                    f"  {completed}/{len(tasks)} graphs complete "
                    f"({time.time() - started:.1f}s)",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    results = pd.DataFrame(trial_rows)
    banks = pd.DataFrame(bank_rows)
    expected_results = len(tasks) * method_count * args.bank_replicates * args.trials
    if len(results) != expected_results:
        raise AssertionError(f"Expected {expected_results} results, found {len(results)}")
    expected_bank_rows = (
        sum(len(bank) for bank in saved.values()) * (1 + len(doses)) * args.bank_replicates
    )
    if len(banks) != expected_bank_rows:
        raise AssertionError(f"Expected {expected_bank_rows} bank rows, found {len(banks)}")
    write_outputs(args.output_dir, results, banks, config, args.sampling_workbook)
    print(f"\nFinished in {time.time() - started:.1f}s. Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
