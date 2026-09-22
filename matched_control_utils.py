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

"""Shared functions for the matched subset source control experiment."""

from __future__ import annotations

from collections import Counter
import importlib
import re
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


framework = importlib.import_module("main_benchmark")

DEFAULT_SIZES = [15, 20, 25, 30, 35, 40]
DEFAULT_GRAPHS_PER_SIZE = 60
DEFAULT_TRIALS = 10
GRAPH_MASTER_SEED = 42


def parse_vertex_set(value: object) -> Set[int]:
    vertices = {int(x) for x in re.findall(r"\d+", str(value))}
    if not vertices:
        raise ValueError(f"Could not parse a nonempty vertex set from {value!r}")
    return vertices


def load_saved_banks(
    workbook: Path, sizes: Sequence[int], graphs_per_size: int
) -> Dict[str, List[Set[int]]]:
    """Load accepted vertex subsets in their archived sample order."""
    if not workbook.exists():
        raise FileNotFoundError(f"Saved sampling workbook not found: {workbook}")
    details = pd.read_excel(workbook, sheet_name="Subgraph Details")
    required = {"Graph Name", "Graph Size", "Sample Index", "Subgraph Size", "Vertices"}
    missing = required - set(details.columns)
    if missing:
        raise ValueError(f"Missing columns in Subgraph Details: {sorted(missing)}")

    wanted_names = {
        f"Random_n{n}_{index + 1}"
        for n in sizes
        for index in range(graphs_per_size)
    }
    details = details[details["Graph Name"].isin(wanted_names)].copy()
    details.sort_values(["Graph Size", "Graph Name", "Sample Index"], inplace=True)

    banks: Dict[str, List[Set[int]]] = {}
    for graph_name, group in details.groupby("Graph Name", sort=False):
        samples = [parse_vertex_set(value) for value in group["Vertices"]]
        if [len(sample) for sample in samples] != group["Subgraph Size"].astype(int).tolist():
            raise ValueError(f"Saved size mismatch in {graph_name}")
        banks[str(graph_name)] = samples

    missing_names = sorted(wanted_names - set(banks))
    if missing_names:
        raise ValueError(
            f"Missing {len(missing_names)} saved banks; first entries: {missing_names[:5]}"
        )
    if any(not bank for bank in banks.values()):
        raise ValueError("Every selected graph must have an accepted subset bank")
    return banks


def graph_seed(n: int, index: int) -> int:
    return GRAPH_MASTER_SEED + index * 100 + n * 10


def guidance_from_bank(graph: object, samples: Sequence[Set[int]]) -> object:
    """Apply the co-occurrence estimator used by the main BipartiteGBS pipeline."""
    if not samples:
        raise ValueError("A guidance bank must be nonempty")
    vertex_freq: Counter = Counter()
    edge_freq: Counter = Counter()
    size_freq: Counter = Counter()
    for vertices in samples:
        size_freq[len(vertices)] += 1
        vertex_freq.update(vertices)
        ordered = sorted(vertices)
        for position, u in enumerate(ordered):
            for v in ordered[position + 1 :]:
                if graph.has_edge(u, v):
                    edge_freq[(u, v)] += 1
                if graph.has_edge(v, u):
                    edge_freq[(v, u)] += 1

    total = len(samples)
    return framework.GBSInfo(
        vertex_freq=vertex_freq,
        edge_freq=edge_freq,
        path_freq=Counter(),
        vertex_prob={vertex: count / total for vertex, count in vertex_freq.items()},
        edge_prob={edge: count / total for edge, count in edge_freq.items()},
        top_vertices=[vertex for vertex, _ in vertex_freq.most_common(graph.n)],
        top_edges=[edge for edge, _ in edge_freq.most_common(min(graph.n * 2, len(edge_freq)))],
        top_paths=[],
        samples=[set(sample) for sample in samples],
        sample_count=total,
        effective_samples=total,
        max_subgraph_size=max(map(len, samples)),
        avg_subgraph_size=float(np.mean([len(sample) for sample in samples])),
        subgraph_size_distribution={size: count / total for size, count in size_freq.items()},
    )


def induced_edge_count(graph: object, subset: Set[int]) -> int:
    return sum(
        1
        for u in subset
        for v in subset
        if u != v and graph.has_edge(u, v)
    )


def bank_rows(
    graph: object,
    sampler: str,
    bank: Sequence[Set[int]],
    bank_seed: int | None,
) -> List[dict]:
    rows = []
    for sample_index, subset in enumerate(bank, start=1):
        size = len(subset)
        edges = induced_edge_count(graph, subset)
        denominator = size * (size - 1)
        rows.append(
            {
                "Graph Name": graph.name,
                "Graph Size": graph.n,
                "Sampler": sampler,
                "Sample Index": sample_index,
                "Subgraph Size": size,
                "Vertices": "{" + ",".join(str(v) for v in sorted(subset)) + "}",
                "Induced Directed Edges": edges,
                "Induced Density": edges / denominator if denominator else 0.0,
                "Bank Seed": bank_seed,
            }
        )
    return rows


def make_init_only(args: dict) -> object:
    return framework.GBS_InitOnly(
        pop_size=args["population"],
        max_generations=args["generations"],
        crossover_rate=args["crossover_rate"],
        mutation_rate=args["mutation_rate"],
        beta=args["beta"],
    )


def bootstrap_ci(values: np.ndarray, resamples: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = np.empty(resamples, dtype=float)
    batch = 1_000
    for start in range(0, resamples, batch):
        stop = min(start + batch, resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_wilcoxon(differences: np.ndarray) -> float:
    nonzero = np.asarray(differences)[np.asarray(differences) != 0]
    if len(nonzero) == 0:
        return 1.0
    return float(wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided").pvalue)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = (count - rank) * float(p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = min(1.0, running)
    return adjusted.tolist()
