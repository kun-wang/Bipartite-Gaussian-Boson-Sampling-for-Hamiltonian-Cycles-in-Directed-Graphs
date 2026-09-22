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

"""Run the fitness-weight sensitivity experiment for directed graphs.

The script performs BipartiteGBS sampling, scans the fitness weight alpha,
stores the trial records, and plots the resulting success rates.
"""

import os
os.environ['NUMBA_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Set, Any
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
from collections import Counter, defaultdict
import itertools
import warnings
import pandas as pd
from datetime import datetime

# Import the optional Strawberry Fields backend.
try:
    import strawberryfields as sf
    from strawberryfields.ops import S2gate, Interferometer, MeasureFock

    SF_AVAILABLE = True
except ImportError:
    print("Warning: StrawberryFields not installed. BipartiteGBS will use simulation mode.")
    SF_AVAILABLE = False

warnings.filterwarnings('ignore')


# ==================== 1. Data structures ====================

@dataclass
class ExperimentResult:
    """Store one algorithm trial."""
    algorithm_name: str
    success: bool
    cycle: List[int]
    path_length: int
    longest_path: List[int] = field(default_factory=list)
    time_seconds: float = 0.0
    generations: int = 0
    gbs_used: bool = False
    graph_type: str = ""
    graph_size: int = 0
    alpha: float = None
    density: float = None


@dataclass
class GBSInfo:
    """Store statistics derived from accepted BipartiteGBS samples."""
    vertex_freq: Counter
    edge_freq: Counter
    path_freq: Counter
    vertex_prob: Dict[int, float]
    edge_prob: Dict[Tuple[int, int], float]
    top_vertices: List[int]
    top_edges: List[Tuple[int, int]]
    top_paths: List[Tuple[int, ...]]
    samples: List[Set[int]] = field(default_factory=list)


class DirectedGraph:
    """Represent a directed graph and its adjacency data."""

    def __init__(self, n: int, edges: List[Tuple[int, int]], name: str = ""):
        self.n = n
        self.edges = edges
        self.name = name
        self.adj_list = {i: [] for i in range(n)}
        self.adj_set = set()
        for u, v in edges:
            self.adj_list[u].append(v)
            self.adj_set.add((u, v))

    def get_out_neighbors(self, v: int) -> List[int]:
        return self.adj_list[v]

    def has_edge(self, u: int, v: int) -> bool:
        return (u, v) in self.adj_set

    def get_density(self) -> float:
        return len(self.edges) / (self.n * (self.n - 1)) if self.n > 1 else 0

    def to_adjacency_matrix(self) -> np.ndarray:
        A = np.zeros((self.n, self.n))
        for u, v in self.edges:
            A[u, v] = 1
        return A

    def summary(self) -> str:
        return f"{self.name}: n={self.n}, edges={len(self.edges)}, density={self.get_density():.3f}"


# ==================== 2. BipartiteGBS sampler ====================

class BipartiteGBSSampler:
    def __init__(self, graph: DirectedGraph):
        self.graph = graph
        self.n = graph.n
        self.adj_matrix = graph.to_adjacency_matrix()

        self.B = np.zeros((self.n, self.n), dtype=np.complex128)
        for i in range(self.n):
            for j in range(self.n):
                if self.adj_matrix[i, j] > 0:
                    self.B[i, j] = 1.0

        self.m = self.n
        self.samples_cache = None

    def _prepare_gbs(self):
        U, s, Vt = np.linalg.svd(self.B, full_matrices=False)
        print(np.max(s))
        scale = 0.75 /  np.max(s)
        self.U = U
        self.V = Vt.T.conj()
        self.r = np.arctanh(scale * s)
        print("r without clip:", self.r)
        self.singular_values = s

    def sample(self, num_samples: int = 500, use_cache: bool = True) -> List[Set[int]]:
        if use_cache and self.samples_cache is not None:
            return self.samples_cache

        samples = self._real_gbs_sampling(num_samples)

        if use_cache:
            self.samples_cache = samples

        return samples

    def _real_gbs_sampling(self, num_samples: int) -> List[Set[int]]:
        if not SF_AVAILABLE:
            raise RuntimeError("Quantum sampling requested but Strawberry Fields is unavailable; use saved results or install the backend.")

        self._prepare_gbs()
        prog = sf.Program(2 * self.m)

        with prog.context as q:
            for i in range(self.m):
                if self.r[i] > 0:
                    S2gate(self.r[i]) | (q[i], q[i + self.m])
            if self.U.shape[0] > 0:
                Interferometer(self.U) | q[:self.m]
            if self.V.shape[0] > 0:
                Interferometer(self.V) | q[self.m:]
            MeasureFock() | q

        # Batched sampling: one engine call for all shots (identical distribution,
        # but far faster than one call per shot) and matches main_benchmark.py.
        results = sf.Engine("gaussian").run(prog, shots=num_samples)
        samples = []
        successful_samples = 0

        for photon_pattern in results.samples:
            s_photons = photon_pattern[:self.m]
            t_photons = photon_pattern[self.m:]

            if any(p > 1 for p in s_photons) or any(p > 1 for p in t_photons):
                continue

            S = {i for i, p in enumerate(s_photons) if p == 1}
            T = {i for i, p in enumerate(t_photons) if p == 1}

            if len(S) == len(T) and len(S) >= 3:
                vertices = S.union(T)
                samples.append(vertices)
                successful_samples += 1

        return samples


    def get_gbs_info(self, num_samples: int = 500) -> GBSInfo:
        samples = self.sample(num_samples)

        vertex_freq = Counter()
        edge_freq = Counter()
        path_freq = Counter()

        for vertices in samples:
            for v in vertices:
                vertex_freq[v] += 1

            vertices_list = sorted(vertices)
            for i in range(len(vertices_list)):
                for j in range(i + 1, len(vertices_list)):
                    u, v = vertices_list[i], vertices_list[j]
                    if self.graph.has_edge(u, v):
                        edge_freq[(u, v)] += 1
                    if self.graph.has_edge(v, u):
                        edge_freq[(v, u)] += 1

        total = len(samples)
        vertex_prob = {v: freq / total for v, freq in vertex_freq.items()}
        edge_prob = {e: freq / total for e, freq in edge_freq.items()}
        top_edges = [e for e, _ in edge_freq.most_common(min(self.n * 2, len(edge_freq)))]

        return GBSInfo(
            vertex_freq=vertex_freq,
            edge_freq=edge_freq,
            path_freq=path_freq,
            vertex_prob=vertex_prob,
            edge_prob=edge_prob,
            top_vertices=[v for v, _ in vertex_freq.most_common(min(self.n, self.n))],
            top_edges=top_edges,
            top_paths=[],
            samples=samples
        )


# ==================== 3. Graph generator ====================

class GraphGenerator:
    @staticmethod
    def generate_random_digraph(n: int, p: float, seed: int = None) -> DirectedGraph:
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and random.random() < p:
                    edges.append((i, j))

        return DirectedGraph(n, edges, f"Random_n{n}_p{p}")


# ==================== 4. Hamiltonian-cycle validator ====================

class HamiltonianValidator:
    @staticmethod
    def verify_cycle(graph: DirectedGraph, cycle: List[int]) -> bool:
        if len(cycle) != graph.n:
            return False
        if len(set(cycle)) != graph.n:
            return False
        for i in range(graph.n):
            u = cycle[i]
            v = cycle[(i + 1) % graph.n]
            if not graph.has_edge(u, v):
                return False
        return True

    @staticmethod
    def extract_longest_path(graph: DirectedGraph, sequence: List[int]) -> List[int]:
        if not sequence:
            return []
        longest = [sequence[0]]
        current = [sequence[0]]
        for i in range(1, len(sequence)):
            if graph.has_edge(sequence[i - 1], sequence[i]):
                current.append(sequence[i])
            else:
                if len(current) > len(longest):
                    longest = current
                current = [sequence[i]]
        if len(current) > len(longest):
            longest = current
        return longest


# ==================== 5. Search algorithms ====================

class HamiltonianAlgorithm(ABC):
    def __init__(self, name: str):
        self.name = name
        self.validator = HamiltonianValidator()

    @abstractmethod
    def search(self, graph: DirectedGraph, gbs_info: GBSInfo = None, alpha: float = None) -> ExperimentResult:
        pass


class GeneticAlgorithm(HamiltonianAlgorithm):
    def __init__(self, pop_size: int, max_generations: int,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        super().__init__("GeneticAlgorithm")
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def search(self, graph: DirectedGraph, gbs_info: GBSInfo = None, alpha: float = None) -> ExperimentResult:
        start_time = time.time()
        population = self._init_population(graph)
        best_individual = None
        best_fitness = 0
        best_cycle = None

        for generation in range(self.max_generations):
            fitness = [self._fitness(graph, ind) for ind in population]
            max_idx = np.argmax(fitness)
            if fitness[max_idx] > best_fitness:
                best_fitness = fitness[max_idx]
                best_individual = population[max_idx][:]
                if self.validator.verify_cycle(graph, best_individual):
                    best_cycle = best_individual[:]

            if best_cycle is not None:
                return ExperimentResult(
                    algorithm_name=self.name,
                    success=True,
                    cycle=best_cycle,
                    path_length=graph.n,
                    longest_path=best_cycle,
                    time_seconds=time.time() - start_time,
                    generations=generation,
                    gbs_used=False,
                    graph_type=graph.name,
                    graph_size=graph.n,
                    alpha=alpha
                )

            new_population = []
            for _ in range(self.pop_size):
                parent1 = self._tournament_select(population, fitness)
                parent2 = self._tournament_select(population, fitness)
                if random.random() < self.crossover_rate:
                    child = self._order_crossover(parent1, parent2)
                else:
                    child = parent1[:]
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                new_population.append(child)
            population = new_population

        longest_path = self.validator.extract_longest_path(graph, best_individual)
        return ExperimentResult(
            algorithm_name=self.name,
            success=False,
            cycle=[],
            path_length=len(longest_path),
            longest_path=longest_path,
            time_seconds=time.time() - start_time,
            generations=self.max_generations,
            gbs_used=False,
            graph_type=graph.name,
            graph_size=graph.n,
            alpha=alpha
        )

    def _init_population(self, graph: DirectedGraph) -> List[List[int]]:
        population = []
        for _ in range(self.pop_size):
            ind = list(range(graph.n))
            random.shuffle(ind)
            population.append(ind)
        return population

    def _fitness(self, graph: DirectedGraph, individual: List[int]) -> float:
        length = 0
        for i in range(graph.n - 1):
            if graph.has_edge(individual[i], individual[i + 1]):
                length += 1
        if graph.has_edge(individual[-1], individual[0]):
            length += 1
        return length / graph.n

    def _tournament_select(self, population: List[List[int]],
                           fitness: List[float], k: int = 3) -> List[int]:
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(indices, key=lambda i: fitness[i])
        return population[best_idx][:]

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        n = len(parent1)
        child = [-1] * n
        start = random.randint(0, n - 2)
        end = random.randint(start + 1, n - 1)
        child[start:end + 1] = parent1[start:end + 1]
        pos = (end + 1) % n
        for i in range(n):
            idx = (end + 1 + i) % n
            if parent2[idx] not in child:
                child[pos] = parent2[idx]
                pos = (pos + 1) % n
        return child

    def _mutate(self, individual: List[int]) -> List[int]:
        n = len(individual)
        i, j = random.sample(range(n), 2)
        individual[i], individual[j] = individual[j], individual[i]
        return individual


class GBSEnhancedGenetic(GeneticAlgorithm):
    """Genetic algorithm guided by BipartiteGBS sample statistics."""

    def __init__(self, pop_size: int, max_generations: int,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 alpha: float = 0.3, beta: float = 0.2):
        super().__init__(pop_size, max_generations, crossover_rate, mutation_rate)
        self.name = "GBS-Genetic"
        self.alpha = alpha
        self.beta = beta

    def search(self, graph: DirectedGraph, gbs_info: GBSInfo = None, alpha: float = None) -> ExperimentResult:
        if gbs_info is None:
            return super().search(graph, None, alpha)

        actual_alpha = alpha if alpha is not None else self.alpha

        start_time = time.time()
        population = self._gbs_guided_init_population(graph, gbs_info)
        best_individual = None
        best_fitness = 0
        best_cycle = None

        for generation in range(self.max_generations):
            current_alpha = self.alpha * (generation / self.max_generations)
            fitness = [self._gbs_enhanced_fitness(graph, ind, gbs_info, current_alpha)
                       for ind in population]

            max_idx = np.argmax(fitness)
            if fitness[max_idx] > best_fitness:
                best_fitness = fitness[max_idx]
                best_individual = population[max_idx][:]
                if self.validator.verify_cycle(graph, best_individual):
                    best_cycle = best_individual[:]

            if best_cycle is not None:
                return ExperimentResult(
                    algorithm_name=self.name,
                    success=True,
                    cycle=best_cycle,
                    path_length=graph.n,
                    longest_path=best_cycle,
                    time_seconds=time.time() - start_time,
                    generations=generation,
                    gbs_used=True,
                    graph_type=graph.name,
                    graph_size=graph.n,
                    alpha=actual_alpha
                )

            new_population = []
            for _ in range(self.pop_size):
                parent1 = self._tournament_select(population, fitness)
                parent2 = self._tournament_select(population, fitness)
                if random.random() < self.crossover_rate:
                    child = self._order_crossover(parent1, parent2)
                else:
                    child = parent1[:]
                if random.random() < self.mutation_rate:
                    child = self._gbs_guided_mutate(child, gbs_info)
                new_population.append(child)
            population = new_population

        longest_path = self.validator.extract_longest_path(graph, best_individual)
        return ExperimentResult(
            algorithm_name=self.name,
            success=False,
            cycle=[],
            path_length=len(longest_path),
            longest_path=longest_path,
            time_seconds=time.time() - start_time,
            generations=self.max_generations,
            gbs_used=True,
            graph_type=graph.name,
            graph_size=graph.n,
            alpha=actual_alpha
        )

    def _gbs_guided_init_population(self, graph: DirectedGraph, gbs_info: GBSInfo) -> List[List[int]]:
        population = []
        n = graph.n

        for _ in range(self.pop_size // 3):
            ind = list(range(n))
            random.shuffle(ind)
            population.append(ind)

        if gbs_info.samples:
            for sample in gbs_info.samples[:self.pop_size // 3]:
                ind = self._construct_from_subgraph(graph, sample, gbs_info)
                if ind:
                    population.append(ind)

        for _ in range(self.pop_size // 3):
            ind = self._construct_hybrid(graph, gbs_info)
            if ind:
                population.append(ind)

        while len(population) < self.pop_size:
            ind = list(range(n))
            random.shuffle(ind)
            population.append(ind)

        return population[:self.pop_size]

    def _construct_from_subgraph(self, graph: DirectedGraph, vertices: Set[int], gbs_info: GBSInfo) -> List[int]:
        n = graph.n
        vertices_list = sorted(vertices)
        if len(vertices_list) < 3:
            return None

        used = set()
        path = []
        start = vertices_list[0]
        path.append(start)
        used.add(start)

        current = start
        remaining = set(vertices_list) - used

        while remaining:
            candidates = []
            for v in remaining:
                if graph.has_edge(current, v):
                    prob = gbs_info.edge_prob.get((current, v), 0.0)
                    candidates.append((v, prob))

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                next_v = candidates[0][0]
                path.append(next_v)
                used.add(next_v)
                current = next_v
                remaining.remove(next_v)
            else:
                break

        all_vertices = set(range(n))
        remaining_all = all_vertices - used
        remaining_list = list(remaining_all)
        random.shuffle(remaining_list)
        path.extend(remaining_list)
        return path

    def _construct_hybrid(self, graph: DirectedGraph, gbs_info: GBSInfo) -> List[int]:
        n = graph.n
        used = set()
        path = []

        if gbs_info.samples and random.random() < 0.5:
            sample = random.choice(gbs_info.samples[:min(50, len(gbs_info.samples))])
            start = random.choice(list(sample))
            path.append(start)
            used.add(start)
        else:
            start = random.randint(0, n - 1)
            path.append(start)
            used.add(start)

        while len(path) < n:
            last = path[-1]

            if random.random() < self.beta and gbs_info.top_edges:
                found = False
                for u, v in gbs_info.top_edges[:n]:
                    if u == last and v not in used:
                        path.append(v)
                        used.add(v)
                        found = True
                        break
                if found:
                    continue

            neighbors = [v for v in graph.get_out_neighbors(last) if v not in used]
            if neighbors:
                path.append(random.choice(neighbors))
                used.add(path[-1])
            else:
                remaining = [v for v in range(n) if v not in used]
                if remaining:
                    path.append(random.choice(remaining))
                    used.add(path[-1])
                else:
                    break

        remaining = [v for v in range(n) if v not in path]
        path.extend(remaining)
        return path

    def _gbs_enhanced_fitness(self, graph: DirectedGraph, individual: List[int],
                              gbs_info: GBSInfo, alpha: float) -> float:
        length = 0
        for i in range(graph.n - 1):
            if graph.has_edge(individual[i], individual[i + 1]):
                length += 1

        can_close = graph.has_edge(individual[-1], individual[0])
        if can_close:
            length += 1
            if length == graph.n:
                return 1.0

        base_fitness = length / graph.n

        # Average co-occurrence over ALL proposed edges; unsampled edges default to 0
        # (matches Alg. 7 in the manuscript and the main_benchmark.py implementation).
        edge_prob = gbs_info.edge_prob if gbs_info else {}
        gbs_score = 0.0
        edge_count = 0
        for i in range(graph.n - 1):
            gbs_score += edge_prob.get((individual[i], individual[i + 1]), 0.0)
            edge_count += 1
        if can_close:
            gbs_score += edge_prob.get((individual[-1], individual[0]), 0.0)
            edge_count += 1
        gbs_score = gbs_score / edge_count if edge_count > 0 else 0.0

        combined = base_fitness * (1 - alpha) + gbs_score * alpha
        return combined

    def _gbs_guided_mutate(self, individual: List[int], gbs_info: GBSInfo) -> List[int]:
        n = len(individual)

        if gbs_info and gbs_info.edge_prob and random.random() < 0.7:
            edge_scores = []
            for i in range(n):
                u = individual[i]
                v = individual[(i + 1) % n]
                score = gbs_info.edge_prob.get((u, v), 0.0)
                edge_scores.append((i, score))

            edge_scores.sort(key=lambda x: x[1])
            if edge_scores:
                idx = edge_scores[0][0]
                jdx = random.randint(0, n - 1)
                individual[idx], individual[jdx] = individual[jdx], individual[idx]
                return individual

        i, j = random.sample(range(n), 2)
        individual[i], individual[j] = individual[j], individual[i]
        return individual


# ==================== 6. Alpha-sensitivity experiment ====================

class AlphaSensitivityRunner:
    """Run the fitness-weight sensitivity experiment."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.results = []
        self.excel_manager = None

    def run_alpha_sensitivity(self, n: int = 25, num_graphs: int = 30,
                              n_trials: int = 5, alpha_values: List[float] = None,
                              densities: List[float] = None):
        """
        Run the alpha hyperparameter sensitivity experiment.

        Parameters:
        -----------
        n: Number of graph vertices.
        num_graphs: Number of graphs at each density.
        n_trials: Number of trials per graph and algorithm.
        alpha_values: Fitness-weight values.
        densities: Directed edge probabilities.
        """
        if alpha_values is None:
            alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        if densities is None:
            densities = [0.25, 0.30, 0.35]

        print("\n" + "=" * 80)
        print("α HYPERPARAMETER SENSITIVITY ANALYSIS")
        print("=" * 80)
        print(f"Graph size: n={n}")
        print(f"Graph densities: {densities}")
        print(f"Number of graphs per density: {num_graphs}")
        print(f"Trials per graph: {n_trials}")
        print(f"α values: {alpha_values}")
        print("=" * 80)

        # Run the experiment at each graph density.
        for density in densities:
            print(f"\n{'=' * 60}")
            print(f"Testing density = {density}")
            print(f"{'=' * 60}")

            # Generate the graph ensemble.
            graphs = []
            for i in range(num_graphs):
                seed = self.seed + i * 100 + int(density * 1000)
                graph = GraphGenerator.generate_random_digraph(n, density, seed)
                graph.name = f"Random_n{n}_d{density}_{i + 1}"
                graphs.append(graph)

            # Prepare BipartiteGBS guidance.
            print("  Preparing GBS information...")
            gbs_infos = []
            for idx, graph in enumerate(graphs):
                sampler = BipartiteGBSSampler(graph)
                gbs_info = sampler.get_gbs_info(num_samples=500)
                gbs_infos.append(gbs_info)
                if (idx + 1) % 10 == 0:
                    print(f"    Processed {idx + 1}/{num_graphs} graphs")

            # Evaluate every alpha value.
            for alpha in alpha_values:
                print(f"\n  Testing α = {alpha:.2f}...")

                # Initialize both algorithms.
                algo_standard = GeneticAlgorithm(pop_size=100, max_generations=200)
                algo_gbs = GBSEnhancedGenetic(pop_size=100, max_generations=200, alpha=alpha, beta=0.2)

                for graph_idx, (graph, gbs_info) in enumerate(zip(graphs, gbs_infos)):
                    # Evaluate the standard GA baseline.
                    for trial in range(n_trials):
                        result = algo_standard.search(graph, None, alpha=alpha)
                        result.graph_type = graph.name
                        result.graph_size = n
                        result.alpha = alpha
                        result.density = density
                        self.results.append(result)

                    # Evaluate the BipartiteGBS-guided GA.
                    for trial in range(n_trials):
                        result = algo_gbs.search(graph, gbs_info, alpha=alpha)
                        result.graph_type = graph.name
                        result.graph_size = n
                        result.alpha = alpha
                        result.density = density
                        self.results.append(result)

                    if (graph_idx + 1) % 10 == 0:
                        print(f"      Processed {graph_idx + 1}/{num_graphs} graphs")

            # Print the summary for this density.
            self._print_density_summary(density, alpha_values)

        print(f"\n{'=' * 80}")
        print("α SENSITIVITY ANALYSIS COMPLETED")
        print(f"Total experiments: {len(self.results)}")
        print(f"{'=' * 80}")

        return self.results

    def _print_density_summary(self, density: float, alpha_values: List[float]):
        """Print a statistical summary for one density."""
        print(f"\n  Summary for density={density}:")
        print(f"  {'α':>6} | {'GBS-Genetic SR':>15} | {'Std GA SR':>12}")
        print(f"  {'-' * 40}")

        for alpha in alpha_values:
            gbs_results = [r for r in self.results
                           if r.algorithm_name == "GBS-Genetic"
                           and r.density == density
                           and r.alpha == alpha]
            ga_results = [r for r in self.results
                          if r.algorithm_name == "GeneticAlgorithm"
                          and r.density == density
                          and r.alpha == alpha]

            gbs_sr = np.mean([r.success for r in gbs_results]) * 100 if gbs_results else 0
            ga_sr = np.mean([r.success for r in ga_results]) * 100 if ga_results else 0

            print(f"  {alpha:6.2f} | {gbs_sr:14.1f}% | {ga_sr:11.1f}%")

    def save_to_excel(self, filename: str = "alpha_sensitivity_results.xlsx"):
        """Save trial records, summary statistics, and metadata."""
        data = []
        for result in self.results:
            data.append({
                'Graph Name': result.graph_type,
                'Graph Size': result.graph_size,
                'Density': result.density,
                'Algorithm': result.algorithm_name,
                'Success': 1 if result.success else 0,
                'Path Length': result.path_length,
                'Longest Path Length': len(result.longest_path) if result.longest_path else 0,
                'Time (seconds)': result.time_seconds,
                'Generations': result.generations,
                'GBS Used': result.gbs_used,
                'Alpha': result.alpha
            })

        df = pd.DataFrame(data)

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Raw Data', index=False)

            # Summary table.
            summary = df.groupby(['Density', 'Alpha', 'Algorithm']).agg({
                'Success': ['count', 'mean'],
                'Generations': 'mean',
                'Longest Path Length': 'mean'
            }).round(4)
            summary.to_excel(writer, sheet_name='Summary')

            # Metadata.
            metadata = pd.DataFrame({
                'Property': ['Experiment Date', 'Total Experiments',
                             'Graph Sizes', 'Densities', 'Alpha Values', 'Seed'],
                'Value': [datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          len(df), ', '.join(map(str, sorted(df['Graph Size'].unique()))),
                          ', '.join(map(str, sorted(df['Density'].unique()))),
                          ', '.join(map(str, sorted(df['Alpha'].unique()))),
                          self.seed]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)

        print(f"\nResults saved to {filename}")
        return df


# ==================== 7. Visualization ====================

class AlphaSensitivityVisualizer:
    """Plot the fitness-weight sensitivity results."""

    @staticmethod
    def plot_alpha_curves(df: pd.DataFrame, save_path: str = None):
        """
        Plot success rate against alpha with binomial standard errors.
        """
        if df is None or len(df) == 0:
            print("No data to plot")
            return None, None

        # Check the required columns.
        if 'Density' not in df.columns:
            print("Error: 'Density' column not found in DataFrame")
            return None, None

        densities = sorted(df['Density'].unique())
        n_densities = len(densities)

        fig, axes = plt.subplots(1, n_densities, figsize=(5 * n_densities, 5))
        if n_densities == 1:
            axes = [axes]

        colors = {'GeneticAlgorithm': '#2E86AB', 'GBS-Genetic': '#A23B72'}
        markers = {'GeneticAlgorithm': 's', 'GBS-Genetic': 'o'}

        for idx, density in enumerate(densities):
            ax = axes[idx]
            subset = df[df['Density'] == density]

            for algo in ['GeneticAlgorithm', 'GBS-Genetic']:
                algo_data = subset[subset['Algorithm'] == algo]

                if len(algo_data) == 0:
                    print(f"Warning: No data for {algo} at density {density}")
                    continue

                alpha_groups = algo_data.groupby('Alpha')

                alphas = []
                success_rates = []
                errors = []  # Binomial standard errors.

                for alpha, group in alpha_groups:
                    alphas.append(alpha)

                    # Compute the success rate and binomial standard error.
                    total_trials = len(group)
                    successes = group['Success'].sum()

                    if total_trials > 0:
                        p = successes / total_trials
                        success_rate = p * 100

                        # Binomial standard error: sqrt(p*(1-p)/n) * 100.
                        se = np.sqrt(p * (1 - p) / total_trials) * 100
                    else:
                        success_rate = 0
                        se = 0

                    success_rates.append(success_rate)
                    errors.append(se)

                # Sort by alpha.
                sorted_indices = np.argsort(alphas)
                alphas = np.array(alphas)[sorted_indices]
                success_rates = np.array(success_rates)[sorted_indices]
                errors = np.array(errors)[sorted_indices]

                label = 'GA' if algo == 'GeneticAlgorithm' else 'BipartiteGBS-GA'
                ax.plot(alphas, success_rates, marker=markers[algo],
                        color=colors[algo], linewidth=2, markersize=8, label=label)
                ax.fill_between(alphas, success_rates - errors, success_rates + errors,
                                alpha=0.2, color=colors[algo])

            ax.set_xlabel('α Value', fontsize=11, fontweight='bold')

            if idx == 0:
                ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel('')
            ax.set_title(f'Density = {density}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(0, 105)

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_alpha_curves.pdf", dpi=150, bbox_inches='tight')
            print(f"Alpha curves saved to {save_path}_alpha_curves.pdf")

        plt.show()
        return fig, axes



def main():
    """Run the alpha scan, save the records, and plot the results."""

    print("\n" + "=" * 80)
    print("α HYPERPARAMETER SENSITIVITY ANALYSIS FOR GBS-ENHANCED GA")
    print("=" * 80)

    # Experiment configuration.
    config = {
        'n': 25,  # Number of vertices.
        'num_graphs': 30,  # Number of graphs at each density.
        'n_trials': 5,  # Number of trials per graph.
        'alpha_values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'densities': [0.25, 0.30, 0.35]
    }

    excel_file = "alpha_sensitivity_results.xlsx"

    # ==================== Phase 1: run experiments ====================
    print("\n" + "=" * 80)
    print("PHASE 1: RUNNING EXPERIMENTS")
    print("=" * 80)

    print("\nExperiment Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Run the experiments.
    runner = AlphaSensitivityRunner(seed=42)
    results = runner.run_alpha_sensitivity(**config)

    # Save the records.
    print("\n" + "=" * 80)
    print("PHASE 2: SAVING RESULTS TO EXCEL")
    print("=" * 80)

    df = runner.save_to_excel(excel_file)

    # ==================== Phase 2: reload and visualize ====================
    print("\n" + "=" * 80)
    print("PHASE 3: VISUALIZING RESULTS FROM EXCEL")
    print("=" * 80)

    try:
        # Reload the workbook to verify that it is readable.
        df = pd.read_excel(excel_file)
        print(f"\nSuccessfully loaded data from {excel_file}")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Algorithms: {df['Algorithm'].unique()}")
        print(f"Alpha values: {sorted(df['Alpha'].unique())}")

        if 'Density' in df.columns:
            print(f"Densities: {df['Density'].unique()}")

    except Exception as e:
        print(f"\nError loading data from Excel: {e}")
        return runner, None

    # Plot the analysis.
    if df is not None and len(df) > 0:
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS...")
        print("=" * 80)

        # Call the plotting method directly.
        AlphaSensitivityVisualizer.plot_alpha_curves(df, save_path="alpha_sensitivity")

    else:
        print("\nNo data available for visualization")

    print("\n" + "=" * 80)
    print("EXPERIMENT AND VISUALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return runner, df


if __name__ == "__main__":
    runner, df = main()


