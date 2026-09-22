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

"""Run the main directed Hamiltonian-cycle benchmark.

The implementation uses a single default probability for unobserved edges,
records accepted-sample rates, and supports paired Wilcoxon comparisons.
"""

import os
os.environ['NUMBA_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Set, Any
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
from collections import Counter, defaultdict
import warnings
import pandas as pd
from datetime import datetime
import sys

try:
    import strawberryfields as sf
    from strawberryfields.ops import S2gate, Interferometer, MeasureFock
    SF_AVAILABLE = True
    print("StrawberryFields loaded successfully!")
except ImportError:
    print("Strawberry Fields is unavailable; quantum sampling will raise an error. Saved-data analysis remains available.")
    SF_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed; statistical tests will be skipped.")

warnings.filterwarnings('ignore')

# ==================== Global constants ====================
DEFAULT_EDGE_PROB = 0.0   # Default co-occurrence probability for unobserved edges.

# ==================== 1. Data structures ====================
@dataclass
class ExperimentResult:
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

@dataclass
class GBSInfo:
    vertex_freq: Counter
    edge_freq: Counter
    path_freq: Counter
    vertex_prob: Dict[int, float]
    edge_prob: Dict[Tuple[int, int], float]
    top_vertices: List[int]
    top_edges: List[Tuple[int, int]]
    top_paths: List[Tuple[int, ...]]
    samples: List[Set[int]] = field(default_factory=list)
    sample_count: int = 0
    effective_samples: int = 0
    max_subgraph_size: int = 0
    avg_subgraph_size: float = 0.0
    subgraph_size_distribution: Dict[int, float] = field(default_factory=dict)

@dataclass
class GBSSamplingResult:
    graph_name: str
    graph_size: int
    total_samples: int
    effective_samples: int
    avg_subgraph_size: float
    max_subgraph_size: int
    vertex_probabilities: Dict[int, float]
    edge_probabilities: Dict[Tuple[int, int], float]
    top_vertices: List[int]
    top_edges: List[Tuple[int, int]]
    all_samples: List[Set[int]] = field(default_factory=list)
    sample_time: float = 0.0

# ==================== 2. Graph classes ====================
class DirectedGraph:
    def __init__(self, n: int, edges: List[Tuple[int, int]], name: str = ""):
        self.n = n
        self.edges = edges
        self.name = name
        self.adj_list = {i: [] for i in range(n)}
        self.adj_set = set()
        for u, v in edges:
            self.adj_list[u].append(v)
            self.adj_set.add((u, v))
        self._edge_count = len(edges)

    def get_out_neighbors(self, v: int) -> List[int]:
        return self.adj_list[v]

    def has_edge(self, u: int, v: int) -> bool:
        return (u, v) in self.adj_set

    def get_density(self) -> float:
        return len(self.edges) / (self.n * (self.n - 1)) if self.n > 1 else 0

    def get_min_outdegree(self) -> int:
        return min(len(self.adj_list[v]) for v in range(self.n))

    def get_max_outdegree(self) -> int:
        return max(len(self.adj_list[v]) for v in range(self.n))

    def is_regular(self) -> bool:
        outdegrees = [len(self.adj_list[v]) for v in range(self.n)]
        return len(set(outdegrees)) == 1

    def to_adjacency_matrix(self) -> np.ndarray:
        A = np.zeros((self.n, self.n))
        for u, v in self.edges:
            A[u, v] = 1
        return A

    def induced_subgraph_edge_count(self, vertices: Set[int]) -> int:
        if len(vertices) < 2:
            return 0
        verts = list(vertices)
        count = 0
        for i in range(len(verts)):
            for j in range(len(verts)):
                if i != j and (verts[i], verts[j]) in self.adj_set:
                    count += 1
        return count

    def summary(self) -> str:
        return f"{self.name}: n={self.n}, edges={len(self.edges)}, density={self.get_density():.3f}"

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
            if graph.has_edge(sequence[i-1], sequence[i]):
                current.append(sequence[i])
            else:
                if len(current) > len(longest):
                    longest = current
                current = [sequence[i]]
        if len(current) > len(longest):
            longest = current
        return longest

# ==================== 3. BipartiteGBS sampler ====================
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
        self.last_sampling_time = 0.0

    def _prepare_gbs(self):
        U, s, Vt = np.linalg.svd(self.B, full_matrices=False)
        scale = 0.75 / np.max(s)
        self.U = U
        self.V = Vt.T.conj()
        self.r = np.arctanh(scale * s)
        self.singular_values = s
        print(f"    BipartiteGBS ready: m={self.m}, largest singular value={np.max(s):.4f}")

    def sample(self, num_samples: int = 500, use_cache: bool = True,
               postsel: str = "strict") -> List[Set[int]]:
        if use_cache and self.samples_cache is not None:
            print(f"    Using {len(self.samples_cache)} cached samples")
            return self.samples_cache
        print("    Sampling from BipartiteGBS...")
        start_time = time.time()
        samples = self._real_gbs_sampling(num_samples, postsel=postsel)
        self.last_sampling_time = time.time() - start_time
        print(f"    Completed {len(samples)} subset samples in {self.last_sampling_time:.2f} s")
        if use_cache:
            self.samples_cache = samples
        return samples

    def _real_gbs_sampling(self, num_samples: int, postsel: str = "strict") -> List[Set[int]]:
        if not SF_AVAILABLE:
            raise RuntimeError("Quantum sampling requested but Strawberry Fields is unavailable; provide a saved bank or install the backend.")
        self._prepare_gbs()
        samples = []
        successful_samples = 0
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
        results = sf.Engine("gaussian").run(prog, shots=num_samples)
        for photon_pattern in results.samples:
            s_photons = photon_pattern[:self.m]
            t_photons = photon_pattern[self.m:]
            if any(p > 1 for p in s_photons) or any(p > 1 for p in t_photons):
                continue
            S = {i for i, p in enumerate(s_photons) if p == 1}
            T = {i for i, p in enumerate(t_photons) if p == 1}
            if postsel == "relaxed":
                keep = (len(S) >= 1 and len(T) >= 1)
            else:
                keep = (len(S) == len(T) and len(S) >= 3)
            if keep:
                vertices = S.union(T)
                samples.append(vertices)
                successful_samples += 1
        print(f"    Accepted samples: {successful_samples}/{num_samples}")
        return samples

    def get_gbs_info(self, num_samples: int = 500, postsel: str = "strict") -> GBSInfo:
        samples = self.sample(num_samples, postsel=postsel)
        vertex_freq = Counter()
        edge_freq = Counter()
        path_freq = Counter()
        subgraph_sizes = []
        size_dist = Counter()
        for vertices in samples:
            subgraph_sizes.append(len(vertices))
            size_dist[len(vertices)] += 1
            for v in vertices:
                vertex_freq[v] += 1
            vertices_list = sorted(vertices)
            for i in range(len(vertices_list)):
                for j in range(i+1, len(vertices_list)):
                    u, v = vertices_list[i], vertices_list[j]
                    if self.graph.has_edge(u, v):
                        edge_freq[(u, v)] += 1
                    if self.graph.has_edge(v, u):
                        edge_freq[(v, u)] += 1
        total = len(samples) if len(samples) > 0 else 1
        vertex_prob = {v: freq / total for v, freq in vertex_freq.items()}
        edge_prob = {e: freq / total for e, freq in edge_freq.items()}
        top_edges = [e for e, _ in edge_freq.most_common(min(self.n*2, len(edge_freq)))]
        avg_size = np.mean(subgraph_sizes) if subgraph_sizes else 0
        max_size = max(subgraph_sizes) if subgraph_sizes else 0
        size_distribution = {k: v / total for k, v in size_dist.items()}
        return GBSInfo(
            vertex_freq=vertex_freq, edge_freq=edge_freq, path_freq=path_freq,
            vertex_prob=vertex_prob, edge_prob=edge_prob,
            top_vertices=[v for v, _ in vertex_freq.most_common(min(self.n, self.n))],
            top_edges=top_edges, top_paths=[],
            samples=samples, sample_count=num_samples, effective_samples=len(samples),
            max_subgraph_size=max_size, avg_subgraph_size=avg_size,
            subgraph_size_distribution=size_distribution
        )

    def get_sampling_result(self, num_samples: int = 500, gbs_info: GBSInfo = None) -> GBSSamplingResult:
        if gbs_info is None:
            gbs_info = self.get_gbs_info(num_samples)
        return GBSSamplingResult(
            graph_name=self.graph.name, graph_size=self.n,
            total_samples=num_samples, effective_samples=gbs_info.effective_samples,
            avg_subgraph_size=gbs_info.avg_subgraph_size, max_subgraph_size=gbs_info.max_subgraph_size,
            vertex_probabilities=gbs_info.vertex_prob, edge_probabilities=gbs_info.edge_prob,
            top_vertices=gbs_info.top_vertices[:10], top_edges=gbs_info.top_edges[:10],
            all_samples=gbs_info.samples, sample_time=self.last_sampling_time
        )

# ==================== 4. Algorithm base class ====================
class HamiltonianAlgorithm(ABC):
    def __init__(self, name: str):
        self.name = name
        self.validator = HamiltonianValidator()

    @abstractmethod
    def search(self, graph: DirectedGraph, guidance_info=None) -> ExperimentResult:
        pass

# ==================== 5. Classical algorithms ====================
class GeneticAlgorithm(HamiltonianAlgorithm):
    def __init__(self, pop_size: int, max_generations: int,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        super().__init__("GeneticAlgorithm")
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def search(self, graph: DirectedGraph, guidance_info=None) -> ExperimentResult:
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
                    algorithm_name=self.name, success=True, cycle=best_cycle, path_length=graph.n,
                    longest_path=best_cycle, time_seconds=time.time()-start_time,
                    generations=generation, gbs_used=False, graph_type=graph.name, graph_size=graph.n
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
            algorithm_name=self.name, success=False, cycle=[], path_length=len(longest_path),
            longest_path=longest_path, time_seconds=time.time()-start_time,
            generations=self.max_generations, gbs_used=False, graph_type=graph.name, graph_size=graph.n
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
            if graph.has_edge(individual[i], individual[i+1]):
                length += 1
        if graph.has_edge(individual[-1], individual[0]):
            length += 1
        return length / graph.n

    def _tournament_select(self, population: List[List[int]], fitness: List[float], k: int = 3) -> List[int]:
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(indices, key=lambda i: fitness[i])
        return population[best_idx][:]

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        n = len(parent1)
        child = [-1] * n
        start = random.randint(0, n-2)
        end = random.randint(start+1, n-1)
        child[start:end+1] = parent1[start:end+1]
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


# ==================== 6. BipartiteGBS ablation variants ====================
# All missing co-occurrences use DEFAULT_EDGE_PROB.

class GBS_InitOnly(GeneticAlgorithm):
    def __init__(self, beta: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.name = "GBS-InitOnly"
        self.beta = beta

    def search(self, graph: DirectedGraph, gbs_info: GBSInfo = None) -> ExperimentResult:
        if gbs_info is None:
            return super().search(graph, None)
        start_time = time.time()
        population = self._gbs_guided_init_population(graph, gbs_info)
        best_individual = None; best_fitness = 0; best_cycle = None
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
                    algorithm_name=self.name, success=True, cycle=best_cycle, path_length=graph.n,
                    longest_path=best_cycle, time_seconds=time.time()-start_time,
                    generations=generation, gbs_used=True, graph_type=graph.name, graph_size=graph.n
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
            algorithm_name=self.name, success=False, cycle=[], path_length=len(longest_path),
            longest_path=longest_path, time_seconds=time.time()-start_time,
            generations=self.max_generations, gbs_used=True, graph_type=graph.name, graph_size=graph.n
        )

    def _gbs_guided_init_population(self, graph: DirectedGraph, gbs_info: GBSInfo) -> List[List[int]]:
        population = []
        n = graph.n
        for _ in range(self.pop_size // 3):
            ind = list(range(n)); random.shuffle(ind); population.append(ind)
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
            ind = list(range(n)); random.shuffle(ind); population.append(ind)
        return population[:self.pop_size]

    def _construct_from_subgraph(self, graph: DirectedGraph, vertices: Set[int], gbs_info: GBSInfo) -> List[int]:
        n = graph.n
        vertices_list = sorted(vertices)
        if len(vertices_list) < 3:
            return None
        used = set()
        path = []
        start = vertices_list[0]
        path.append(start); used.add(start); current = start
        remaining = set(vertices_list) - used
        while remaining:
            candidates = []
            for v in remaining:
                if graph.has_edge(current, v):
                    prob = gbs_info.edge_prob.get((current, v), DEFAULT_EDGE_PROB)
                    candidates.append((v, prob))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                next_v = candidates[0][0]
                path.append(next_v); used.add(next_v); current = next_v; remaining.remove(next_v)
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
        used = set(); path = []
        if gbs_info.samples and random.random() < 0.5:
            sample = random.choice(gbs_info.samples[:min(50, len(gbs_info.samples))])
            start = random.choice(list(sample))
            path.append(start); used.add(start)
        else:
            start = random.randint(0, n-1)
            path.append(start); used.add(start)
        while len(path) < n:
            last = path[-1]
            if random.random() < self.beta and gbs_info.top_edges:
                found = False
                for u, v in gbs_info.top_edges[:n]:
                    if u == last and v not in used:
                        path.append(v); used.add(v); found = True
                        break
                if found:
                    continue
            neighbors = [v for v in graph.get_out_neighbors(last) if v not in used]
            if neighbors:
                path.append(random.choice(neighbors)); used.add(path[-1])
            else:
                remaining = [v for v in range(n) if v not in used]
                if remaining:
                    path.append(random.choice(remaining)); used.add(path[-1])
                else:
                    break
        remaining = [v for v in range(n) if v not in path]
        path.extend(remaining)
        return path



class DensityBiasedGA(GeneticAlgorithm):
    def __init__(self, pop_size: int, max_generations: int,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 beta: float = 0.2):
        super().__init__(pop_size, max_generations, crossover_rate, mutation_rate)
        self.name = "Density-Biased-GA"
        self.beta = beta

    def search(self, graph: DirectedGraph, guidance_info: GBSInfo = None) -> ExperimentResult:
        start_time = time.time()
        n = graph.n
        size_dist = {}
        if guidance_info and guidance_info.samples:
            size_dist = Counter(len(s) for s in guidance_info.samples)
            total = len(guidance_info.samples)
            size_dist_norm = {k: v/total for k, v in size_dist.items()}
        else:
            size_dist_norm = {s: 1.0/(min(n,20)-2) for s in range(3, min(n,20)+1)}
        num_density_subsets = self.pop_size // 3 * 2
        density_subsets = self._sample_density_biased(graph, num_density_subsets, size_dist_norm)
        density_edge_prob = self._compute_edge_probability(graph, density_subsets)
        density_info = GBSInfo(
            vertex_freq=Counter(), edge_freq=Counter(), path_freq=Counter(),
            vertex_prob={}, edge_prob=density_edge_prob,
            top_vertices=[], top_edges=[e for e, _ in sorted(density_edge_prob.items(), key=lambda x: x[1], reverse=True)[:n]],
            top_paths=[], samples=density_subsets,
            sample_count=len(density_subsets), effective_samples=len(density_subsets),
            max_subgraph_size=max([len(s) for s in density_subsets]) if density_subsets else 0,
            avg_subgraph_size=np.mean([len(s) for s in density_subsets]) if density_subsets else 0,
            subgraph_size_distribution=size_dist_norm
        )
        population = self._gbs_guided_init_population(graph, density_info)
        while len(population) < self.pop_size:
            ind = list(range(n)); random.shuffle(ind); population.append(ind)
        population = population[:self.pop_size]
        best_individual = None; best_fitness = 0; best_cycle = None
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
                    algorithm_name=self.name, success=True, cycle=best_cycle, path_length=graph.n,
                    longest_path=best_cycle, time_seconds=time.time()-start_time,
                    generations=generation, gbs_used=False, graph_type=graph.name, graph_size=graph.n
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
            algorithm_name=self.name, success=False, cycle=[], path_length=len(longest_path),
            longest_path=longest_path, time_seconds=time.time()-start_time,
            generations=self.max_generations, gbs_used=False, graph_type=graph.name, graph_size=graph.n
        )

    def _sample_density_biased(self, graph: DirectedGraph, num_subsets: int,
                               size_dist: Dict[int, float]) -> List[Set[int]]:
        n = graph.n
        samples = []
        outdegrees = [len(graph.adj_list[i]) for i in range(n)]
        indegrees = [0]*n
        for u in range(n):
            for v in graph.adj_list[u]:
                indegrees[v] += 1
        total_degrees = [outdegrees[i] + indegrees[i] for i in range(n)]
        sorted_by_degree = sorted(range(n), key=lambda x: total_degrees[x], reverse=True)
        for _ in range(num_subsets):
            if size_dist:
                sizes = list(size_dist.keys())
                probs = [size_dist[s] for s in sizes]
                if len(sizes)>1:
                    size = random.choices(sizes, weights=probs, k=1)[0]
                else:
                    size = sizes[0] if sizes else random.randint(3, min(n,10))
            else:
                size = random.randint(3, min(n,10))
            size = min(size, n)
            if size < 3: size = min(max(3,n), n)
            seed_size = min(max(1, size//3), len(sorted_by_degree))
            seeds = set(sorted_by_degree[:seed_size])
            if len(seeds) < seed_size:
                remaining_seeds = set(range(n)) - seeds
                if remaining_seeds:
                    extra_seeds = random.sample(list(remaining_seeds), min(seed_size - len(seeds), len(remaining_seeds)))
                    seeds.update(extra_seeds)
            vertices = set(seeds)
            candidates = set(range(n)) - vertices
            while len(vertices) < size and candidates:
                best_vertex = None; best_gain = -1
                candidate_list = list(candidates)
                if len(candidate_list) > n*2:
                    connected_candidates = []
                    for v in candidate_list:
                        for u in vertices:
                            if graph.has_edge(u, v) or graph.has_edge(v, u):
                                connected_candidates.append(v); break
                    if connected_candidates:
                        candidate_list = connected_candidates
                    else:
                        candidate_list = random.sample(candidate_list, min(n*2, len(candidate_list)))
                for v in candidate_list:
                    gain = 0
                    for u in vertices:
                        if graph.has_edge(u, v): gain += 1
                        if graph.has_edge(v, u): gain += 1
                    if gain > best_gain:
                        best_gain = gain; best_vertex = v
                if best_vertex is not None and best_gain >= 0:
                    vertices.add(best_vertex); candidates.remove(best_vertex)
                else:
                    if candidates:
                        v = random.choice(list(candidates)); vertices.add(v); candidates.remove(v)
                    else:
                        break
            while len(vertices) < size:
                remaining = set(range(n)) - vertices
                if not remaining: break
                vertices.add(random.choice(list(remaining)))
            samples.append(vertices)
        return samples

    def _compute_edge_probability(self, graph: DirectedGraph, subsets: List[Set[int]]) -> Dict[Tuple[int,int], float]:
        edge_freq = Counter()
        total = len(subsets)
        for vertices in subsets:
            vertices_list = sorted(vertices)
            for i in range(len(vertices_list)):
                for j in range(i+1, len(vertices_list)):
                    u, v = vertices_list[i], vertices_list[j]
                    if graph.has_edge(u, v): edge_freq[(u,v)] += 1
                    if graph.has_edge(v, u): edge_freq[(v,u)] += 1
        edge_prob = {e: freq/total for e, freq in edge_freq.items()}
        return edge_prob

    def _gbs_guided_init_population(self, graph: DirectedGraph, info: GBSInfo) -> List[List[int]]:
        population = []
        n = graph.n
        for _ in range(self.pop_size // 3):
            ind = list(range(n)); random.shuffle(ind); population.append(ind)
        if info.samples:
            num_subset_individuals = min(self.pop_size // 3, len(info.samples))
            for sample in info.samples[:num_subset_individuals]:
                ind = self._construct_from_subgraph(graph, sample, info)
                if ind:
                    population.append(ind)
        for _ in range(self.pop_size // 3):
            ind = self._construct_hybrid(graph, info)
            if ind:
                population.append(ind)
        while len(population) < self.pop_size:
            ind = list(range(n)); random.shuffle(ind); population.append(ind)
        return population[:self.pop_size]

    def _construct_from_subgraph(self, graph: DirectedGraph, vertices: Set[int], info: GBSInfo) -> List[int]:
        n = graph.n
        vertices_list = sorted(vertices)
        if len(vertices_list) < 3:
            return None
        used = set()
        path = []
        start = vertices_list[0]
        path.append(start); used.add(start); current = start
        remaining = set(vertices_list) - used
        while remaining:
            candidates = []
            for v in remaining:
                if graph.has_edge(current, v):
                    prob = info.edge_prob.get((current, v), DEFAULT_EDGE_PROB)
                    candidates.append((v, prob))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                next_v = candidates[0][0]
                path.append(next_v); used.add(next_v); current = next_v; remaining.remove(next_v)
            else:
                break
        all_vertices = set(range(n))
        remaining_all = all_vertices - used
        remaining_list = list(remaining_all)
        random.shuffle(remaining_list)
        path.extend(remaining_list)
        return path

    def _construct_hybrid(self, graph: DirectedGraph, info: GBSInfo) -> List[int]:
        n = graph.n
        used = set(); path = []
        if info.samples and random.random() < 0.5:
            sample = random.choice(info.samples[:min(50, len(info.samples))])
            start = random.choice(list(sample))
            path.append(start); used.add(start)
        else:
            start = random.randint(0, n-1)
            path.append(start); used.add(start)
        while len(path) < n:
            last = path[-1]
            if random.random() < self.beta and info.top_edges:
                found = False
                for u, v in info.top_edges[:n]:
                    if u == last and v not in used:
                        path.append(v); used.add(v); found = True
                        break
                if found:
                    continue
            neighbors = [v for v in graph.get_out_neighbors(last) if v not in used]
            if neighbors:
                path.append(random.choice(neighbors)); used.add(path[-1])
            else:
                remaining = [v for v in range(n) if v not in used]
                if remaining:
                    path.append(random.choice(remaining)); used.add(path[-1])
                else:
                    break
        remaining = [v for v in range(n) if v not in path]
        path.extend(remaining)
        return path

class GBS_FitnessOnly(GeneticAlgorithm):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.name = "GBS-FitnessOnly"
        self.alpha = alpha

    def search(self, graph: DirectedGraph, gbs_info: GBSInfo = None) -> ExperimentResult:
        if gbs_info is None:
            return super().search(graph, None)
        start_time = time.time()
        population = self._init_population(graph)
        best_individual = None; best_fitness = 0; best_cycle = None
        for generation in range(self.max_generations):
            current_alpha = self.alpha * (generation / self.max_generations)
            fitness = [self._gbs_enhanced_fitness(graph, ind, gbs_info, current_alpha) for ind in population]
            max_idx = np.argmax(fitness)
            if fitness[max_idx] > best_fitness:
                best_fitness = fitness[max_idx]
                best_individual = population[max_idx][:]
                if self.validator.verify_cycle(graph, best_individual):
                    best_cycle = best_individual[:]
            if best_cycle is not None:
                return ExperimentResult(
                    algorithm_name=self.name, success=True, cycle=best_cycle, path_length=graph.n,
                    longest_path=best_cycle, time_seconds=time.time()-start_time,
                    generations=generation, gbs_used=True, graph_type=graph.name, graph_size=graph.n
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
            algorithm_name=self.name, success=False, cycle=[], path_length=len(longest_path),
            longest_path=longest_path, time_seconds=time.time()-start_time,
            generations=self.max_generations, gbs_used=True, graph_type=graph.name, graph_size=graph.n
        )

    def _gbs_enhanced_fitness(self, graph: DirectedGraph, individual: List[int],
                              gbs_info: GBSInfo, alpha: float) -> float:
        length = 0
        for i in range(graph.n - 1):
            if graph.has_edge(individual[i], individual[i+1]):
                length += 1
        can_close = graph.has_edge(individual[-1], individual[0])
        if can_close:
            length += 1
            if length == graph.n:
                return 1.0
        base_fitness = length / graph.n
        gbs_score = 0
        edge_count = 0
        for i in range(graph.n - 1):
            edge = (individual[i], individual[i+1])
            score = gbs_info.edge_prob.get(edge, DEFAULT_EDGE_PROB)
            gbs_score += score
            edge_count += 1
        if can_close:
            edge = (individual[-1], individual[0])
            score = gbs_info.edge_prob.get(edge, DEFAULT_EDGE_PROB)
            gbs_score += score
            edge_count += 1
        gbs_score = gbs_score / edge_count if edge_count > 0 else 0
        combined = base_fitness * (1 - alpha) + gbs_score * alpha
        return combined

class GBS_MutationOnly(GeneticAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "GBS-MutationOnly"

    def search(self, graph: DirectedGraph, gbs_info: GBSInfo = None) -> ExperimentResult:
        if gbs_info is None:
            return super().search(graph, None)
        start_time = time.time()
        population = self._init_population(graph)
        best_individual = None; best_fitness = 0; best_cycle = None
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
                    algorithm_name=self.name, success=True, cycle=best_cycle, path_length=graph.n,
                    longest_path=best_cycle, time_seconds=time.time()-start_time,
                    generations=generation, gbs_used=True, graph_type=graph.name, graph_size=graph.n
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
            algorithm_name=self.name, success=False, cycle=[], path_length=len(longest_path),
            longest_path=longest_path, time_seconds=time.time()-start_time,
            generations=self.max_generations, gbs_used=True, graph_type=graph.name, graph_size=graph.n
        )

    def _gbs_guided_mutate(self, individual: List[int], gbs_info: GBSInfo) -> List[int]:
        n = len(individual)
        if gbs_info and gbs_info.edge_prob and random.random() < 0.7:
            edge_scores = []
            for i in range(n):
                u = individual[i]
                v = individual[(i+1) % n]
                score = gbs_info.edge_prob.get((u, v), DEFAULT_EDGE_PROB)
                edge_scores.append((i, score))
            edge_scores.sort(key=lambda x: x[1])
            if edge_scores:
                idx = edge_scores[0][0]
                jdx = random.randint(0, n-1)
                individual[idx], individual[jdx] = individual[jdx], individual[idx]
                return individual
        i, j = random.sample(range(n), 2)
        individual[i], individual[j] = individual[j], individual[i]
        return individual

class GBS_NoSubgraph(GeneticAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "GBS-NoSubgraph"

    def search(self, graph: DirectedGraph, gbs_info: GBSInfo = None) -> ExperimentResult:
        if gbs_info is None:
            return super().search(graph, None)
        start_time = time.time()
        population = self._gbs_guided_init_population(graph, gbs_info)
        best_individual = None; best_fitness = 0; best_cycle = None
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
                    algorithm_name=self.name, success=True, cycle=best_cycle, path_length=graph.n,
                    longest_path=best_cycle, time_seconds=time.time()-start_time,
                    generations=generation, gbs_used=True, graph_type=graph.name, graph_size=graph.n
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
            algorithm_name=self.name, success=False, cycle=[], path_length=len(longest_path),
            longest_path=longest_path, time_seconds=time.time()-start_time,
            generations=self.max_generations, gbs_used=True, graph_type=graph.name, graph_size=graph.n
        )

    def _gbs_guided_init_population(self, graph: DirectedGraph, gbs_info: GBSInfo) -> List[List[int]]:
        population = []
        n = graph.n
        for _ in range(self.pop_size // 2):
            ind = list(range(n)); random.shuffle(ind); population.append(ind)
        for _ in range(self.pop_size // 2):
            ind = self._construct_by_edge_prob(graph, gbs_info)
            if ind:
                population.append(ind)
        while len(population) < self.pop_size:
            ind = list(range(n)); random.shuffle(ind); population.append(ind)
        return population[:self.pop_size]

    def _construct_by_edge_prob(self, graph: DirectedGraph, gbs_info: GBSInfo) -> List[int]:
        n = graph.n
        used = set(); path = []
        start = random.randint(0, n-1)
        path.append(start); used.add(start)
        while len(path) < n:
            current = path[-1]
            candidates = []
            for v in range(n):
                if v not in used and graph.has_edge(current, v):
                    prob = gbs_info.edge_prob.get((current, v), DEFAULT_EDGE_PROB)
                    candidates.append((v, prob))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                next_v = candidates[0][0]
                path.append(next_v); used.add(next_v)
            else:
                remaining = [v for v in range(n) if v not in used]
                if remaining:
                    path.append(random.choice(remaining)); used.add(path[-1])
                else:
                    break
        return path

class GBS_SubgraphOnly(GeneticAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "GBS-SubgraphOnly"

    def search(self, graph: DirectedGraph, gbs_info: GBSInfo = None) -> ExperimentResult:
        if gbs_info is None:
            return super().search(graph, None)
        start_time = time.time()
        population = self._subgraph_only_init_population(graph, gbs_info)
        best_individual = None; best_fitness = 0; best_cycle = None
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
                    algorithm_name=self.name, success=True, cycle=best_cycle, path_length=graph.n,
                    longest_path=best_cycle, time_seconds=time.time()-start_time,
                    generations=generation, gbs_used=True, graph_type=graph.name, graph_size=graph.n
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
            algorithm_name=self.name, success=False, cycle=[], path_length=len(longest_path),
            longest_path=longest_path, time_seconds=time.time()-start_time,
            generations=self.max_generations, gbs_used=True, graph_type=graph.name, graph_size=graph.n
        )

    def _subgraph_only_init_population(self, graph: DirectedGraph, gbs_info: GBSInfo) -> List[List[int]]:
        population = []
        n = graph.n
        for _ in range(self.pop_size // 2):
            ind = list(range(n)); random.shuffle(ind); population.append(ind)
        if gbs_info.samples:
            for _ in range(self.pop_size // 2):
                ind = self._construct_from_subgraph_random(graph, gbs_info)
                if ind:
                    population.append(ind)
        while len(population) < self.pop_size:
            ind = list(range(n)); random.shuffle(ind); population.append(ind)
        return population[:self.pop_size]

    def _construct_from_subgraph_random(self, graph: DirectedGraph, gbs_info: GBSInfo) -> List[int]:
        n = graph.n
        if not gbs_info.samples:
            return None
        sample = random.choice(gbs_info.samples[:min(100, len(gbs_info.samples))])
        vertices_list = list(sample)
        if len(vertices_list) < 3:
            return None
        random.shuffle(vertices_list)
        used = set(); path = []
        start = vertices_list[0]
        path.append(start); used.add(start); current = start
        remaining = set(vertices_list) - used
        while remaining:
            candidates = [v for v in remaining if graph.has_edge(current, v)]
            if candidates:
                next_v = random.choice(candidates)
                path.append(next_v); used.add(next_v); current = next_v; remaining.remove(next_v)
            else:
                break
        all_vertices = set(range(n))
        remaining_all = list(all_vertices - used)
        random.shuffle(remaining_all)
        path.extend(remaining_all)
        return path

class GBSEnhancedGenetic(GeneticAlgorithm):
    def __init__(self, alpha: float, beta: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.name = "GBS-Genetic"
        self.alpha = alpha
        self.beta = beta

    def search(self, graph: DirectedGraph, gbs_info: GBSInfo = None) -> ExperimentResult:
        if gbs_info is None:
            return super().search(graph, None)
        start_time = time.time()
        population = self._gbs_guided_init_population(graph, gbs_info)
        best_individual = None; best_fitness = 0; best_cycle = None
        for generation in range(self.max_generations):
            current_alpha = self.alpha * (generation / self.max_generations)
            fitness = [self._gbs_enhanced_fitness(graph, ind, gbs_info, current_alpha) for ind in population]
            max_idx = np.argmax(fitness)
            if fitness[max_idx] > best_fitness:
                best_fitness = fitness[max_idx]
                best_individual = population[max_idx][:]
                if self.validator.verify_cycle(graph, best_individual):
                    best_cycle = best_individual[:]
            if best_cycle is not None:
                return ExperimentResult(
                    algorithm_name=self.name, success=True, cycle=best_cycle, path_length=graph.n,
                    longest_path=best_cycle, time_seconds=time.time()-start_time,
                    generations=generation, gbs_used=True, graph_type=graph.name, graph_size=graph.n
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
            algorithm_name=self.name, success=False, cycle=[], path_length=len(longest_path),
            longest_path=longest_path, time_seconds=time.time()-start_time,
            generations=self.max_generations, gbs_used=True, graph_type=graph.name, graph_size=graph.n
        )

    def _gbs_guided_init_population(self, graph: DirectedGraph, gbs_info: GBSInfo) -> List[List[int]]:
        population = []
        n = graph.n
        for _ in range(self.pop_size // 3):
            ind = list(range(n)); random.shuffle(ind); population.append(ind)
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
            ind = list(range(n)); random.shuffle(ind); population.append(ind)
        return population[:self.pop_size]

    def _construct_from_subgraph(self, graph: DirectedGraph, vertices: Set[int], gbs_info: GBSInfo) -> List[int]:
        n = graph.n
        vertices_list = sorted(vertices)
        if len(vertices_list) < 3:
            return None
        used = set(); path = []
        start = vertices_list[0]
        path.append(start); used.add(start); current = start
        remaining = set(vertices_list) - used
        while remaining:
            candidates = []
            for v in remaining:
                if graph.has_edge(current, v):
                    prob = gbs_info.edge_prob.get((current, v), DEFAULT_EDGE_PROB)
                    candidates.append((v, prob))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                next_v = candidates[0][0]
                path.append(next_v); used.add(next_v); current = next_v; remaining.remove(next_v)
            else:
                break
        all_vertices = set(range(n))
        remaining_all = all_vertices - used
        remaining_list = list(remaining_all)
        random.shuffle(remaining_list)
        path.extend(remaining_list)
        return path

    # def _construct_hybrid(self, graph: DirectedGraph, gbs_info: GBSInfo) -> List[int]:
    #     n = graph.n
    #     used = set(); path = []
    #     if gbs_info.samples and random.random() < 0.5:
    #         sample = random.choice(gbs_info.samples[:min(50, len(gbs_info.samples))])
    #         start = random.choice(list(sample))
    #         path.append(start); used.add(start)
    #     else:
    #         start = random.randint(0, n-1)
    #         path.append(start); used.add(start)
    #     while len(path) < n:
    #         last = path[-1]
    #         neighbors = [v for v in graph.get_out_neighbors(last) if v not in used]
    #         if neighbors:
    #             path.append(random.choice(neighbors)); used.add(path[-1])
    #         else:
    #             remaining = [v for v in range(n) if v not in used]
    #             if remaining:
    #                 path.append(random.choice(remaining)); used.add(path[-1])
    #             else:
    #                 break
    #     remaining = [v for v in range(n) if v not in path]
    #     path.extend(remaining)
    #     return path

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
            if graph.has_edge(individual[i], individual[i+1]):
                length += 1
        can_close = graph.has_edge(individual[-1], individual[0])
        if can_close:
            length += 1
            if length == graph.n:
                return 1.0
        base_fitness = length / graph.n
        gbs_score = 0
        edge_count = 0
        for i in range(graph.n - 1):
            edge = (individual[i], individual[i+1])
            score = gbs_info.edge_prob.get(edge, DEFAULT_EDGE_PROB)
            gbs_score += score
            edge_count += 1
        if can_close:
            edge = (individual[-1], individual[0])
            score = gbs_info.edge_prob.get(edge, DEFAULT_EDGE_PROB)
            gbs_score += score
            edge_count += 1
        gbs_score = gbs_score / edge_count if edge_count > 0 else 0
        combined = base_fitness * (1 - alpha) + gbs_score * alpha
        return combined

    def _gbs_guided_mutate(self, individual: List[int], gbs_info: GBSInfo) -> List[int]:
        n = len(individual)
        if gbs_info and gbs_info.edge_prob and random.random() < 0.7:
            edge_scores = []
            for i in range(n):
                u = individual[i]
                v = individual[(i+1) % n]
                score = gbs_info.edge_prob.get((u, v), DEFAULT_EDGE_PROB)
                edge_scores.append((i, score))
            edge_scores.sort(key=lambda x: x[1])
            if edge_scores:
                idx = edge_scores[0][0]
                jdx = random.randint(0, n-1)
                individual[idx], individual[jdx] = individual[jdx], individual[idx]
                return individual
        i, j = random.sample(range(n), 2)
        individual[i], individual[j] = individual[j], individual[i]
        return individual

# ==================== 9. Multistage BipartiteGBS-guided algorithm ====================

class MultiStageGBSEnhancedGenetic(HamiltonianAlgorithm):

    def __init__(self, pop_size: int, max_generations: int):
        super().__init__("MultiStageGBS")
        self.pop_size = pop_size
        self.max_generations = max_generations

    def search(self, graph: DirectedGraph, gbs_info: GBSInfo = None) -> ExperimentResult:
        start_time = time.time()

        stage1 = GeneticAlgorithm(
            pop_size=self.pop_size // 2,
            max_generations=self.max_generations // 3
        )
        result1 = stage1.search(graph, None)

        if result1.success:
            result1.time_seconds = time.time() - start_time
            result1.algorithm_name = self.name
            result1.gbs_used = True
            return result1

        good_edges = self._extract_good_edges(result1.longest_path if result1.longest_path else [])

        stage2 = GBSEnhancedGenetic(
            # pop_size=self.pop_size // 2,
            # max_generations=self.max_generations * 2 // 3,
            pop_size=self.pop_size,
            max_generations=self.max_generations // 2,
            alpha=0.1,
            beta=0.2
        )

        merged_info = self._merge_gbs_info(gbs_info, good_edges) if gbs_info else None

        result2 = stage2.search(graph, merged_info)
        result2.time_seconds = time.time() - start_time
        result2.algorithm_name = self.name
        result2.gbs_used = True

        return result2

    def _extract_good_edges(self, path: List[int]) -> Set[Tuple]:
        if not path or len(path) < 2:
            return set()
        edges = set()
        for i in range(len(path) - 1):
            edges.add((path[i], path[i + 1]))
        return edges

    def _merge_gbs_info(self, gbs_info: GBSInfo, good_edges: Set[Tuple]) -> GBSInfo:
        import copy
        merged = copy.deepcopy(gbs_info)
        for edge in good_edges:
            merged.edge_prob[edge] = max(merged.edge_prob.get(edge, 0), 0.8)
            if edge not in merged.top_edges:
                merged.top_edges.insert(0, edge)
        return merged

# ==================== Experiment runner ====================
def _make_algorithms():
    return [
        GeneticAlgorithm(pop_size=100, max_generations=200),
        DensityBiasedGA(pop_size=100, max_generations=200),
        GBS_InitOnly(pop_size=100, max_generations=200),
        GBS_FitnessOnly(pop_size=100, max_generations=200, alpha=0.1),
        GBS_MutationOnly(pop_size=100, max_generations=200),
        GBS_NoSubgraph(pop_size=100, max_generations=200),
        GBS_SubgraphOnly(pop_size=100, max_generations=200),
        GBSEnhancedGenetic(pop_size=100, max_generations=200, alpha=0.1),
        MultiStageGBSEnhancedGenetic(pop_size=100, max_generations=300)
    ]

def _run_one_graph(seed: int, n: int, i: int, n_trials: int):
    graph = GraphGenerator.generate_random_digraph(n, 0.3, seed)
    graph.name = f"Random_n{n}_{i+1}"
    print(f"    [n={n} graph {i+1}] Generating BipartiteGBS info...", flush=True)
    gbs_sampler = BipartiteGBSSampler(graph)
    gbs_info = gbs_sampler.get_gbs_info(num_samples=500)
    gbs_sampling_result = gbs_sampler.get_sampling_result(num_samples=500, gbs_info=gbs_info)
    algorithms = _make_algorithms()
    results = []
    for algo in algorithms:
        for trial in range(n_trials):
            if "GBS" in algo.name or "Density" in algo.name:
                result = algo.search(graph, gbs_info)
            else:
                result = algo.search(graph, None)
            result.graph_type = graph.name
            result.graph_size = graph.n
            result.algorithm_name = algo.name
            results.append(result)
    return graph.name, results, gbs_sampling_result

class ExperimentRunner:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed); np.random.seed(seed)
        self.results = []
        self.graph_results = defaultdict(list)
        self.setup_complete = False
        self.excel_manager = ExcelDataManager()
        self.gbs_results = {}

    def setup(self):
        if not self.setup_complete:
            print("="*60)
            print("HAMILTONIAN CYCLE SEARCH EXPERIMENT")
            print("GBS vs Classical Baselines (Density-Biased)")
            print("="*60)
            self.setup_complete = True

    def run_comparison_on_single_graph(self, graph: DirectedGraph, n_trials: int):
        print(f"\n  Running on {graph.summary()}")
        print("    Generating BipartiteGBS info...")
        gbs_sampler = BipartiteGBSSampler(graph)
        gbs_info = gbs_sampler.get_gbs_info(num_samples=500)
        gbs_sampling_result = gbs_sampler.get_sampling_result(num_samples=500)
        self.gbs_results[graph.name] = gbs_sampling_result
        self.excel_manager.add_gbs_sampling_result(gbs_sampling_result)
        algorithms = _make_algorithms()
        for algo in algorithms:
            print(f"    Running {algo.name}...")
            algo_successes = 0
            for trial in range(n_trials):
                if "GBS" in algo.name or "Density" in algo.name:
                    result = algo.search(graph, gbs_info)
                else:
                    result = algo.search(graph, None)
                result.graph_type = graph.name
                result.graph_size = graph.n
                result.algorithm_name = algo.name
                self.results.append(result)
                if result.success:
                    algo_successes += 1
            print(f"      {algo.name} success rate: {(algo_successes/n_trials)*100:.1f}% ({algo_successes}/{n_trials})")

    def run_batch(self, sizes: List[int], graphs_per_type: int, n_trials: int,
                  parallel: Optional[bool] = None, workers: int = 48):
        self.setup()
        tasks = []
        for n in sizes:
            print(f"\n{'='*60}\nTesting graphs of size n={n}\n{'='*60}")
            for i in range(graphs_per_type):
                seed = self.seed + i*100 + n*10
                tasks.append((seed, n, i, n_trials))
        if parallel is None:
            parallel = '--seq' not in sys.argv
        if parallel and len(tasks)>1:
            from multiprocessing import Pool
            print(f"\nRunning {len(tasks)} graphs in parallel ({workers} workers)...", flush=True)
            with Pool(processes=workers) as pool:
                outputs = pool.starmap(_run_one_graph, tasks)
        else:
            outputs = [_run_one_graph(*task) for task in tasks]
        for graph_name, results, gbs_sampling_result in outputs:
            self.gbs_results[graph_name] = gbs_sampling_result
            self.excel_manager.add_gbs_sampling_result(gbs_sampling_result)
            self.results.extend(results)
        return self.results

    def save_to_excel(self):
        return self.excel_manager.save_all(self)

# ==================== Excel data management ====================
class ExcelDataManager:
    def __init__(self, result_filename: str = "experiment_results.xlsx",
                 sampling_filename: str = "gbs_sampling_data.xlsx"):
        self.result_filename = result_filename
        self.sampling_filename = sampling_filename
        self.gbs_sampling_data = []

    def add_gbs_sampling_result(self, result: GBSSamplingResult):
        self.gbs_sampling_data.append(result)

    def save_results(self, runner):
        if not runner.results:
            print("No results to save")
            return False
        print(f"\n📊 Saving experiment results to {self.result_filename}...")
        data = []
        for result in runner.results:
            data.append({
                'Graph Name': result.graph_type,
                'Graph Type': result.graph_type.split('_')[0] if '_' in result.graph_type else result.graph_type,
                'Graph Size': result.graph_size,
                'Algorithm': result.algorithm_name,
                'Success': 1 if result.success else 0,
                'Path Length': result.path_length,
                'Longest Path Length': len(result.longest_path) if result.longest_path else 0,
                'Time (seconds)': result.time_seconds,
                'Generations': result.generations,
                'GBS Used': result.gbs_used
            })
        df_main = pd.DataFrame(data)
        with pd.ExcelWriter(self.result_filename, engine='openpyxl') as writer:
            df_main.to_excel(writer, sheet_name='Raw Data', index=False)
            summary_df = df_main.groupby(['Graph Type', 'Graph Size', 'Algorithm']).agg({
                'Success': ['count', 'sum', 'mean'],
                'Path Length': 'mean',
                'Longest Path Length': 'mean',
                'Time (seconds)': 'mean'
            }).round(4)
            summary_df.to_excel(writer, sheet_name='Summary')
        print(f"Experiment results saved to {self.result_filename}")
        print(f"   - {len(df_main)} experiment records")
        return True

    def save_sampling_data(self):
        if not self.gbs_sampling_data:
            print("No GBS sampling data to save")
            return False
        print(f"\n📊 Saving GBS sampling data to {self.sampling_filename}...")
        gbs_data = []
        for result in self.gbs_sampling_data:
            samples_list = ['{' + ','.join(map(str, sorted(s))) + '}' for s in result.all_samples]
            all_samples_str = '; '.join(samples_list)
            size_distribution = Counter([len(s) for s in result.all_samples])
            size_dist_str = ', '.join([f"size={k}:{v}" for k, v in sorted(size_distribution.items())])
            vertex_prob_str = ', '.join([f"{v}:{p:.3f}" for v, p in list(result.vertex_probabilities.items())[:10]])
            edge_prob_str = ', '.join([f"({u},{v}):{p:.3f}" for (u,v), p in list(result.edge_probabilities.items())[:10]])
            top_vertices_str = ', '.join([str(v) for v in result.top_vertices])
            top_edges_str = ', '.join([f"({u},{v})" for u,v in result.top_edges])
            gbs_data.append({
                'Graph Name': result.graph_name,
                'Graph Size': result.graph_size,
                'Total Samples': result.total_samples,
                'Effective Samples': result.effective_samples,
                'Effective Rate (%)': (result.effective_samples / result.total_samples * 100) if result.total_samples > 0 else 0,
                'Avg Subgraph Size': result.avg_subgraph_size,
                'Max Subgraph Size': result.max_subgraph_size,
                'Sample Time (s)': result.sample_time,
                'Subgraph Size Distribution': size_dist_str,
                'All Subgraphs': all_samples_str,
                'Top Vertices': top_vertices_str,
                'Top Edges': top_edges_str,
                'Vertex Probabilities (top10)': vertex_prob_str,
                'Edge Probabilities (top10)': edge_prob_str
            })
        df_gbs = pd.DataFrame(gbs_data)
        with pd.ExcelWriter(self.sampling_filename, engine='openpyxl') as writer:
            df_gbs.to_excel(writer, sheet_name='GBS Sampling Data', index=False)
            if len(df_gbs)>0:
                detail_data = []
                for result in self.gbs_sampling_data:
                    for idx, sample in enumerate(result.all_samples):
                        sample_str = '{' + ','.join(map(str, sorted(sample))) + '}'
                        detail_data.append({
                            'Graph Name': result.graph_name,
                            'Graph Size': result.graph_size,
                            'Sample Index': idx+1,
                            'Subgraph Size': len(sample),
                            'Vertices': sample_str
                        })
                df_detail = pd.DataFrame(detail_data)
                df_detail.to_excel(writer, sheet_name='Subgraph Details', index=False)
                gbs_summary = df_gbs.groupby(['Graph Size']).agg({
                    'Effective Samples': 'mean',
                    'Effective Rate (%)': 'mean',
                    'Avg Subgraph Size': 'mean',
                    'Max Subgraph Size': 'mean',
                    'Sample Time (s)': 'mean'
                }).round(2)
                gbs_summary.to_excel(writer, sheet_name='GBS Sampling Summary')
        print(f"GBS sampling data saved to {self.sampling_filename}")
        print(f"   - {len(df_gbs)} GBS sampling records")
        print(f"   - Total subgraphs saved: {sum(len(r.all_samples) for r in self.gbs_sampling_data)}")
        return True

    def save_all(self, runner):
        result_success = self.save_results(runner)
        sampling_success = self.save_sampling_data()
        return result_success and sampling_success

    def load_results(self, filename: str = None):
        if filename:
            self.result_filename = filename
        try:
            df = pd.read_excel(self.result_filename, sheet_name='Raw Data')
            print(f"Loaded {len(df)} results from {self.result_filename}")
            return df
        except FileNotFoundError:
            print(f"File {self.result_filename} not found")
            return None

    def load_sampling_data(self, filename: str = None):
        if filename:
            self.sampling_filename = filename
        try:
            df = pd.read_excel(self.sampling_filename, sheet_name='GBS Sampling Data')
            print(f"Loaded {len(df)} GBS sampling records from {self.sampling_filename}")
            try:
                df_detail = pd.read_excel(self.sampling_filename, sheet_name='Subgraph Details')
                print(f"   - Loaded {len(df_detail)} subgraph details")
                return df, df_detail
            except:
                print(f"   - No subgraph details sheet found")
                return df, None
        except FileNotFoundError:
            print(f"File {self.sampling_filename} not found")
            return None, None

# ==================== Statistical tests ====================
def paired_tests_for_size(df: pd.DataFrame, size: int, algo1: str, algo2: str):
    if not SCIPY_AVAILABLE:
        return None, None, None, None
    d1 = df[(df['Graph Size'] == size) & (df['Algorithm'] == algo1)]
    d2 = df[(df['Graph Size'] == size) & (df['Algorithm'] == algo2)]
    s1 = d1.groupby('Graph Name')['Success'].sum()
    s2 = d2.groupby('Graph Name')['Success'].sum()
    common = set(s1.index) & set(s2.index)
    if len(common) < 2:
        return None, None, None, None
    x = s1[list(common)].values
    y = s2[list(common)].values
    t_stat, p_t = stats.ttest_rel(x, y)
    diff = x - y
    if np.all(diff == 0):
        p_w = 1.0
    else:
        w_stat, p_w = stats.wilcoxon(x, y, alternative='two-sided')
    return p_t, p_w, x, y

def perform_all_pairwise_tests(df: pd.DataFrame, sizes: List[int]):
    if not SCIPY_AVAILABLE:
        print("scipy not available, skipping statistical tests.")
        return
    pairs = [
        ("GeneticAlgorithm", "GBS-Genetic"),
        ("GeneticAlgorithm", "GBS-InitOnly"),
        ("GBS-InitOnly", "GBS-Genetic"),
        ("Density-Biased-GA", "GBS-InitOnly"),
        ("Density-Biased-GA", "GBS-Genetic"),
    ]
    print("\n" + "="*70)
    print("PAIRED STATISTICAL TESTS (per-instance success counts, n_trials=10)")
    print("Wilcoxon signed-rank p-values (two-sided) for key algorithm pairs")
    print("="*70)
    for size in sizes:
        print(f"\n--- Graph Size n = {size} ---")
        for algo1, algo2 in pairs:
            p_t, p_w, x, y = paired_tests_for_size(df, size, algo1, algo2)
            if p_w is None:
                print(f"  {algo1:25s} vs {algo2:25s}: insufficient data")
            else:
                mean1 = np.mean(x)
                mean2 = np.mean(y)
                diff = mean1 - mean2
                sign = "+" if diff > 0 else "-" if diff < 0 else "="
                print(f"  {algo1:25s} vs {algo2:25s}: mean diff = {diff:+.2f} (out of 10), Wilcoxon p = {p_w:.4f}  {sign}")

# ==================== Visualization ====================
class EnhancedVisualizer:
    @staticmethod
    def plot_main_comparison(df: pd.DataFrame, save_path: str = None):
        if df is None or len(df) == 0:
            print("No data to plot")
            return
        if 'Graph Size' not in df.columns:
            print("Error: 'Graph Size' column not found in DataFrame")
            return
        main_algorithms = [
            "GeneticAlgorithm",
            "Density-Biased-GA",
            "GBS-InitOnly",
            "GBS-Genetic",
            "MultiStageGBS"
        ]
        df_main = df[df['Algorithm'].isin(main_algorithms)]
        if len(df_main) == 0:
            print("No data for main comparison")
            return
        size_algo_stats = defaultdict(lambda: defaultdict(dict))
        for (graph_size, algo), group in df_main.groupby(['Graph Size', 'Algorithm']):
            total_trials = len(group)
            successes = group['Success'].sum()
            success_rate = (successes / total_trials) * 100
            p = successes / total_trials
            se = np.sqrt(p * (1-p) / total_trials) * 100
            size_algo_stats[graph_size][algo]['success_mean'] = success_rate
            size_algo_stats[graph_size][algo]['success_std'] = se
            failed_cases = group[group['Success'] == 0]
            if len(failed_cases) > 0:
                path_lengths = failed_cases.groupby('Graph Name')['Longest Path Length'].mean()
                path_mean = path_lengths.mean()
                path_se = path_lengths.std() / np.sqrt(len(path_lengths))
            else:
                path_mean = 0; path_se = 0
            size_algo_stats[graph_size][algo]['path_mean'] = path_mean
            size_algo_stats[graph_size][algo]['path_std'] = path_se
        target_sizes = [15,20,25,30,35,40]
        graph_sizes = sorted([size for size in size_algo_stats.keys() if size in target_sizes])
        if len(graph_sizes) == 0:
            graph_sizes = sorted(size_algo_stats.keys())
        n_sizes = len(graph_sizes)
        if n_sizes == 0:
            print("No data to plot")
            return
        algo_labels = {
            "GeneticAlgorithm": "GA",
            "Density-Biased-GA": "Density-Biased-GA",
            "GBS-InitOnly": "InitOnly",
            "GBS-Genetic": "BipartiteGBS-GA",
            "MultiStageGBS": "MS-BipartiteGBS-GA"
        }
        algo_colors = {
            "GeneticAlgorithm": "#666666",
            "Density-Biased-GA": "#A6D854",
            "GBS-InitOnly": "#1F78B4",
            "GBS-Genetic": "#D95F02",
            "MultiStageGBS": "#E7298A"
        }
        fig, axes = plt.subplots(1, n_sizes, figsize=(6.5*n_sizes, 8.5))
        if n_sizes == 1:
            axes = [axes]
        for idx, size in enumerate(graph_sizes):
            ax = axes[idx]
            success_means = []; success_stds = []; path_means = []; path_stds = []
            for algo in main_algorithms:
                stats = size_algo_stats[size].get(algo, {})
                success_means.append(stats.get('success_mean', 0))
                success_stds.append(stats.get('success_std', 0))
                path_means.append(stats.get('path_mean', 0))
                path_stds.append(stats.get('path_std', 0))
            x = np.arange(len(main_algorithms))
            width = 0.35
            ax2 = ax.twinx()
            bar_colors = [algo_colors.get(algo, "#AAAAAA") for algo in main_algorithms]
            bars1 = ax.bar(x - width/2, success_means, width, alpha=0.8,
                           color=bar_colors, yerr=success_stds, capsize=5,
                           error_kw={'linewidth':1.5, 'ecolor':'darkblue'})
            bars2 = ax2.bar(x + width/2, path_means, width, alpha=0.8,
                            color='coral', yerr=path_stds, capsize=5,
                            error_kw={'linewidth':1.5, 'ecolor':'darkred'})
            ax.set_ylim(0, 100)
            ax2.set_ylim(0, size)
            if idx == 0:
                ax.set_ylabel('Success Rate (%)', fontsize=18, fontweight='bold', color='steelblue')
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelcolor='steelblue')
            if idx == n_sizes-1:
                ax2.set_ylabel('Longest Path Length\n(when failed)', fontsize=18, fontweight='bold', color='coral')
            else:
                ax2.set_ylabel('')
                ax2.tick_params(axis='y', labelcolor='coral')
            ax.set_title(f'n = {size}', fontsize=22, fontweight='bold')
            display_names = [algo_labels.get(algo, algo) for algo in main_algorithms]
            ax.set_xticks(x)
            ax.set_xticklabels(display_names, fontsize=18, rotation=30, ha='right')
            ax.tick_params(axis='y', labelcolor='steelblue', labelsize=16)
            ax2.tick_params(axis='y', labelcolor='coral', labelsize=16)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            for bar, mean in zip(bars1, success_means):
                if mean > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x()+bar.get_width()/2., height+1.5,
                            f'{mean:.1f}%', ha='center', va='bottom',
                            fontsize=14, fontweight='bold', color='steelblue')
            for bar, mean in zip(bars2, path_means):
                if mean > 0.1:
                    height = bar.get_height()
                    ax2.text(bar.get_x()+bar.get_width()/2., height+0.5,
                             f'{mean:.1f}', ha='center', va='bottom',
                             fontsize=14, fontweight='bold', color='coral')
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_main_comparison.pdf", dpi=150, bbox_inches='tight', format='pdf')
            print(f"Main comparison chart saved to {save_path}_main_comparison.pdf")
            manuscript_dir = os.path.dirname(save_path)
            figure04_path = os.path.join(manuscript_dir, 'Figure04.pdf') if manuscript_dir else 'Figure04.pdf'
            plt.savefig(figure04_path, dpi=150, bbox_inches='tight', format='pdf')
            print(f"Main comparison chart also saved as {figure04_path}")
        plt.show()
        return fig, axes

    @staticmethod
    def plot_ablation_study(df: pd.DataFrame, save_path: str = None):
        if df is None or len(df) == 0:
            print("No data to plot")
            return
        if 'Graph Size' not in df.columns:
            print("Error: 'Graph Size' column not found in DataFrame")
            return
        ablation_algorithms = [
            "GeneticAlgorithm",
            "Density-Biased-GA",
            "GBS-InitOnly",
            "GBS-MutationOnly",
            "GBS-FitnessOnly",
            "GBS-NoSubgraph",
            "GBS-SubgraphOnly",
            "GBS-Genetic",
            "MultiStageGBS",
        ]
        df_ablation = df[df['Algorithm'].isin(ablation_algorithms)]
        if len(df_ablation) == 0:
            print("No data for ablation study")
            return
        size_algo_stats = defaultdict(lambda: defaultdict(dict))
        for (graph_size, algo), group in df_ablation.groupby(['Graph Size', 'Algorithm']):
            total_trials = len(group)
            successes = group['Success'].sum()
            success_rate = (successes / total_trials) * 100
            p = successes / total_trials
            se = np.sqrt(p * (1-p) / total_trials) * 100
            size_algo_stats[graph_size][algo]['success_mean'] = success_rate
            size_algo_stats[graph_size][algo]['success_std'] = se
        target_sizes = [15,20,25,30,35,40]
        graph_sizes = sorted([size for size in size_algo_stats.keys() if size in target_sizes])
        if len(graph_sizes) == 0:
            graph_sizes = sorted(size_algo_stats.keys())
        n_sizes = len(graph_sizes)
        if n_sizes == 0:
            print("No data to plot")
            return
        algo_labels = {
            "GeneticAlgorithm": "GA",
            "Density-Biased-GA": "Density-Biased",
            "GBS-InitOnly": "InitOnly",
            "GBS-MutationOnly": "MutationOnly",
            "GBS-FitnessOnly": "FitnessOnly",
            "GBS-NoSubgraph": "EdgeOnly",
            "GBS-SubgraphOnly": "SubgraphOnly",
            "GBS-Genetic": "BipartiteGBS-GA",
            "MultiStageGBS": "MultiStageGBS"
        }
        ablation_colors = {
            "GeneticAlgorithm": "#999999",
            "Density-Biased-GA": "#A6D854",
            "GBS-InitOnly": "#1F78B4",
            "GBS-MutationOnly": "#E78AC3",
            "GBS-FitnessOnly": "#8DA0CB",
            "GBS-NoSubgraph": "#FFD92F",
            "GBS-SubgraphOnly": "#1B9E77",
            "GBS-Genetic": "#D95F02",
            "MultiStageGBS": "#E41A1C"
        }
        fig, axes = plt.subplots(1, len(graph_sizes), figsize=(7.5*len(graph_sizes), 6))
        if n_sizes == 1:
            axes = [axes]
        for idx, size in enumerate(graph_sizes):
            ax = axes[idx]
            success_means = []; success_stds = []
            for algo in ablation_algorithms:
                stats = size_algo_stats[size].get(algo, {})
                success_means.append(stats.get('success_mean', 0))
                success_stds.append(stats.get('success_std', 0))
            x = np.arange(len(ablation_algorithms))
            colors = [ablation_colors.get(algo, "#AAAAAA") for algo in ablation_algorithms]
            bars = ax.bar(x, success_means, alpha=0.8, color=colors,
                          yerr=success_stds, capsize=5,
                          error_kw={'linewidth':1.5, 'ecolor':'darkgray'})
            ax.set_ylim(0, 100)
            if idx == 0:
                ax.set_ylabel('Success Rate (%)', fontsize=20, fontweight='bold')
            else:
                ax.set_ylabel('')
            ax.set_title(f'n = {size}', fontsize=24, fontweight='bold', pad=12)
            display_names = [algo_labels.get(algo, algo) for algo in ablation_algorithms]
            ax.set_xticks(x)
            ax.set_xticklabels(display_names, fontsize=20, rotation=45, ha='right')
            ax.tick_params(axis='y', labelsize=24)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            for bar, mean in zip(bars, success_means):
                if mean > 1:
                    height = bar.get_height()
                    y_pos = min(height+2, 98)
                    ax.text(bar.get_x()+bar.get_width()/2, y_pos,
                            f'{mean:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_ablation_study.pdf", dpi=150, bbox_inches='tight', format='pdf')
            print(f"Ablation study chart saved to {save_path}_ablation_study.pdf")
            manuscript_dir = os.path.dirname(save_path)
            figure06_path = os.path.join(manuscript_dir, 'Figure06.pdf') if manuscript_dir else 'Figure06.pdf'
            plt.savefig(figure06_path, dpi=150, bbox_inches='tight', format='pdf')
            print(f"Ablation study chart also saved as {figure06_path}")
        plt.show()
        return fig, axes

# ==================== Main program ====================
def main():
    print("="*60)
    print("EXPERIMENT: Hamiltonian Cycle Search on Directed Graphs")
    print("Comparing: GA, Density-Biased-GA,")
    print("GBS Ablation Variants, Full GBS")
    print("="*60)
    runner = ExperimentRunner(seed=42)
    test_sizes = [15, 20, 25, 30, 35, 40]
    results = runner.run_batch(
        sizes=test_sizes,
        graphs_per_type=60,
        n_trials=10
    )
    runner.save_to_excel()
    df_results = runner.excel_manager.load_results()
    df_sampling, df_detail = runner.excel_manager.load_sampling_data()

    if df_results is not None:
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"Total experiments: {len(df_results)}")
        summary = df_results.groupby(['Graph Size', 'Algorithm'])['Success'].agg(['mean', 'count'])
        summary['Success Rate %'] = summary['mean'] * 100
        print("\nSuccess Rates by Size and Algorithm:")
        print(summary.round(2))

        # Report the best algorithm at each graph size.
        print("\n" + "="*60)
        print("BEST ALGORITHM PER GRAPH SIZE")
        print("="*60)
        pivot = df_results.groupby(['Graph Size', 'Algorithm'])['Success'].mean().unstack() * 100
        for size in pivot.index:
            row = pivot.loc[size]
            best = row.idxmax()
            best_val = row[best]
            star = " ★ (best)" if best == "GBS-InitOnly" else ""
            print(f"n={size:2d} : {best:20s}  {best_val:5.1f}%{star}")

        # Compare InitOnly with the full single-stage variant.
        print("\n" + "="*60)
        print("INITONLY vs FULL GBS-GENETIC")
        print("="*60)
        for size in pivot.index:
            init = pivot.loc[size, "GBS-InitOnly"] if "GBS-InitOnly" in pivot.columns else 0
            full = pivot.loc[size, "GBS-Genetic"] if "GBS-Genetic" in pivot.columns else 0
            diff = init - full
            print(f"n={size:2d} : InitOnly={init:5.1f}%, Full={full:5.1f}%, Diff={diff:+.1f}%")

        # Paired statistical tests.
        perform_all_pairwise_tests(df_results, test_sizes)

        # Visualization.
        visualizer = EnhancedVisualizer()
        visualizer.plot_main_comparison(df_results, save_path="experiment_results")
        visualizer.plot_ablation_study(df_results, save_path="experiment_results")

    if df_sampling is not None:
        print("\n" + "="*60)
        print("GBS SAMPLING SUMMARY")
        print("="*60)
        print(f"Total GBS sampling records: {len(df_sampling)}")
        gbs_summary = df_sampling.groupby(['Graph Size']).agg({
            'Effective Samples': 'mean',
            'Effective Rate (%)': 'mean',
            'Avg Subgraph Size': 'mean',
            'Max Subgraph Size': 'mean',
            'Sample Time (s)': 'mean'
        }).round(2)
        print("\nGBS Sampling Statistics by Graph Size:")
        print(gbs_summary)

        # Report accepted-sample rates explicitly.
        print("\n" + "="*60)
        print("EFFECTIVE SAMPLE RATE (ACCEPTANCE RATE) PER GRAPH SIZE")
        print("(This is the fraction of 500 shots that passed postselection)")
        print("="*60)
        rate_stats = df_sampling.groupby('Graph Size')['Effective Rate (%)'].agg(
            mean='mean', std='std', min='min', max='max', count='count'
        ).round(2)
        print(rate_stats)
        print("\nNote: This rate directly conditions the quality of co-occurrence estimates.")
        print("Lower acceptance rates for larger n imply more shots needed for stable GBS guidance.")

        if df_detail is not None:
            print(f"\nTotal subgraph details: {len(df_detail)}")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    print(f"📁 Results saved to: {runner.excel_manager.result_filename}")
    print(f"📁 Sampling data saved to: {runner.excel_manager.sampling_filename}")
    return runner, df_results, df_sampling

if __name__ == "__main__":
    runner, df_results, df_sampling = main()
