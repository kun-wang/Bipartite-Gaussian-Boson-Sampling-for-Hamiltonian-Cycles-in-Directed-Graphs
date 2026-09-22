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

"""Sensitivity test for the two ad-hoc parameters of MultiStageGBS (appendix).

Scan 1: reinforcement level (the value good edges are boosted to in _merge_gbs_info)
          -> reinforcement in {0.5, 0.7, 0.8, 0.9, 1.0}   (default 0.8)
Scan 2: stage-2 generation budget (fraction of G_max^MS = 300)
          -> stage2_frac in {1/3, 1/2, 2/3}                (default 1/2 -> 150 gens)

Neither parameter affects the BipartiteGBS sampling itself, so each graph is sampled
once and the SAME samples are reused across all values. To isolate the effect of the
parameter under test, every (graph, trial) is run with a fixed random seed, so Stage 1
(the plain classical GA, which uses no BipartiteGBS information and no scanned parameter)
produces an identical result across all parameter values; only Stage 2 differs.

main_benchmark.py is not modified: the two parameters are made configurable by a local subclass of
MultiStageGBSEnhancedGenetic that overrides ``search`` (stage-2 budget) and
``_merge_gbs_info`` (reinforcement level).
"""
import matplotlib
matplotlib.use('Agg')
import importlib
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

m = importlib.import_module('main_benchmark')

GraphGenerator = m.GraphGenerator
BipartiteGBSSampler = m.BipartiteGBSSampler
MultiStageGBSEnhancedGenetic = m.MultiStageGBSEnhancedGenetic


class MultiStageGBSParams(MultiStageGBSEnhancedGenetic):
    """MultiStageGBS with a configurable stage-2 budget fraction and reinforcement level."""

    def __init__(self, pop_size: int, max_generations: int,
                 stage2_frac: float = 0.5, reinforcement: float = 0.8):
        super().__init__(pop_size, max_generations)
        self.name = "MultiStageGBS"
        self.stage2_frac = stage2_frac
        self.reinforcement = reinforcement

    def search(self, graph, gbs_info=None):
        start_time = time.time()

        stage1 = m.GeneticAlgorithm(
            pop_size=self.pop_size // 2,
            max_generations=self.max_generations // 3,
        )
        result1 = stage1.search(graph, None)

        if result1.success:
            result1.time_seconds = time.time() - start_time
            result1.algorithm_name = self.name
            result1.gbs_used = True
            return result1

        good_edges = self._extract_good_edges(
            result1.longest_path if result1.longest_path else [])

        stage2 = m.GBSEnhancedGenetic(
            pop_size=self.pop_size,
            max_generations=int(self.max_generations * self.stage2_frac),
            alpha=0.1,
            beta=0.2,
        )
        merged_info = self._merge_gbs_info(gbs_info, good_edges) if gbs_info else None
        result2 = stage2.search(graph, merged_info)
        result2.time_seconds = time.time() - start_time
        result2.algorithm_name = self.name
        result2.gbs_used = True
        return result2

    def _merge_gbs_info(self, gbs_info, good_edges):
        merged = copy.deepcopy(gbs_info)
        for edge in good_edges:
            merged.edge_prob[edge] = max(merged.edge_prob.get(edge, 0), self.reinforcement)
            if edge not in merged.top_edges:
                merged.top_edges.insert(0, edge)
        return merged


# Match the main benchmark (main_benchmark.py _make_algorithms): MultiStageGBS uses
# pop_size=100, max_generations=300 (stage1=100, stage2=150).
SIZES = [20, 25]
NUM_GRAPHS = 30
TRIALS = 10
POP_SIZE = 100
MAX_GENERATIONS = 300
NUM_SAMPLES = 500

REINFORCEMENT_VALUES = [0.5, 0.7, 0.8, 0.9, 1.0]
STAGE2_FRACS = [1 / 3, 1 / 2, 2 / 3]

# ---- sample BipartiteGBS once per graph (shared across both scans) ----
print("=" * 66)
print("Sampling BipartiteGBS (once per graph, reused across all values)")
print("=" * 66)
graphs = []
for size in SIZES:
    for i in range(NUM_GRAPHS):
        seed = 42 + i * 100 + size * 10
        g = GraphGenerator.generate_random_digraph(size, 0.3, seed)
        g.name = f"Sens_n{size}_{i + 1}"
        gbs_info = BipartiteGBSSampler(g).get_gbs_info(num_samples=NUM_SAMPLES)
        graphs.append((g, gbs_info))
        print(f"  [n={size}] graph {i + 1}/{NUM_GRAPHS} "
              f"(effective={gbs_info.effective_samples})", flush=True)


def run_one(algo_factory):
    """algo_factory() -> fresh MultiStageGBS; returns {size: list of 0/1 success}.

    Each (graph, trial) uses a fixed seed so Stage 1 is identical across all
    parameter values; only the scanned parameter differs.
    """
    per_size = {s: [] for s in SIZES}
    for gi, (g, gbs_info) in enumerate(graphs):
        for t in range(TRIALS):
            seed = 1000 + gi * 100 + t
            random.seed(seed)
            np.random.seed(seed)
            algo = algo_factory()
            res = algo.search(g, gbs_info)
            per_size[g.n].append(1 if res.success else 0)
    return per_size


def summarize(per_size):
    out = {}
    for s in SIZES:
        v = np.asarray(per_size[s], dtype=float)
        rate = 100.0 * v.mean()
        se = 100.0 * np.sqrt(v.mean() * (1.0 - v.mean()) / len(v)) if len(v) else 0.0
        out[s] = (rate, se, int(v.sum()), len(v))
    return out


# ---- Scan 1: reinforcement level (default 0.8) ----
print("\n" + "=" * 66)
print("SCAN 1: reinforcement level (default 0.8)")
print("=" * 66)
reinf_results = {}
for rho in REINFORCEMENT_VALUES:
    flags = run_one(lambda rho=rho: MultiStageGBSParams(
        pop_size=POP_SIZE, max_generations=MAX_GENERATIONS, reinforcement=rho))
    reinf_results[rho] = summarize(flags)
    line = "  ".join(f"n={s}: {r[0]:.1f}% ({r[2]}/{r[3]})"
                     for s, r in reinf_results[rho].items())
    print(f"  reinforcement={rho:<4}  {line}", flush=True)

# ---- Scan 2: stage-2 budget fraction (default 1/2 -> 150 gens) ----
print("\n" + "=" * 66)
print("SCAN 2: stage-2 budget fraction of G_max^MS=300 (default 1/2)")
print("=" * 66)
frac_results = {}
for f in STAGE2_FRACS:
    flags = run_one(lambda f=f: MultiStageGBSParams(
        pop_size=POP_SIZE, max_generations=MAX_GENERATIONS, stage2_frac=f))
    frac_results[f] = summarize(flags)
    line = "  ".join(f"n={s}: {r[0]:.1f}% ({r[2]}/{r[3]})"
                     for s, r in frac_results[f].items())
    print(f"  stage2_frac={f:.3f}  {line}", flush=True)

# ---- figure (two panels) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, results, xvals, xlabel, title, xtickfmt in [
    (axes[0], reinf_results, REINFORCEMENT_VALUES, 'reinforcement level',
     'MultiStageGBS: reinforcement-level sensitivity', lambda v: f'{v:g}'),
    (axes[1], frac_results, STAGE2_FRACS, r'stage-2 budget (fraction of $G_{\max}^{\mathrm{MS}}$)',
     'MultiStageGBS: stage-2 budget sensitivity', lambda v: f'{v:.2f}'),
]:
    for s, color in zip(SIZES, ['#1F78B4', '#E78AC3']):
        rates = [results[v][s][0] for v in xvals]
        ses = [results[v][s][1] for v in xvals]
        ax.errorbar(xvals, rates, yerr=ses, marker='o', capsize=3,
                    label=f'n={s}', color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Success rate (%)')
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.set_xticks(xvals)
    ax.set_xticklabels([xtickfmt(v) for v in xvals])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('sensitivity_ms_gbs.pdf', dpi=150, bbox_inches='tight')
print("\nSaved sensitivity_ms_gbs.pdf")
print("Note: 30 instances x 10 runs = 300 per (size, value); error bars are +/- 1 SE.")
