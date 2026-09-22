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

"""Sensitivity test for two BipartiteGBS-GA hyperparameters (appendix).

Scan 1: hybrid parameter beta (GBS-InitOnly)          -> beta  in {0.1, 0.2, 0.3, 0.5}
Scan 2: guided-mutation fraction (GBS-MutationOnly)   -> p     in {0.5, 0.7, 0.9, 1.0}

Neither hyperparameter affects the BipartiteGBS sampling itself, so each graph is
sampled once and the SAME samples are reused across all values; the only factor
that varies is the hyperparameter under test. main_benchmark.py is not modified: the guided-
mutation fraction is made configurable by a local subclass that overrides
``_gbs_guided_mutate`` (whose upstream default of 0.7 corresponds to the 70/30
split).
"""
import matplotlib
matplotlib.use('Agg')
import importlib
import random
import numpy as np
import matplotlib.pyplot as plt

m = importlib.import_module('main_benchmark')

GraphGenerator = m.GraphGenerator
BipartiteGBSSampler = m.BipartiteGBSSampler
GBS_InitOnly = m.GBS_InitOnly
GBS_MutationOnly = m.GBS_MutationOnly
DEFAULT_EDGE_PROB = m.DEFAULT_EDGE_PROB


class MutationOnlyGuidedProb(GBS_MutationOnly):
    """GBS-MutationOnly with a configurable guided-mutation probability."""

    def __init__(self, guided_prob: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.guided_prob = guided_prob

    def _gbs_guided_mutate(self, individual, gbs_info):
        n = len(individual)
        if gbs_info and gbs_info.edge_prob and random.random() < self.guided_prob:
            edge_scores = []
            for i in range(n):
                u = individual[i]
                v = individual[(i + 1) % n]
                score = gbs_info.edge_prob.get((u, v), DEFAULT_EDGE_PROB)
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


# Match the main benchmark (main_benchmark.py _make_algorithms / _run_batch).
SIZES = [20, 25]
NUM_GRAPHS = 30
TRIALS = 10
POP_SIZE = 100
MAX_GENERATIONS = 200
NUM_SAMPLES = 500

BETA_VALUES = [0.1, 0.2, 0.3, 0.5]
SPLIT_VALUES = [0.5, 0.7, 0.9, 1.0]

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
    """algo_factory() -> fresh algorithm; returns {size: list of 0/1 success}."""
    per_size = {s: [] for s in SIZES}
    for g, gbs_info in graphs:
        algo = algo_factory()
        for _ in range(TRIALS):
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


# ---- Scan 1: hybrid parameter beta (GBS-InitOnly) ----
print("\n" + "=" * 66)
print("SCAN 1: hybrid parameter beta (GBS-InitOnly)")
print("=" * 66)
beta_results = {}
for b in BETA_VALUES:
    flags = run_one(lambda b=b: GBS_InitOnly(pop_size=POP_SIZE,
                                             max_generations=MAX_GENERATIONS,
                                             beta=b))
    beta_results[b] = summarize(flags)
    line = "  ".join(f"n={s}: {r[0]:.1f}% ({r[2]}/{r[3]})"
                     for s, r in beta_results[b].items())
    print(f"  beta={b:<4}  {line}")

# ---- Scan 2: guided-mutation fraction (GBS-MutationOnly) ----
print("\n" + "=" * 66)
print("SCAN 2: guided-mutation fraction (GBS-MutationOnly)")
print("=" * 66)
split_results = {}
for p in SPLIT_VALUES:
    flags = run_one(lambda p=p: MutationOnlyGuidedProb(pop_size=POP_SIZE,
                                                       max_generations=MAX_GENERATIONS,
                                                       guided_prob=p))
    split_results[p] = summarize(flags)
    line = "  ".join(f"n={s}: {r[0]:.1f}% ({r[2]}/{r[3]})"
                     for s, r in split_results[p].items())
    print(f"  guided={p:<4}  {line}")

# ---- figure (two panels) ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, results, xvals, xlabel, title in [
    (axes[0], beta_results, BETA_VALUES, r'hybrid parameter $\beta$',
     r'GBS-InitOnly: $\beta$ sensitivity'),
    (axes[1], split_results, SPLIT_VALUES, r'guided-mutation fraction',
     'GBS-MutationOnly: split sensitivity'),
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
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('sensitivity_beta_split.pdf', dpi=150, bbox_inches='tight')
print("\nSaved sensitivity_beta_split.pdf")
print("Note: 30 instances x 10 runs = 300 per (size, value); error bars are +/- 1 SE.")
