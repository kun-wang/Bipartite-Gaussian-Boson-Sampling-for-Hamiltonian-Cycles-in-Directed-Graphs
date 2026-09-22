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

"""Compare default co-occurrence values 0 and 0.5 in the full guided GA.

Compares, on the first 20 saved graphs at n=20,30 and their accepted sample banks
(no new photon sampling), the full guided GA (GBSEnhancedGenetic = GBS-Genetic)
under two conventions:
  A  "zero"  : init/build-path default = 0, mutation default = 0, fitness default = 0
               (the convention used in the main benchmark, main_benchmark.py DEFAULT_EDGE_PROB=0.0)
  B  "mixed" : init/build-path default = 0.5, mutation default = 0.5, fitness default = 0
               (the original mixed convention)

Graphs, banks, hyperparameters, and per-trial GA seeds are held fixed across the two
settings so that any difference is attributable only to the default value.
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
import importlib, sys, random, os
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, str(REPO_ROOT))
m = importlib.import_module('main_benchmark')

GraphGenerator = m.GraphGenerator
DirectedGraph = m.DirectedGraph
GBSInfo = m.GBSInfo
GBSEnhancedGenetic = m.GBSEnhancedGenetic
DEFAULT_EDGE_PROB = m.DEFAULT_EDGE_PROB   # 0.0

# ---- parameterised full guided GA with per-component defaults ------------------
class ParamGA(GBSEnhancedGenetic):
    """GBS-Genetic with separate default edge-probability per component."""
    def __init__(self, init_default=0.0, mutation_default=0.0, fitness_default=0.0,
                 pop_size=100, max_generations=200, alpha=0.1, beta=0.2):
        super().__init__(pop_size=pop_size, max_generations=max_generations,
                         alpha=alpha, beta=beta)
        self.init_default = init_default
        self.mutation_default = mutation_default
        self.fitness_default = fitness_default

    # init / build-path (used by _construct_from_subgraph)
    def _construct_from_subgraph(self, graph, vertices, gbs_info):
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
                    prob = gbs_info.edge_prob.get((current, v), self.init_default)
                    candidates.append((v, prob))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                next_v = candidates[0][0]
                path.append(next_v); used.add(next_v); current = next_v; remaining.remove(next_v)
            else:
                break
        all_vertices = set(range(n))
        remaining_all = list(all_vertices - used)
        random.shuffle(remaining_all)
        path.extend(remaining_all)
        return path

    # fitness
    def _gbs_enhanced_fitness(self, graph, individual, gbs_info, alpha):
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
        gbs_score = 0; edge_count = 0
        for i in range(graph.n - 1):
            edge = (individual[i], individual[i+1])
            gbs_score += gbs_info.edge_prob.get(edge, self.fitness_default)
            edge_count += 1
        if can_close:
            edge = (individual[-1], individual[0])
            gbs_score += gbs_info.edge_prob.get(edge, self.fitness_default)
            edge_count += 1
        gbs_score = gbs_score / edge_count if edge_count > 0 else 0
        return base_fitness * (1 - alpha) + gbs_score * alpha

    # mutation
    def _gbs_guided_mutate(self, individual, gbs_info):
        n = len(individual)
        if gbs_info and gbs_info.edge_prob and random.random() < 0.7:
            edge_scores = []
            for i in range(n):
                u = individual[i]
                v = individual[(i+1) % n]
                score = gbs_info.edge_prob.get((u, v), self.mutation_default)
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


# ---- reconstruct GBSInfo from saved sample bank (mirrors get_gbs_info) --------
def parse_samples(all_subgraphs_str):
    samples = []
    if not isinstance(all_subgraphs_str, str):
        return samples
    for tok in all_subgraphs_str.split(';'):
        tok = tok.strip()
        if not tok.startswith('{'):
            continue
        inner = tok.strip('{}')
        if inner == '':
            continue
        samples.append({int(x) for x in inner.split(',')})
    return samples

def build_gbs_info(graph, samples):
    vertex_freq = Counter(); edge_freq = Counter()
    for vertices in samples:
        for v in vertices:
            vertex_freq[v] += 1
        vl = sorted(vertices)
        for i in range(len(vl)):
            for j in range(i+1, len(vl)):
                u, v = vl[i], vl[j]
                if graph.has_edge(u, v):
                    edge_freq[(u, v)] += 1
                if graph.has_edge(v, u):
                    edge_freq[(v, u)] += 1
    total = len(samples) if len(samples) > 0 else 1
    vertex_prob = {v: f/total for v, f in vertex_freq.items()}
    edge_prob = {e: f/total for e, f in edge_freq.items()}
    top_edges = [e for e, _ in edge_freq.most_common(min(graph.n*2, len(edge_freq)))]
    return GBSInfo(
        vertex_freq=vertex_freq, edge_freq=edge_freq, path_freq=Counter(),
        vertex_prob=vertex_prob, edge_prob=edge_prob,
        top_vertices=[v for v, _ in vertex_freq.most_common(min(graph.n, graph.n))],
        top_edges=top_edges, top_paths=[], samples=samples,
        sample_count=500, effective_samples=len(samples),
        max_subgraph_size=max([len(s) for s in samples]) if samples else 0,
        avg_subgraph_size=np.mean([len(s) for s in samples]) if samples else 0,
        subgraph_size_distribution={},
    )

# ---- load saved banks ---------------------------------------------------------
def load_banks():
    df = pd.read_excel(str(REPO_ROOT / 'gbs_sampling_data.xlsx'),
                       sheet_name='GBS Sampling Data')
    banks = {}
    for _, r in df.iterrows():
        banks[(r['Graph Size'], r['Graph Name'])] = parse_samples(r['All Subgraphs'])
    return banks

def run():
    banks = load_banks()
    sizes = [20, 30]
    ngraphs = 20
    trials = 10
    rows = []
    for n in sizes:
        for i in range(ngraphs):
            seed = 42 + i*100 + n*10
            graph = GraphGenerator.generate_random_digraph(n, 0.3, seed)
            gname = f"Random_n{n}_{i+1}"
            graph.name = gname
            samples = banks.get((n, gname), [])
            gbs_info = build_gbs_info(graph, samples)
            for setting, (idv, mut, fit) in {
                'zero': (0.0, 0.0, 0.0),
                'mixed': (0.5, 0.5, 0.0),
            }.items():
                for t in range(trials):
                    rnd = 1000 + i*100 + t
                    random.seed(rnd); np.random.seed(rnd)
                    algo = ParamGA(init_default=idv, mutation_default=mut,
                                   fitness_default=fit, pop_size=100,
                                   max_generations=200, alpha=0.1, beta=0.2)
                    res = algo.search(graph, gbs_info)
                    rows.append({
                        'n': n, 'graph_id': i+1, 'graph_name': gname,
                        'setting': setting, 'init_default': idv,
                        'mutation_default': mut, 'fitness_default': fit,
                        'trial': t+1, 'seed': rnd,
                        'success': int(res.success),
                        'longest_path': len(res.longest_path) if res.longest_path else 0,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(str(REPO_ROOT / 'default_check_results.csv'), index=False)

    # ---- paired graph-level summary -----------------------------------------
    print("=== DEFAULT CHECK: zero vs mixed (full guided GA, n=20,30; 20 graphs x 10 trials) ===\n")
    summary = []
    for n in sizes:
        d = df[df['n'] == n]
        z = d[d['setting'] == 'zero'].groupby('graph_id')['success'].sum()
        mx = d[d['setting'] == 'mixed'].groupby('graph_id')['success'].sum()
        common = sorted(set(z.index) & set(mx.index))
        zz = z[common].values.astype(float); mm = mx[common].values.astype(float)
        diff = (zz - mm) * 10.0   # percentage points (10 trials)
        from scipy import stats
        if np.all(diff == 0):
            p = 1.0
        else:
            p = stats.wilcoxon(diff, alternative='two-sided')[1]
        # bootstrap CI
        rng = np.random.default_rng(20260919)
        B = 20000
        means = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, len(diff), size=len(diff))
            means[b] = diff[idx].mean()
        lo, hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
        print(f"n={n}: zero rate={zz.mean()*10:.1f}%  mixed rate={mm.mean()*10:.1f}%  "
              f"mean diff (zero-mixed)={diff.mean():+.2f} pp  CI95=[{lo:+.2f},{hi:+.2f}]  "
              f"Wilcoxon p={p:.3g}")
        summary.append({
            'n': n, 'zero_rate%': round(zz.mean()*10, 2),
            'mixed_rate%': round(mm.mean()*10, 2),
            'mean_diff_pp': round(diff.mean(), 2),
            'CI95_lo': round(lo, 2), 'CI95_hi': round(hi, 2),
            'wilcoxon_p': f"{p:.3g}",
        })
    pd.DataFrame(summary).to_csv(str(REPO_ROOT / 'default_check_summary.csv'), index=False)
    print("\nWrote default_check_results.csv and default_check_summary.csv")

if __name__ == '__main__':
    run()
