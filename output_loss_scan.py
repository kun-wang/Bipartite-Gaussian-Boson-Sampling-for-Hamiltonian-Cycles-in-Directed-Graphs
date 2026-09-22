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

"""Run the uniform output photon loss experiment reported in Fig. 7.

The implementation builds the BipartiteGBS covariance matrix directly with
NumPy and samples the quantum raw mode counts with
``thewalrus.samples.hafnian_sample_state``. The classical raw mode counts come
from the source-resolved sampler.

Archived scan scope:
  * first 30 graphs at each of n=20, 30;
  * 500 raw shots per sampler per graph;
  * transmissions tau = 1, 0.95, 0.90, 0.80;
  * both schedules (single-stage GBSEnhancedGenetic and multistage
    MultiStageGBSEnhancedGenetic), 10 trials per (sampler, schedule, tau).

Thinning: c'_j ~ Binomial(c_j, tau) on raw mode counts, then reapply the SAME
postselection (collision-free, balanced arm, >= 3 pairs). Raw quantum shots are
generated once per graph and reused for all tau.

Seeds: graph seed = 42 + i*100 + n*10; quantum/classical raw-shot rng = graph
seed; thinning rng = 555 + i*100 + n*10 (same stream for every tau); per-trial
GA seed = 1000 + i*100 + t (identical to source_resolved_control.py).
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
import os, sys, time, random
import numpy as np
import pandas as pd
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, str(REPO_ROOT))
import source_resolved as sr
import importlib
m = importlib.import_module('main_benchmark')
from thewalrus.samples import hafnian_sample_state

SIZES = [20, 30]
NUM_GRAPHS = 30
TRIALS = 10
NUM_SHOTS = 500
TAUS = [1.0, 0.95, 0.90, 0.80]


def build_cov(graph):
    """Return the real-adjacency BipartiteGBS covariance in xxpp ordering, hbar=2."""
    B = graph.to_adjacency_matrix().astype(np.float64)
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    V = Vt.T.conj()
    r = np.arctanh(0.75 * s / np.max(s))
    n = graph.n
    N = 2 * n
    c = np.cosh(r); sh = np.sinh(r)
    # covariance of independent two-mode squeezers (xp ordering)
    cov_sq = np.eye(2 * N)
    for i in range(n):
        ci, si = c[i], sh[i]
        a, b = i, i + n
        idx = [2 * a, 2 * a + 1, 2 * b, 2 * b + 1]
        blk = np.array([[ci ** 2 + si ** 2, 0, 2 * ci * si, 0],
                        [0, ci ** 2 + si ** 2, 0, -2 * ci * si],
                        [2 * ci * si, 0, ci ** 2 + si ** 2, 0],
                        [0, -2 * ci * si, 0, ci ** 2 + si ** 2]])
        cov_sq[np.ix_(idx, idx)] = blk
    # passive interferometers U (upper), V (lower): S_int in xp ordering
    S_int = np.eye(2 * N)
    Ublock = np.block([[U, np.zeros((n, n))], [np.zeros((n, n)), V]])
    pos_idx = np.arange(0, 2 * N, 2)
    mom_idx = np.arange(1, 2 * N, 2)
    S_int[np.ix_(pos_idx, pos_idx)] = Ublock
    S_int[np.ix_(mom_idx, mom_idx)] = Ublock
    cov_xp = S_int @ cov_sq @ S_int.T
    # thewalrus expects xxpp ordering [x_0..x_{N-1}, p_0..p_{N-1}]
    N = graph.n * 2
    order = np.concatenate([np.arange(0, 2 * N, 2), np.arange(1, 2 * N, 2)])
    return cov_xp[np.ix_(order, order)]


def quantum_raw_patterns(graph, seed, num_shots=NUM_SHOTS):
    cov = build_cov(graph)
    np.random.seed(seed)
    samples = hafnian_sample_state(cov, samples=num_shots,
                                   mean=np.zeros(4 * graph.n), cutoff=8, max_photons=30)  # explicit documented cap; historical package version unrecorded
    return [np.asarray(p, dtype=int) for p in samples]


def classical_raw_patterns(graph, seed, num_shots=NUM_SHOTS):
    s = sr.SourceResolvedSampler(graph)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_shots):
        up, lo = s.raw_shot(rng)
        out.append(np.concatenate([up, lo]))
    return out


def thin_and_postselect(patterns, m, tau, rng):
    samples = []
    for p in patterns:
        pc = rng.binomial(p, tau)
        s_ph = pc[:m]; t_ph = pc[m:]
        if (s_ph > 1).any() or (t_ph > 1).any():
            continue
        S = set(np.nonzero(s_ph == 1)[0].tolist())
        T = set(np.nonzero(t_ph == 1)[0].tolist())
        if len(S) == len(T) and len(S) >= 3:
            samples.append(S | T)
    return samples


def build_info(graph, samples):
    vertex_freq = Counter(); edge_freq = Counter()
    for verts in samples:
        for v in verts:
            vertex_freq[v] += 1
        vl = sorted(verts)
        for i in range(len(vl)):
            for j in range(i + 1, len(vl)):
                u, v = vl[i], vl[j]
                if graph.has_edge(u, v):
                    edge_freq[(u, v)] += 1
                if graph.has_edge(v, u):
                    edge_freq[(v, u)] += 1
    total = len(samples) if samples else 1
    edge_prob = {e: f / total for e, f in edge_freq.items()}
    top_edges = [e for e, _ in edge_freq.most_common(min(graph.n * 2, len(edge_freq)))]
    return SimpleNamespace(
        vertex_freq=vertex_freq, edge_freq=edge_freq, path_freq=Counter(),
        vertex_prob={v: f / total for v, f in vertex_freq.items()},
        edge_prob=edge_prob,
        top_vertices=[v for v, _ in vertex_freq.most_common(min(graph.n, graph.n))],
        top_edges=top_edges, top_paths=[], samples=samples,
        sample_count=NUM_SHOTS, effective_samples=len(samples),
        max_subgraph_size=max([len(s) for s in samples]) if samples else 0,
        avg_subgraph_size=np.mean([len(s) for s in samples]) if samples else 0.0,
        subgraph_size_distribution={},
    )


def run_graph(args):
    n, i = args
    seed = 42 + i * 100 + n * 10
    graph = m.GraphGenerator.generate_random_digraph(n, 0.3, seed)
    graph.name = f"Random_n{n}_{i+1}"

    qp = quantum_raw_patterns(graph, seed)
    cp = classical_raw_patterns(graph, seed)

    rows = []
    for tau in TAUS:
        rng = np.random.default_rng(555 + i * 100 + n * 10)
        q_samp = thin_and_postselect(qp, n, tau, rng)
        rng = np.random.default_rng(555 + i * 100 + n * 10)
        c_samp = thin_and_postselect(cp, n, tau, rng)
        q_info = build_info(graph, q_samp)
        c_info = build_info(graph, c_samp)
        methods = [
            ('quantum', 'single', lambda: m.GBSEnhancedGenetic(pop_size=100, max_generations=200, alpha=0.1, beta=0.2), q_info),
            ('quantum', 'multi',  lambda: m.MultiStageGBSEnhancedGenetic(pop_size=100, max_generations=300), q_info),
            ('classical', 'single', lambda: m.GBSEnhancedGenetic(pop_size=100, max_generations=200, alpha=0.1, beta=0.2), c_info),
            ('classical', 'multi',  lambda: m.MultiStageGBSEnhancedGenetic(pop_size=100, max_generations=300), c_info),
        ]
        for sampler, sched, make, info in methods:
            succ = 0
            for t in range(TRIALS):
                rnd = 1000 + i * 100 + t
                random.seed(rnd); np.random.seed(rnd)
                res = make().search(graph, info)
                succ += 1 if res.success else 0
            rows.append({
                'n': n, 'graph_id': i + 1, 'tau': tau,
                'sampler': sampler, 'schedule': sched,
                'accepted_samples': info.effective_samples,
                'acceptance_rate%': round(100.0 * info.effective_samples / NUM_SHOTS, 2),
                'success_rate%': round(100.0 * succ / TRIALS, 2),
            })
    return rows


if __name__ == '__main__':
    # Bound BLAS/numba threading to avoid oversubscription on the 128-core host
    # (default OpenBLAS spawned ~74 threads per worker, thrashing the machine).
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMBA_NUM_THREADS'):
        os.environ.setdefault(v, os.environ.get('THREADS', '2'))
    tasks = [(n, i) for n in SIZES for i in range(NUM_GRAPHS)]
    workers = int(os.environ.get('WORKERS', 48))
    print(f"Loss scan v2: {len(tasks)} graphs x 4 tau x 4 (sampler,schedule) x 10 trials, {workers} workers, {NUM_SHOTS} raw shots, threads={os.environ['OMP_NUM_THREADS']}", flush=True)
    t0 = time.time()
    from multiprocessing import Pool
    rows = []
    done = 0
    with Pool(processes=workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_graph, tasks, chunksize=1)):
            rows.extend(r)
            done += 1
            print(f"  {done}/{len(tasks)} graphs done, {time.time()-t0:.0f}s", flush=True)
            # incremental checkpoint so partial results survive an interruption
            if done % 5 == 0 or done == len(tasks):
                pd.DataFrame(rows).to_csv(str(REPO_ROOT / 'loss_scan_results.csv'), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(str(REPO_ROOT / 'loss_scan_results.csv'), index=False)
    print(f"Done in {time.time()-t0:.0f}s; {len(df)} rows saved", flush=True)

    print("\n=== LOSS SCAN: mean success & acceptance by (size, sampler, schedule, tau) ===")
    g = df.groupby(['n', 'sampler', 'schedule', 'tau'])[['success_rate%', 'acceptance_rate%']].mean().round(1)
    print(g.to_string())
