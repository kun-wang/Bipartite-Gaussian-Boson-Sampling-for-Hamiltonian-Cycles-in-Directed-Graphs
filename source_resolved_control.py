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

"""Run the source-resolved classical controls used in the main benchmark.

The experiment uses the same 60 graphs per size as ``main_benchmark.py``.

It runs two methods on the saved benchmark graphs:
  * SourceResolved-GA     : single-stage GBSEnhancedGenetic  fed source-resolved info
  * MS-SourceResolved-GA  : MultiStageGBSEnhancedGenetic     fed source-resolved info

This is the retained standalone control runner. The revised manuscript's paired
main outcomes are generated jointly by replay_benchmark.py, not by this script.
The original control analysis reused the quantum side from experiment_results.xlsx.
Downstream operators / hyperparameters are identical to the quantum methods:
  single : pop 100, 200 gens, alpha=0.1, beta=0.2
  multi  : pop 100, 300 gens (stage1 50/100, stage2 100/150), reinforcement 0.8

Seeds:
  graph seed     = 42 + i*100 + n*10   (identical to main_benchmark.py)
  sampler rng    = same as graph seed (deterministic; raw counts regenerable)
  per-trial GA   = 1000 + i*100 + t

Saves:
  sr_control_results.csv   (trial outcomes, same schema as experiment_results.xlsx)
  sr_control_sampling.csv  (per-graph accepted counts / size distributions / seeds)
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
import os, sys, random, time
import numpy as np
import pandas as pd

sys.path.insert(0, str(REPO_ROOT))
import source_resolved as sr
import importlib
m = importlib.import_module('main_benchmark')

SIZES = [15, 20, 25, 30, 35, 40]
NUM_GRAPHS = 60
TRIALS = 10
NUM_SHOTS = 500
CODE_VERSION = 'main_benchmark.py @ 2026-09-14 + source_resolved.py (source-resolved control)'

def run_graph(args):
    n, i = args
    seed = 42 + i * 100 + n * 10
    g = m.GraphGenerator.generate_random_digraph(n, 0.3, seed)
    g.name = f"Random_n{n}_{i+1}"

    sampler_rng = np.random.default_rng(seed)
    info = sr.SourceResolvedSampler(g).get_gbs_info(num_shots=NUM_SHOTS, rng=sampler_rng)

    trial_rows = []
    for method, make in [
        ('SourceResolved-GA', lambda: m.GBSEnhancedGenetic(pop_size=100, max_generations=200, alpha=0.1, beta=0.2)),
        ('MS-SourceResolved-GA', lambda: m.MultiStageGBSEnhancedGenetic(pop_size=100, max_generations=300)),
    ]:
        for t in range(TRIALS):
            rnd = 1000 + i * 100 + t
            random.seed(rnd); np.random.seed(rnd)
            algo = make()
            res = algo.search(g, info)
            trial_rows.append({
                'Graph Name': g.name, 'Graph Type': 'Random', 'Graph Size': n,
                'Algorithm': method, 'Success': 1 if res.success else 0,
                'Path Length': res.path_length,
                'Longest Path Length': len(res.longest_path) if res.longest_path else 0,
                'Time (seconds)': res.time_seconds, 'Generations': res.generations,
                'GBS Used': res.gbs_used,
            })

    # per-graph sampling summary (size distribution of accepted subsets)
    sizes_count = {}
    for s in info.samples:
        sizes_count[len(s)] = sizes_count.get(len(s), 0) + 1
    size_dist_str = ', '.join(f"{k}:{v}" for k, v in sorted(sizes_count.items()))
    sampling_row = {
        'Graph Name': g.name, 'Graph Size': n, 'graph_seed': seed,
        'sampler_rng_seed': seed, 'num_shots': NUM_SHOTS,
        'accepted_samples': info.effective_samples,
        'acceptance_rate%': round(100.0 * info.effective_samples / NUM_SHOTS, 2),
        'mean_pairs_theory': round(sr.SourceResolvedSampler(g).mean_pairs, 4),
        'subgraph_size_distribution': size_dist_str,
        'code_version': CODE_VERSION,
    }
    return trial_rows, sampling_row

if __name__ == '__main__':
    tasks = [(n, i) for n in SIZES for i in range(NUM_GRAPHS)]
    workers = int(os.environ.get('WORKERS', min(16, os.cpu_count() or 4)))
    print(f"Running {len(tasks)} graphs x (sampler + 2 methods x 10 trials) on {workers} workers...", flush=True)
    t0 = time.time()
    trial_rows, sampling_rows = [], []
    from multiprocessing import Pool
    with Pool(processes=workers) as pool:
        for k, (tr, sr_) in enumerate(pool.imap_unordered(run_graph, tasks, chunksize=4)):
            trial_rows.extend(tr); sampling_rows.append(sr_)
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(tasks)} graphs done, {time.time()-t0:.0f}s", flush=True)
    print(f"Total wall time {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(trial_rows)
    df.to_csv(str(REPO_ROOT / 'sr_control_results.csv'), index=False)
    pd.DataFrame(sampling_rows).to_csv(str(REPO_ROOT / 'sr_control_sampling.csv'), index=False)

    print("\n=== SOURCE-RESOLVED CONTROL: success rate by size ===")
    summ = df.groupby(['Graph Size', 'Algorithm'])['Success'].agg(['count', 'sum', 'mean'])
    summ['rate%'] = (summ['mean'] * 100).round(1)
    print(summ[['sum', 'rate%']].to_string())
    print(f"\nSaved sr_control_results.csv ({len(df)} rows), sr_control_sampling.csv ({len(sampling_rows)} rows)")
