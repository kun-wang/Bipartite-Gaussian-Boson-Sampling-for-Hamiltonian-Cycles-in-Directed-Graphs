"""Replay the classical searches with archived quantum banks; no quantum backend needed.

Each graph has a checkpoint containing complete trial records and the actual
multistage insertion order. Existing checkpoints are used only with an identical
configuration, input hash, source hashes, and numerical environment. Full runs
replace the canonical main and source-control records only after all tasks finish.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time
import numpy as np
import pandas as pd
import main_benchmark as m
import matched_control_utils as helpers
import source_resolved as sr

HERE = Path(__file__).resolve().parent
SIZES = [15, 20, 25, 30, 35, 40]
SOURCE_FILES = ["main_benchmark.py", "matched_control_utils.py", "source_resolved.py", "replay_benchmark.py"]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def replay_guidance(graph, bank):
    # Canonical construction removes dependence on set pickle/reconstruction
    # history while retaining the same subsets and uniform vertex choices.
    info = helpers.guidance_from_bank(graph, [sorted(sample) for sample in bank])
    info.sample_count = 500
    return info

def run_graph(task):
    n, index, bank, trials, directory = task
    directory = Path(directory)
    name = f"Random_n{n}_{index+1}"
    checkpoint = directory / f"{name}.json"
    if checkpoint.exists():
        return name
    seed = helpers.graph_seed(n, index)
    graph = m.GraphGenerator.generate_random_digraph(n, 0.3, seed)
    graph.name = name
    info = replay_guidance(graph, bank)
    sampler = sr.SourceResolvedSampler(graph)
    source_bank, raw = sampler.sample(500, rng=np.random.default_rng(seed))
    source_info = replay_guidance(graph, source_bank)
    np.savez_compressed(directory / f"{name}-source-raw.npz", counts=np.asarray(raw, dtype=np.int16))
    rows = []
    algorithms = [(a.name, a, info) for a in m._make_algorithms()]
    algorithms += [
        ("SourceResolved-GA", m.GBSEnhancedGenetic(pop_size=100, max_generations=200, alpha=0.1), source_info),
        ("MS-SourceResolved-GA", m.MultiStageGBSEnhancedGenetic(pop_size=100, max_generations=300), source_info),
    ]
    for label, algorithm, guidance in algorithms:
        for trial in range(trials):
            trial_seed = 20260922 + n*1000000 + index*1000 + trial
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            trace = {}
            if isinstance(algorithm, m.MultiStageGBSEnhancedGenetic):
                original = m.MultiStageGBSEnhancedGenetic._merge_gbs_info
                def merge(current, good_edges):
                    trace["reinforced_edge_iteration"] = [list(e) for e in good_edges]
                    merged = original(algorithm, current, good_edges)
                    trace["top_edges_after_reinforcement"] = [list(e) for e in merged.top_edges]
                    return merged
                algorithm._merge_gbs_info = merge
            result = algorithm.search(graph, guidance)
            if result.success:
                assert len(result.cycle) == n and len(set(result.cycle)) == n
                assert all(graph.has_edge(result.cycle[j], result.cycle[(j+1)%n]) for j in range(n))
            rows.append({
                "Graph Name": name, "Graph Type": "Random", "Graph Size": n,
                "Graph Seed": seed, "Trial": trial, "Trial Seed": trial_seed,
                "Algorithm": label, "Success": int(result.success),
                "Path Length": result.path_length, "Longest Path Length": len(result.longest_path),
                "Time (seconds)": result.time_seconds, "Generations": result.generations,
                "GBS Used": result.gbs_used, "Cycle": result.cycle,
                "Longest Path": result.longest_path, **trace,
            })
    payload = {"graph": name, "graph_seed": seed, "edges": graph.edges,
               "quantum_top_edges": info.top_edges,
               "source_top_edges": source_info.top_edges,
               "source_subsets": [sorted(s) for s in source_bank], "trials": rows,
               "source_summary": {
                   "Graph Name": name, "Graph Size": n, "graph_seed": seed,
                   "sampler_rng_seed": seed, "num_shots": 500,
                   "accepted_samples": len(source_bank), "acceptance_rate%": len(source_bank)/5,
                   "mean_pairs_theory": sampler.mean_pairs,
                   "code_version": "replay_benchmark.py; hashes in replay/manifest.json"}}
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(checkpoint)
    return name

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--graphs", type=int, default=60)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=HERE / "replay")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sizes": args.sizes, "graphs_per_size": args.graphs, "trials": args.trials,
        "edge_probability": 0.3, "unobserved_edge_default": m.DEFAULT_EDGE_PROB,
        "population": 100, "single_generations": 200, "multistage_generations": [100,150],
        "stage_populations": [50,100], "alpha_max": 0.1, "beta": 0.2,
        "crossover_rate": 0.8, "mutation_rate": 0.1, "reinforcement_floor": 0.8,
        "raw_shots": 500, "graph_seed": "42 + index*100 + n*10 (index is zero-based)",
        "trial_seed": "20260922 + n*1000000 + index*1000 + trial (zero-based)",
        "source_sampler_seed": "graph_seed", "top_edges": "Counter.most_common, stable first-occurrence ties; stored list retained; missing reinforced edges prepended in recorded Python set iteration order",
        "subset_construction": "Sorted vertex lists passed to guidance_from_bank, which constructs fresh sets; independent of input set pickle history",
        "quantum_sampling": "Retained accepted subsets; original optical environment unknown; no new quantum shots",
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy","pandas","scipy","matplotlib","openpyxl"]},
        "source_sha256": {p: digest(HERE/p) for p in SOURCE_FILES},
        "bank_sha256": digest(HERE / "gbs_sampling_data.xlsx"),
    }
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("Checkpoint configuration differs; use a new output directory")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    banks = helpers.load_saved_banks(HERE/"gbs_sampling_data.xlsx", args.sizes, args.graphs)
    tasks = [(n,i,banks[f"Random_n{n}_{i+1}"],args.trials,str(output)) for n in args.sizes for i in range(args.graphs)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_graph, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            name = future.result()
            print(f"{completed}/{len(tasks)} {name}: {time.time()-started:.1f}s", flush=True)
    records, sampling = [], []
    for n,i,*_ in tasks:
        payload = json.loads((output/f"Random_n{n}_{i+1}.json").read_text())
        records.extend(payload["trials"])
        sampling.append(payload["source_summary"])
    frame = pd.DataFrame(records)
    frame.to_csv(output/"trial_records.csv.gz", index=False)
    if args.sizes == SIZES and args.graphs == 60 and args.trials == 10:
        scalar = frame.drop(columns=["Cycle","Longest Path","reinforced_edge_iteration","top_edges_after_reinforcement"], errors="ignore")
        source = scalar[scalar.Algorithm.isin(["SourceResolved-GA","MS-SourceResolved-GA"])]
        main = scalar[~scalar.index.isin(source.index)]
        with pd.ExcelWriter(HERE/"experiment_results.xlsx") as writer:
            main.to_excel(writer, sheet_name="Raw Data", index=False)
        source.to_csv(HERE/"sr_control_results.csv", index=False)
        pd.DataFrame(sampling).to_csv(HERE/"sr_control_sampling.csv", index=False)
        outputs = {p: digest(HERE/p) for p in ["experiment_results.xlsx","sr_control_results.csv","sr_control_sampling.csv"]}
        outputs["replay/trial_records.csv.gz"] = digest(output/"trial_records.csv.gz")
        (output/"completed.json").write_text(json.dumps({"rows":len(frame),"seconds":time.time()-started,"output_sha256":outputs},indent=2))
    print(f"Completed {len(frame)} trial records", flush=True)

if __name__ == "__main__":
    main()
