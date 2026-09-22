"""Analytic ideal-state tails for the 60 graphs in the archived output-loss scan.

For independent two-mode squeezed sources, pair counts are geometric with
parameter 1-tanh(r_i)^2. Passive interferometers preserve total photon number.
Each output mode has a thermal marginal; a union bound controls any mode >=8.
These ideal-state bounds do not certify a historical numerical sampler or bound
conditional postselection error. No photon sampling is performed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from main_benchmark import GraphGenerator

HERE = Path(__file__).resolve().parent

def tail_records():
    records = []
    for n in [20,30]:
        for index in range(30):
            seed = 42 + index*100 + n*10
            graph = GraphGenerator.generate_random_digraph(n, 0.3, seed)
            u, s, vh = np.linalg.svd(graph.to_adjacency_matrix(), full_matrices=False)
            rho = (0.75*s/s.max())**2
            # Coefficients through 15 of product_i (1-rho_i)/(1-rho_i*z).
            distribution = np.array([1.0])
            for value in rho:
                distribution = np.convolve(distribution, (1-value)*value**np.arange(16))[:16]
            pair_means = rho/(1-rho)
            mode_means = np.concatenate([np.abs(u)**2@pair_means, np.abs(vh.T)**2@pair_means])
            records.append({"n":n,"graph_index":index,"graph_seed":seed,
                            "P_total_photons_gt_30":float(1-distribution.sum()),
                            "union_bound_any_mode_ge_8":float(np.sum((mode_means/(1+mode_means))**8))})
    return pd.DataFrame(records)

def run(output_dir=None):
    output = Path(output_dir) if output_dir else HERE/"generated"
    output.mkdir(parents=True,exist_ok=True)
    records = tail_records()
    records.to_csv(output/"loss_tail_checks.csv",index=False)
    report = {"assumptions":"Ideal lossless bipartite Gaussian state; eta=0.75; n=20,30; first 30 graph seeds; total photons=2*total pairs; cutoff 8 excludes occupations >=8",
              "scope":"Ideal-state omitted mass only, not historical backend or conditional sampler validation",
              "graphs":len(records),
              "total_tail_min":float(records.P_total_photons_gt_30.min()),
              "total_tail_max":float(records.P_total_photons_gt_30.max()),
              "mode_union_bound_max":float(records.union_bound_any_mode_ge_8.max())}
    (output/"loss_tail_checks.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return report

if __name__ == "__main__":
    run()
