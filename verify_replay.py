"""Verify archived search certificates and execute selected independent replays.

Run after reproduce_all.py. This checks all replay cycles and the stored source
acceptance decisions, then reruns three algorithms on one graph at each size and
checks four matched-control trials per size. It does not certify the historical
quantum sampling distribution.
"""
from pathlib import Path
import json
import hashlib
import platform
import pickle
import random
import numpy as np
import pandas as pd
import main_benchmark as m
import matched_control_utils as h
import matched_density_controls as md
from replay_benchmark import replay_guidance

HERE=Path(__file__).resolve().parent

def audit_records():
    """Connect each plotted outcome to its checkpoint and frozen input bank."""
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((HERE/"replay/manifest.json").read_text())
    completion = json.loads((HERE/"replay/completed.json").read_text())
    for filename, expected in manifest["source_sha256"].items():
        assert digest(HERE/filename) == expected, filename
    assert digest(HERE/"gbs_sampling_data.xlsx") == manifest["bank_sha256"]
    for filename, expected in completion["output_sha256"].items():
        assert digest(HERE/filename) == expected, filename
    rows = []
    bank_ids = {}
    banks = h.load_saved_banks(HERE/"gbs_sampling_data.xlsx", h.DEFAULT_SIZES, 60)
    for path in sorted((HERE/"replay").glob("Random_n*.json")):
        payload = json.loads(path.read_text())
        name = payload["graph"]
        n = int(name.split("_")[1][1:])
        index = int(name.split("_")[2])-1
        assert payload["graph_seed"] == h.graph_seed(n, index)
        graph = m.GraphGenerator.generate_random_digraph(n, .3, payload["graph_seed"])
        for key, bank in [("quantum", banks[name]), ("source", payload["source_subsets"])]:
            info = replay_guidance(graph, bank)
            assert [list(e) for e in info.top_edges] == payload[key+"_top_edges"]
            canonical = json.dumps([sorted(s) for s in bank], separators=(",", ":"))
            bank_ids[name+"/"+key] = hashlib.sha256(canonical.encode()).hexdigest()
        for row in payload["trials"]:
            assert row["Graph Name"] == name
            assert row["Trial Seed"] == 20260922+n*1000000+index*1000+row["Trial"]
            if "reinforced_edge_iteration" in row:
                key = "source" if row["Algorithm"] == "MS-SourceResolved-GA" else "quantum"
                order = [tuple(e) for e in payload[key+"_top_edges"]]
                for edge in row["reinforced_edge_iteration"]:
                    edge = tuple(edge)
                    if edge not in order:
                        order.insert(0, edge)
                assert [list(e) for e in order] == row["top_edges_after_reinforcement"]
        rows.extend(payload["trials"])
    keys = ["Graph Name", "Algorithm", "Trial"]
    expected = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    assert len(expected) == 39600 and not expected.duplicated(keys).any()
    assert expected.groupby(keys[:2]).Trial.apply(lambda s: set(s) == set(range(10))).all()
    archives = [pd.read_csv(HERE/"replay/trial_records.csv.gz"), pd.concat([
        pd.read_excel(HERE/"experiment_results.xlsx", sheet_name="Raw Data"),
        pd.read_csv(HERE/"sr_control_results.csv")], ignore_index=True)]
    for frame in archives:
        frame = frame.sort_values(keys).reset_index(drop=True)
        for column in keys+["Graph Seed", "Trial Seed", "Success", "Generations", "Path Length", "Longest Path Length"]:
            assert expected[column].equals(frame[column]), column
    report = {"python": platform.python_version(), "rows_cross_checked": len(expected),
              "frozen_source_and_output_hashes_verified": True,
              "graph_seed_and_trial_seed_formulas_verified": True,
              "stored_reinforcement_orders_verified": True,
              "ordered_bank_sha256": bank_ids}
    (HERE/"generated/provenance_audit.json").write_text(json.dumps(report, indent=2))
    return report


def run():
    audit_records()
    banks=h.load_saved_banks(HERE/"gbs_sampling_data.xlsx",h.DEFAULT_SIZES,60)
    certificates=0
    selected=0
    paths=sorted((HERE/"replay").glob("Random_n*.json"))
    assert {p.stem for p in paths}=={f"Random_n{n}_{i}" for n in h.DEFAULT_SIZES for i in range(1,61)}
    for path in paths:
        payload=json.loads(path.read_text())
        n=int(payload["graph"].split("_")[1][1:])
        graph=m.GraphGenerator.generate_random_digraph(n,.3,payload["graph_seed"])
        graph.name=payload["graph"]
        assert [list(e) for e in graph.edges]==payload["edges"]
        raw=np.load(path.with_name(path.stem+"-source-raw.npz"))["counts"]
        accepted=[]
        for up,lo in raw:
            if up.max()<=1 and lo.max()<=1 and up.sum()==lo.sum() and up.sum()>=3:
                accepted.append(sorted(set(np.flatnonzero(up))|set(np.flatnonzero(lo))))
        assert accepted==payload["source_subsets"]
        for row in payload["trials"]:
            if row["Success"]:
                c=row["Cycle"]
                assert len(c)==n and set(c)==set(range(n))
                assert all(graph.has_edge(c[j],c[(j+1)%n]) for j in range(n))
                certificates+=1
            path_vertices=row["Longest Path"]
            assert len(path_vertices)==row["Longest Path Length"]
            assert len(set(path_vertices))==len(path_vertices)
            assert all(graph.has_edge(u,v) for u,v in zip(path_vertices,path_vertices[1:]))
        if not graph.name.endswith("_1"):
            continue
        quantum=replay_guidance(graph,banks[graph.name])
        source=replay_guidance(graph,payload["source_subsets"])
        for name,algorithm,info in [
            ("GBS-Genetic",m.GBSEnhancedGenetic(pop_size=100,max_generations=200,alpha=.1),quantum),
            ("MultiStageGBS",m.MultiStageGBSEnhancedGenetic(pop_size=100,max_generations=300),quantum),
            ("MS-SourceResolved-GA",m.MultiStageGBSEnhancedGenetic(pop_size=100,max_generations=300),source),
        ]:
            row=next(r for r in payload["trials"] if r["Algorithm"]==name and r["Trial"]==0)
            random.seed(row["Trial Seed"]); np.random.seed(row["Trial Seed"])
            result=algorithm.search(graph,info)
            assert (int(result.success),result.generations,result.path_length,result.cycle,result.longest_path)==(row["Success"],row["Generations"],row["Path Length"],row["Cycle"],row["Longest Path"]), (graph.name,name,result,row)
            selected+=1
    config=json.loads((HERE/"matched_control_data/density_dose_metadata.json").read_text())["config"]
    trials=pd.read_csv(HERE/"matched_control_data/density_dose_results.csv.gz")
    checks=0
    for n in h.DEFAULT_SIZES:
        name=f"Random_n{n}_1"
        graph=m.GraphGenerator.generate_random_digraph(n,.3,h.graph_seed(n,0));graph.name=name
        pilot_config={**config,"bank_replicates":1,"trials":1,"doses":[1,4]}
        # The archived matched runner sends tasks through multiprocessing.Pool.
        # Mirror that serialization step, including its set reconstruction.
        task=pickle.loads(pickle.dumps((n,0,banks[name],pilot_config)))
        new_trials,new_banks=md.run_one_graph(task)
        for new in new_trials:
            row=trials[(trials["Graph Name"]==name)&(trials.Algorithm==new["Algorithm"])&(trials["Bank Replicate"]==0)&(trials.Trial==0)].iloc[0]
            assert np.isclose(new["Initial Mean Fitness"],row["Initial Mean Fitness"],atol=1e-14,rtol=0), (n,new["Algorithm"],new["Initial Mean Fitness"],row["Initial Mean Fitness"])
            assert (new["Success"],new["Generations"],new["Longest Path Length"])==(row.Success,row.Generations,row["Longest Path Length"]), (n,new["Algorithm"])
            checks+=1
    stats=pd.read_csv(HERE/"generated/matched_paired_statistics.csv")
    value=stats[(stats.Contrast=="BipartiteGBS-InitOnly vs Uniform-Matched-InitOnly")&(stats["Graph Size"]==35)]["Wilcoxon p Holm"].item()
    assert np.isclose(value,.10183988975169551,atol=1e-12,rtol=0)
    # Reinforcement intentionally preserves existing order, even for a new maximum.
    from types import SimpleNamespace
    edges=[(0,1),(0,2),(0,3),(1,0),(1,2),(1,3)]
    info=SimpleNamespace(edge_prob=dict(zip(edges,[.6,.5,.4,.3,.2,.1])),top_edges=edges[:])
    merged=m.MultiStageGBSEnhancedGenetic(100,300)._merge_gbs_info(info,{edges[-1]})
    assert merged.top_edges==edges and merged.edge_prob[edges[-1]]==.8 and info.edge_prob[edges[-1]]==.1
    report={"validated_hamiltonian_cycles":certificates,"source_banks_checked":360,"fresh_replay_trials":selected,"fresh_matched_trials":checks,"matched_tie_correction_p":value,"reinforcement_order":"stored-list behavior verified","scope":"Full saved-record certificates plus selected fresh trials; no certification of historical quantum sampling"}
    (HERE/"generated/replay_verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    run()
