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

"""Validate the archived records used by the numbered artifact scripts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from artifact_builders import _paired_statistics


HERE = Path(__file__).resolve().parent
PRIMARY_INPUTS = [
    "experiment_results.xlsx",
    "gbs_sampling_data.xlsx",
    "alpha_sensitivity_results.xlsx",
    "sr_control_results.csv",
    "sr_control_sampling.csv",
    "default_check_results.csv",
    "loss_scan_results.csv",
    "matched_control_data/density_dose_banks.csv.gz",
    "matched_control_data/density_dose_results.csv.gz",
    "matched_control_data/density_dose_metadata.json",
    "replay/manifest.json",
    "replay/completed.json",
    "replay/trial_records.csv.gz",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_main_records(report: dict) -> None:
    manifest = json.loads((HERE / "replay/manifest.json").read_text())
    completion = json.loads((HERE / "replay/completed.json").read_text())
    for filename, expected in manifest["source_sha256"].items():
        assert _sha256(HERE / filename) == expected, f"Replay source changed: {filename}"
    assert _sha256(HERE / "gbs_sampling_data.xlsx") == manifest["bank_sha256"]
    for filename, expected in completion["output_sha256"].items():
        assert _sha256(HERE / filename) == expected, f"Replay output changed: {filename}"
    quantum = pd.read_excel(HERE / "experiment_results.xlsx", sheet_name="Raw Data")
    classical = pd.read_csv(HERE / "sr_control_results.csv")
    combined = pd.concat([quantum, classical], ignore_index=True)
    counts = combined.groupby(["Graph Size", "Algorithm", "Graph Name"]).size()
    assert counts.eq(10).all()
    graph_counts = combined.groupby(["Graph Size", "Algorithm"])["Graph Name"].nunique()
    assert graph_counts.eq(60).all()
    assert len(combined) == completion["rows"] == 39600
    assert combined.groupby(["Graph Name", "Trial"])["Trial Seed"].nunique().eq(1).all()
    paired = _paired_statistics()
    assert len(paired) == 48
    assert paired.groupby("contrast").size().eq(6).all()
    report["main_benchmark"] = {
        "quantum_and_original_rows": len(quantum),
        "source_resolved_rows": len(classical),
        "paired_comparisons": int(paired["contrast"].nunique()),
        "provenance": "replay/manifest.json; source, bank, and output hashes verified",
    }


def _validate_sampling_records(report: dict) -> None:
    sampling = pd.read_excel(HERE / "gbs_sampling_data.xlsx", sheet_name="GBS Sampling Data")
    assert sampling["Total Samples"].eq(500).all()
    for serialized, accepted in zip(sampling["All Subgraphs"], sampling["Effective Samples"]):
        assert sum(part.strip().startswith("{") for part in serialized.split(";")) == accepted
    source = pd.read_csv(HERE / "sr_control_sampling.csv")
    assert source["num_shots"].eq(500).all()
    report["sampling_banks"] = {
        "bipartite_gbs_graphs": len(sampling),
        "source_resolved_graphs": len(source),
    }


def _validate_sensitivity_records(report: dict) -> None:
    alpha = pd.read_excel(HERE / "alpha_sensitivity_results.xlsx", sheet_name="Raw Data")
    counts = alpha.groupby(["Density", "Alpha", "Algorithm", "Graph Name"]).size()
    assert counts.eq(5).all()
    defaults = pd.read_csv(HERE / "default_check_results.csv")
    assert defaults.groupby(["n", "graph_id", "setting"]).size().eq(10).all()
    report["sensitivity_records"] = {
        "fitness_weight_rows": len(alpha),
        "unobserved_edge_rows": len(defaults),
    }


def _validate_matched_controls(report: dict) -> None:
    data_dir = HERE / "matched_control_data"
    banks = pd.read_csv(data_dir / "density_dose_banks.csv.gz")
    results = pd.read_csv(data_dir / "density_dose_results.csv.gz")
    assert set(banks["Graph Size"].unique()) == {15, 20, 25, 30, 35, 40}
    assert set(results["Graph Size"].unique()) == {15, 20, 25, 30, 35, 40}
    pooled = banks.groupby("Sampler")["Induced Density"].mean()
    candidates = pooled[[name for name in pooled.index if name.startswith("DensityDose-m")]]
    selected = (candidates - pooled["BipartiteGBS"]).abs().idxmin()
    assert selected == "DensityDose-m4"
    report["matched_controls"] = {
        "bank_rows": len(banks),
        "trial_rows": len(results),
        "density_calibrated_sampler": selected,
    }


def _validate_loss_records(report: dict) -> None:
    loss = pd.read_csv(HERE / "loss_scan_results.csv")
    assert len(loss) == 960
    keys = ["n", "graph_id", "tau", "sampler", "schedule"]
    assert len(loss.drop_duplicates(keys)) == 960
    assert loss.groupby(["n", "tau", "sampler", "schedule"]).size().eq(30).all()

    tree = ast.parse((HERE / "output_loss_scan.py").read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_cov"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "output_loss_scan.py", "exec"), namespace)

    class Graph:
        def __init__(self, n: int, seed: int):
            self.n = n
            rng = random.Random(seed)
            self.matrix = np.array(
                [[float(i != j and rng.random() < 0.3) for j in range(n)] for i in range(n)]
            )

        def to_adjacency_matrix(self) -> np.ndarray:
            return self.matrix

    maximum_error = 0.0
    for n in [20, 30]:
        for index in range(30):
            covariance = namespace["build_cov"](Graph(n, 42 + index * 100 + n * 10))
            modes = 2 * n
            identity = np.eye(modes)
            zero = np.zeros((modes, modes))
            omega = np.block([[zero, identity], [-identity, zero]])
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(covariance @ omega @ covariance - omega))),
            )
            assert np.allclose(covariance, covariance.T)
            assert np.linalg.eigvalsh(covariance).min() > 0
    assert maximum_error < 1e-10
    report["loss_scan"] = {"rows": len(loss), "covariance_purity_max_error": maximum_error}


def _validate_code(report: dict) -> None:
    for number in range(1, 10):
        assert (HERE / f"Figure-{number:02d}.py").is_file()
    for number in range(1, 9):
        assert (HERE / f"Table-{number:02d}.py").is_file()
    for number in [1, 2, 3, 8, 9]:
        assert (HERE / "static" / f"Figure{number:02d}-source.pdf").is_file()

    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"))

    for filename in ["main_benchmark.py", "fitness_weight_scan.py"]:
        tree = ast.parse((HERE / filename).read_text(encoding="utf-8-sig"))
        function = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_real_gbs_sampling"
        )
        function.returns = None
        for argument in function.args.args:
            argument.annotation = None
        namespace = {"SF_AVAILABLE": False}
        exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
        try:
            namespace["_real_gbs_sampling"](None, 500)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{filename} did not reject a missing quantum backend")
    report["code"] = "numbered entry points, syntax, and missing-backend checks passed"


def validate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative in PRIMARY_INPUTS:
        if not (HERE / relative).is_file():
            raise FileNotFoundError(f"Missing required input: {relative}")
    report = {
        "input_sha256": {relative: _sha256(HERE / relative) for relative in PRIMARY_INPUTS}
    }
    _validate_main_records(report)
    _validate_sampling_records(report)
    _validate_sensitivity_records(report)
    _validate_matched_controls(report)
    _validate_loss_records(report)
    _validate_code(report)

    versions = {}
    for package in ["numpy", "pandas", "scipy", "matplotlib", "openpyxl"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    environment = {
        "purpose": "Saved-data validation environment",
        "packages": versions,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "validation-environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    print("Saved data and public artifact scripts passed validation.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    args = parser.parse_args()
    validate(args.output_dir)


if __name__ == "__main__":
    main()
