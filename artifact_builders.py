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
"""Build the numbered figures and tables accompanying the manuscript.

The public entry points are the ``Figure-XX.py`` and ``Table-XX.py`` scripts.
This module keeps their data loading, validation, and formatting in one place.
"""

from __future__ import annotations

import argparse
import math
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parent
STATIC_DIR = REPO_ROOT / "static"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "generated"
SIZES = [15, 20, 25, 30, 35, 40]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "pdf.fonttype": 42})


def _output_dir(path: str | Path | None) -> Path:
    output = Path(path) if path else DEFAULT_OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    return output


def _main_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    quantum = pd.read_excel(REPO_ROOT / "experiment_results.xlsx", sheet_name="Raw Data")
    classical = pd.read_csv(REPO_ROOT / "sr_control_results.csv")
    combined = pd.concat([quantum, classical], ignore_index=True)
    assert combined.groupby(["Graph Size", "Algorithm", "Graph Name"]).size().eq(10).all()
    assert combined.groupby(["Graph Size", "Algorithm"])["Graph Name"].nunique().eq(60).all()
    return quantum, combined


def _bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    """Return the manuscript's paired graph bootstrap interval."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20_260_919)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    return tuple(np.percentile(values[indices].mean(axis=1), [2.5, 97.5]))


def _holm_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = np.minimum(
        1.0,
        np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)),
    )
    return adjusted


def _paired_records(
    data: pd.DataFrame,
    contrasts: list[tuple[str, str, str]],
    sizes: list[int],
    trials: int,
) -> pd.DataFrame:
    records: list[dict] = []
    for label, first, second in contrasts:
        contrast_rows: list[dict] = []
        raw_p_values: list[float] = []
        for size in sizes:
            selected = data[data["Graph Size"] == size]
            pivot = selected.groupby(["Graph Name", "Algorithm"])["Success"].sum().unstack()
            if first not in pivot or second not in pivot:
                raise ValueError(f"Missing data for {label} at n={size}")
            paired = pivot[[first, second]].dropna()
            count_differences = (paired[first] - paired[second]).to_numpy(dtype=int)
            differences = count_differences * (100.0 / trials)
            low, high = _bootstrap_interval(differences)
            p_value = (
                1.0
                if np.all(differences == 0)
                else float(stats.wilcoxon(count_differences, zero_method="wilcox").pvalue)
            )
            raw_p_values.append(p_value)
            contrast_rows.append(
                {
                    "contrast": label,
                    "n": size,
                    "n_graphs": len(paired),
                    "algo1_rate%": paired[first].mean() * (100.0 / trials),
                    "algo2_rate%": paired[second].mean() * (100.0 / trials),
                    "mean_diff_pp": differences.mean(),
                    "CI95_lo": low,
                    "CI95_hi": high,
                    "wilcoxon_p_raw": p_value,
                }
            )
        adjusted = _holm_adjust(raw_p_values)
        for row, p_value in zip(contrast_rows, adjusted):
            row["wilcoxon_p_holm"] = p_value
        records.extend(contrast_rows)
    return pd.DataFrame(records)


def _summarize(frame: pd.DataFrame, algorithm: str, size: int) -> tuple[float, float, float, float]:
    rows = frame[(frame["Graph Size"] == size) & (frame["Algorithm"] == algorithm)]
    success = rows.groupby("Graph Name")["Success"].mean() * 100
    failed_path = rows[rows["Success"] == 0].groupby("Graph Name")["Longest Path Length"].mean()
    return success.mean(), success.sem(), failed_path.mean(), failed_path.sem()


def _save(fig: plt.Figure, number: int, output: Path) -> Path:
    target = output / f"Figure{number:02d}.pdf"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def _stage_static_figure(number: int, output: Path) -> Path:
    source = STATIC_DIR / f"Figure{number:02d}-source.pdf"
    if not source.exists():
        raise FileNotFoundError(f"Missing canonical source: {source}")
    target = output / f"Figure{number:02d}.pdf"
    shutil.copy2(source, target)
    return target


def build_figure(number: int, output_dir: str | Path | None = None) -> Path:
    """Build one manuscript figure by its displayed figure number."""
    output = _output_dir(output_dir)
    if number in {1, 2, 3, 8, 9}:
        return _stage_static_figure(number, output)
    if number == 4:
        _, data = _main_records()
        algorithms = [
            "GeneticAlgorithm", "Density-Biased-GA", "SourceResolved-GA",
            "GBS-InitOnly", "GBS-Genetic", "MultiStageGBS", "MS-SourceResolved-GA",
        ]
        labels = ["GA", "Density", "Source", "InitOnly", "BGBS", "MS-BGBS", "MS-Source"]
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.2), layout="constrained")
        for axis, size in zip(axes.flat, SIZES):
            values = np.array([_summarize(data, algorithm, size) for algorithm in algorithms])
            x = np.arange(len(algorithms))
            right = axis.twinx()
            axis.bar(x - 0.19, values[:, 0], 0.36, yerr=values[:, 1], capsize=2,
                     color="#377eb8", label="Success")
            right.bar(x + 0.19, values[:, 2], 0.36, yerr=values[:, 3], capsize=2,
                      color="#ed9570", label="Failed-run path")
            axis.set(ylim=(0, 105), title=f"n = {size}", ylabel="Success (%)")
            right.set(ylim=(0, size * 1.05), ylabel="Path length")
            axis.set_xticks(x, labels, rotation=55, ha="right")
            axis.tick_params(axis="x", labelsize=9)
            axis.yaxis.label.set_color("#377eb8")
            right.yaxis.label.set_color("#bd5c37")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        return _save(fig, number, output)
    if number == 5:
        quantum, _ = _main_records()
        algorithms = [
            "GeneticAlgorithm", "Density-Biased-GA", "GBS-InitOnly", "GBS-MutationOnly",
            "GBS-FitnessOnly", "GBS-NoSubgraph", "GBS-SubgraphOnly", "GBS-Genetic",
            "MultiStageGBS",
        ]
        labels = ["GA", "Density", "InitOnly", "Mutation", "Fitness", "EdgeOnly",
                  "Subgraph", "BGBS", "MS-BGBS"]
        colors = ["#999999", "#a6d854", "#377eb8", "#e78ac3", "#8da0cb", "#e6c637",
                  "#1b9e77", "#d95f02", "#e41a1c"]
        fig, axes = plt.subplots(2, 3, figsize=(11, 7.6), layout="constrained")
        for axis, size in zip(axes.flat, SIZES):
            values = np.array([_summarize(quantum, algorithm, size) for algorithm in algorithms])
            x = np.arange(len(algorithms))
            axis.bar(x, values[:, 0], yerr=values[:, 1], capsize=2, color=colors)
            axis.set(ylim=(0, 105), title=f"n = {size}", ylabel="Success (%)")
            axis.set_xticks(x, labels, rotation=55, ha="right")
            axis.tick_params(axis="x", labelsize=9)
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        return _save(fig, number, output)
    if number == 6:
        data = pd.read_excel(REPO_ROOT / "alpha_sensitivity_results.xlsx", sheet_name="Raw Data")
        assert data.groupby(["Density", "Alpha", "Algorithm", "Graph Name"]).size().eq(5).all()
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
        for axis, density in zip(axes, sorted(data["Density"].unique())):
            for algorithm, label, color in [
                ("GBS-Genetic", "BipartiteGBS-GA", "#377eb8"),
                ("GeneticAlgorithm", "GA", "#666666"),
            ]:
                selected = data[(data["Density"] == density) & (data["Algorithm"] == algorithm)]
                grouped = selected.groupby(["Alpha", "Graph Name"])["Success"].mean().mul(100)
                summary = grouped.groupby(level=0).agg(["mean", "sem"])
                axis.plot(summary.index, summary["mean"], "o-", ms=3, color=color, label=label)
                axis.fill_between(summary.index, summary["mean"] - summary["sem"],
                                  summary["mean"] + summary["sem"], alpha=0.16, color=color)
            axis.set(title=f"p = {density:.2f}", xlabel=r"$\alpha_{\max}$",
                     ylabel="Success (%)", ylim=(0, 103))
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, loc="lower right")
        return _save(fig, number, output)
    if number == 7:
        data = pd.read_csv(REPO_ROOT / "loss_scan_results.csv")
        assert len(data) == 960
        transmissions = [1.0, 0.95, 0.90, 0.80]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        axis = axes[0]
        for sampler, color, label in [
            ("quantum", "#1f77b4", "BipartiteGBS (quantum)"),
            ("classical", "#d62728", "source-resolved (classical)"),
        ]:
            selected = data[data["sampler"] == sampler]
            acceptance = [selected[selected["tau"] == tau]["acceptance_rate%"].mean()
                          for tau in transmissions]
            axis.plot(transmissions, acceptance, "o-", color=color, label=label)
        axis.set(xlabel=r"Transmission $\tau$", ylabel="Acceptance rate (%)",
                 title="(a) Fraction of accepted samples")
        axis.legend(fontsize=8, frameon=False)
        axis.grid(alpha=0.3)
        axis = axes[1]
        methods = [
            ("quantum", "single", "BipartiteGBS-GA", "#1f77b4", "-"),
            ("classical", "single", "SourceResolved-GA", "#d62728", "-"),
            ("quantum", "multi", "MS-BipartiteGBS-GA", "#1f77b4", "--"),
            ("classical", "multi", "MS-SourceResolved-GA", "#d62728", "--"),
        ]
        for sampler, schedule, label, color, style in methods:
            selected = data[(data["sampler"] == sampler) & (data["schedule"] == schedule)]
            success = [selected[selected["tau"] == tau]["success_rate%"].mean()
                       for tau in transmissions]
            axis.plot(transmissions, success, "o" + style, color=color, label=label)
        axis.set(xlabel=r"Transmission $\tau$", ylabel="Success rate (%)",
                 title="(b) Search success rate", ylim=(50, 100))
        axis.legend(fontsize=8, frameon=False)
        axis.grid(alpha=0.3)
        fig.tight_layout()
        return _save(fig, number, output)
    raise ValueError(f"Unknown figure number: {number}")


def _format_number(value: float, digits: int = 2, signed: bool = False) -> str:
    rounded = Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    prefix = "+" if signed and rounded > 0 else ""
    return f"{prefix}{rounded:.{digits}f}"


def _write_table(number: int, lines: list[str], output: Path) -> Path:
    target = output / f"Table{number:02d}.tex"
    target.write_text("%% Generated from the archived data by Table-%02d.py.\n" % number
                      + "\n".join(lines) + "\n", encoding="utf-8")
    return target


def _table_1() -> list[str]:
    rows = []
    tanh_r = 0.75
    for pairs in [4, 8, 10, 20, 40, 100]:
        modes = pairs * pairs
        log_q = (
            math.lgamma(modes + pairs) - math.lgamma(pairs + 1) - math.lgamma(modes)
            + pairs * math.log(tanh_r ** 2) + modes * math.log(1 - tanh_r ** 2)
        ) / math.log(10)
        rows.append(f"${pairs}$ & ${modes}$ & ${-log_q:.2f}$ \\\\")
    return rows


def _table_2() -> list[str]:
    samples = pd.read_excel(REPO_ROOT / "gbs_sampling_data.xlsx", sheet_name="GBS Sampling Data")
    data = (
        samples.groupby("Graph Size")["Effective Samples"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "Graph Size": "n",
                "count": "graphs",
                "mean": "mean_accepted",
                "std": "sd_accepted",
                "min": "min_accepted",
                "max": "max_accepted",
            }
        )
    )
    data["mean_rate_%"] = (data["mean_accepted"] / 5.0).round(2)
    data["sd_rate_%"] = (data["sd_accepted"] / 5.0).round(2)
    lines = []
    for _, row in data.iterrows():
        lines.append(
            f"${int(row['n'])}$ & ${_format_number(row['mean_accepted'], 1)}$ & "
            f"${_format_number(row['sd_accepted'], 1)}$ & "
            f"${int(row['min_accepted'])}$--${int(row['max_accepted'])}$ & "
            f"${row['mean_rate_%']:.2f}\\%$ & ${row['sd_rate_%']:.2f}\\%$ \\\\"
        )
    return lines


CONTRAST_MACROS = {
    "BipartiteGBS-GA vs GA": r"\BGBSabbr{} vs.\ \GAabbr{}",
    "BipartiteGBS-InitOnly vs GA": r"\InitOnly{} vs.\ \GAabbr{}",
    "BipartiteGBS-InitOnly vs BipartiteGBS-GA": r"\InitOnly{} vs.\ \BGBSabbr{}",
    "MS-BipartiteGBS-GA vs BipartiteGBS-InitOnly": r"\MSBGBSabbr{} vs.\ \InitOnly{}",
    "MS-BipartiteGBS-GA vs BipartiteGBS-GA": r"\MSBGBSabbr{} vs.\ \BGBSabbr{}",
    "BipartiteGBS-GA vs SourceResolved-GA": r"\BGBSabbr{} vs.\ \Sourceabbr{}",
    "MS-BipartiteGBS-GA vs MS-SourceResolved-GA": r"\MSBGBSabbr{} vs.\ \MSSourceabbr{}",
    "MS-SourceResolved-GA vs SourceResolved-GA": r"\MSSourceabbr{} vs.\ \Sourceabbr{}",
}


def _paired_statistics() -> pd.DataFrame:
    """Return the eight comparisons used in Tables III and VIII."""
    quantum, combined = _main_records()
    main_contrasts = [
        ("BipartiteGBS-GA vs GA", "GBS-Genetic", "GeneticAlgorithm"),
        ("BipartiteGBS-InitOnly vs GA", "GBS-InitOnly", "GeneticAlgorithm"),
        ("BipartiteGBS-InitOnly vs BipartiteGBS-GA", "GBS-InitOnly", "GBS-Genetic"),
        ("MS-BipartiteGBS-GA vs BipartiteGBS-InitOnly", "MultiStageGBS", "GBS-InitOnly"),
        ("MS-BipartiteGBS-GA vs BipartiteGBS-GA", "MultiStageGBS", "GBS-Genetic"),
    ]
    source_contrasts = [
        ("BipartiteGBS-GA vs SourceResolved-GA", "GBS-Genetic", "SourceResolved-GA"),
        ("MS-BipartiteGBS-GA vs MS-SourceResolved-GA", "MultiStageGBS", "MS-SourceResolved-GA"),
        ("MS-SourceResolved-GA vs SourceResolved-GA", "MS-SourceResolved-GA", "SourceResolved-GA"),
    ]
    data = pd.concat(
        [
            _paired_records(quantum, main_contrasts, SIZES, trials=10),
            _paired_records(combined, source_contrasts, SIZES, trials=10),
        ],
        ignore_index=True,
    )
    assert data.groupby("contrast").size().eq(6).all()
    return data


def _table_3() -> list[str]:
    data = _paired_statistics()
    lines = []
    for contrast in CONTRAST_MACROS:
        selected = data[data["contrast"] == contrast].set_index("n").loc[SIZES]
        cells = []
        for row in selected.itertuples():
            value = _format_number(row.mean_diff_pp, signed=True)
            cells.append(r"$\mathbf{" + value + "}$" if row.wilcoxon_p_holm < 0.05 else f"${value}$")
        lines.append(CONTRAST_MACROS[contrast] + " & " + " & ".join(cells) + r" \\")
    return lines


def _table_4() -> list[str]:
    return [
        r"\GAabbr{} & Random & Standard & Random & -- \\",
        r"\InitOnly{} & BipartiteGBS & Standard & Random & Subgraph + Edge \\",
        r"\FitnessOnly{} & Random & BipartiteGBS & Random & Edge only \\",
        r"\MutationOnly{} & Random & Standard & BipartiteGBS & Edge only \\",
        r"\SubgraphOnly{} & BipartiteGBS & Standard & Random & Subgraph only \\",
        r"\EdgeOnly{} & BipartiteGBS & Standard & Random & Edge only \\",
        r"\BipartiteGBSGA{} & BipartiteGBS & BipartiteGBS & BipartiteGBS & Subgraph + Edge \\",
        r"\MSBipartiteGBSGA{} & \multicolumn{4}{c}{Adaptive, two stages} \\",
    ]


def _table_5() -> list[str]:
    control_dir = REPO_ROOT / "matched_control_data"
    banks = pd.read_csv(control_dir / "density_dose_banks.csv.gz")
    results = pd.read_csv(control_dir / "density_dose_results.csv.gz")
    per_graph_bank = (
        banks.groupby(["Graph Size", "Graph Name", "Sampler"], as_index=False)
        .agg(Mean_Bank_Density=("Induced Density", "mean"))
    )
    bank = (
        per_graph_bank.groupby(["Graph Size", "Sampler"], as_index=False)
        .agg(Mean_Bank_Density=("Mean_Bank_Density", "mean"))
    )
    per_graph_results = (
        results.groupby(["Graph Size", "Graph Name", "Algorithm"], as_index=False)
        .agg(
            Mean_Success_Rate=("Success", "mean"),
            Mean_Initial_Fitness=("Initial Mean Fitness", "mean"),
        )
    )
    summary = (
        per_graph_results.groupby(["Graph Size", "Algorithm"], as_index=False)
        .agg(
            Mean_Success_Rate=("Mean_Success_Rate", "mean"),
            Mean_Initial_Fitness=("Mean_Initial_Fitness", "mean"),
        )
    )
    summary["Mean_Success_Rate (%)"] = 100.0 * summary["Mean_Success_Rate"]
    bank_names = [
        ("Uniform-Matched", r"\UniformMatchedInitOnly{}"),
        ("DensityDose-m4", r"\DensityCalibratedInitOnly{} ($m=4$)"),
        ("BipartiteGBS", r"\BipartiteGBSInitOnly{}"),
    ]
    algorithm_names = [
        ("Uniform-Matched-InitOnly", r"\UniformMatchedInitOnly{}"),
        ("DensityDose-m4-InitOnly", r"\DensityCalibratedInitOnly{} ($m=4$)"),
        ("BipartiteGBS-InitOnly", r"\BipartiteGBSInitOnly{}"),
    ]
    lines = [r"\multicolumn{7}{@{}l}{\textit{Mean induced edge density of the subset bank}} \\"]
    for name, macro in bank_names:
        values = bank[bank["Sampler"] == name].set_index("Graph Size").loc[SIZES, "Mean_Bank_Density"]
        lines.append(macro + " & " + " & ".join(f"{value:.3f}" for value in values) + r" \\")
    lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\textit{Mean valid edge fraction of the initial population}} \\"]
    for name, macro in algorithm_names:
        values = summary[summary["Algorithm"] == name].set_index("Graph Size").loc[SIZES, "Mean_Initial_Fitness"]
        rendered = [_format_number(value, 3) for value in values]
        lines.append(macro + " & " + " & ".join(rendered) + r" \\")
    lines += [r"\midrule", r"\multicolumn{7}{@{}l}{\textit{Final success rate (\%)}} \\"]
    final_names = [("Standard-GA", r"\StandardGA{}"), *algorithm_names]
    for name, macro in final_names:
        values = summary[summary["Algorithm"] == name].set_index("Graph Size").loc[SIZES, "Mean_Success_Rate (%)"]
        lines.append(macro + " & " + " & ".join(f"{value:.2f}" for value in values) + r" \\")
    return lines


def _table_6() -> list[str]:
    records = pd.read_excel(REPO_ROOT / "alpha_sensitivity_results.xlsx", sheet_name="Raw Data")
    records = records[np.isclose(records["Alpha"], 0.1)]
    contrasts = [
        ("BipartiteGBS-GA vs GA", "GBS-Genetic", "GeneticAlgorithm"),
    ]
    scan_records = records.drop(columns=["Graph Size"]).rename(columns={"Density": "Graph Size"})
    data = _paired_records(
        scan_records,
        contrasts,
        sorted(records["Density"].unique().tolist()),
        trials=5,
    ).rename(
        columns={
            "n": "density",
            "algo1_rate%": "GBS_rate%",
            "algo2_rate%": "GA_rate%",
        }
    )
    lines = []
    for _, row in data.iterrows():
        lines.append(
            f"${row['density']:.2f}$ & ${row['GBS_rate%']:.1f}$ & ${row['GA_rate%']:.1f}$ & "
            f"${_format_number(row['mean_diff_pp'], 1, signed=True)}$ & "
            f"$[{row['CI95_lo']:.1f},\\,{row['CI95_hi']:.1f}]$ & "
            f"\\makecell{{${row['wilcoxon_p_raw']:.4f}$\\\\$({row['wilcoxon_p_holm']:.3f})$}} \\\\"
        )
    return lines


def _table_7() -> list[str]:
    return [
        r"Population size & $N_{\mathrm{pop}}$ & 100 \\",
        r"Maximum generations & $G_{\max}$ & 200 \\",
        r"Crossover rate & $p_c$ & 0.8 \\",
        r"Mutation rate & $p_m$ & 0.1 \\",
        r"Tournament size & $k$ & 3 \\",
        r"BipartiteGBS hybrid parameter & $\beta$ & 0.2 \\",
        r"Fitness weight (max) & $\alpha_{\max}$ & 0.1 \\",
        r"Squeezing scaling factor & $\eta$ & 0.75 \\",
        r"Number of BipartiteGBS samples & $N_{\mathrm{BipartiteGBS}}$ & 500 \\",
    ]


def _scientific_tex(value: float) -> str:
    if value == 0:
        return "0"
    if 0.001 <= abs(value) < 1000:
        return f"{value:.3g}"
    exponent = math.floor(math.log10(abs(value)))
    coefficient = value / 10 ** exponent
    return rf"{coefficient:.3g}\times10^{{{exponent}}}"


def _table_8() -> list[str]:
    data = _paired_statistics()
    lines = []
    for size in SIZES:
        lines += [rf"\multicolumn{{5}}{{c}}{{\textbf{{Graph size $n={size}$}}}}\\*", r"\midrule"]
        selected = data[data["n"] == size]
        for contrast in CONTRAST_MACROS:
            row = selected[selected["contrast"] == contrast].iloc[0]
            lines.append(
                CONTRAST_MACROS[contrast] + " & "
                + f"${_format_number(row['mean_diff_pp'], signed=True)}$ & "
                + f"$[{row['CI95_lo']:.2f},\\,{row['CI95_hi']:.2f}]$ & "
                + f"${_scientific_tex(row['wilcoxon_p_raw'])}$ & "
                + f"${_scientific_tex(row['wilcoxon_p_holm'])}$ \\\\*"
            )
        if size != SIZES[-1]:
            lines.append(r"\midrule")
    return lines


TABLE_BUILDERS = {
    1: _table_1, 2: _table_2, 3: _table_3, 4: _table_4,
    5: _table_5, 6: _table_6, 7: _table_7, 8: _table_8,
}


def build_table(number: int, output_dir: str | Path | None = None) -> Path:
    """Build one manuscript table body by its displayed table number."""
    output = _output_dir(output_dir)
    builder = TABLE_BUILDERS.get(number)
    if builder is None:
        raise ValueError(f"Unknown table number: {number}")
    lines = builder()
    return _write_table(number, lines, output)


def cli(kind: str, number: int) -> None:
    parser = argparse.ArgumentParser(description=f"Build manuscript {kind} {number}.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Destination directory (default: generated/ in the repository root).")
    args = parser.parse_args()
    target = build_figure(number, args.output_dir) if kind == "figure" else build_table(number, args.output_dir)
    print(target)
