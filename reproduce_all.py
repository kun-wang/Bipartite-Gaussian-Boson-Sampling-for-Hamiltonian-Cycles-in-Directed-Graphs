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

"""Rebuild all numbered manuscript artifacts and validate the saved data."""

from pathlib import Path

from artifact_builders import build_figure, build_table
from validate_saved_data import validate
from loss_tail_check import run as check_loss_tails
import json
import pandas as pd
from artifact_builders import _paired_statistics
from matched_density_controls import paired_statistics, initial_metric_statistics


HERE = Path(__file__).resolve().parent


def main() -> None:
    output = HERE / "generated"
    output.mkdir(exist_ok=True)
    validate(output)
    check_loss_tails(output)
    for number in range(1, 10):
        print(build_figure(number, output))
    for number in range(1, 9):
        print(build_table(number, output))
    _paired_statistics().to_csv(output / "main_paired_statistics.csv", index=False)
    control_dir = HERE / "matched_control_data"
    records = pd.read_csv(control_dir / "density_dose_results.csv.gz")
    metadata = json.loads((control_dir / "density_dose_metadata.json").read_text())
    config = metadata["config"]
    paired_statistics(records, config["sizes"], config["doses"], metadata["selected_calibrated_dose"], config["bootstrap_resamples"]).to_csv(output / "matched_paired_statistics.csv", index=False)
    initial_metric_statistics(records, config["sizes"], metadata["selected_calibrated_dose"], config["bootstrap_resamples"]).to_csv(output / "matched_initial_statistics.csv", index=False)
    print(f"All artifacts are available in {output}")


if __name__ == "__main__":
    main()
