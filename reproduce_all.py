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


HERE = Path(__file__).resolve().parent


def main() -> None:
    output = HERE / "generated"
    output.mkdir(exist_ok=True)
    validate(output)
    for number in range(1, 10):
        print(build_figure(number, output))
    for number in range(1, 9):
        print(build_table(number, output))
    print(f"All artifacts are available in {output}")


if __name__ == "__main__":
    main()
