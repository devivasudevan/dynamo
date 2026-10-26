# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a DGDR's native AISimulate input through the public Sweeper API.

DGDR stores ``SmartSearchConfig`` as its native AISimulate compatibility
boundary. The unified ``aisimulate recommend`` CLI accepts a different,
higher-level configuration, so using that CLI would require translating and
therefore constraining the opaque input overrides. This module keeps the native
input intact and connects the public Sweeper and output-adapter APIs directly.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml
from aisimulate.output_adapter import (
    RecommendationOutputContext,
    resolve_output_adapters,
    resolve_output_callbacks,
    write_output_adapters,
)
from aisimulate.sweeper import SmartSearchConfig, Sweeper

from dynamo.aisimulate.output.dgdr_run.adapter import DGDRRunOutputConfig
from dynamo.replay.simulation import DynamoReplayRunnerFactory

_OUTPUT_NAME = "dgdr_run"


def _load_input(path: Path) -> tuple[SmartSearchConfig, dict[str, Any], int]:
    """Load the native search input and separate the DGDR output configuration."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain one YAML mapping")

    output = raw.pop(_OUTPUT_NAME, None)
    if not isinstance(output, dict):
        raise TypeError(f"{path} must contain a {_OUTPUT_NAME!r} mapping")

    search = SmartSearchConfig.model_validate(raw)
    resolved_output = DGDRRunOutputConfig.model_validate(output)
    return search, output, resolved_output.max_candidates


def run(config_path: Path, output_dir: Path) -> None:
    """Execute one native DGDR search and publish its candidate snapshot."""
    search, output, max_candidates = _load_input(config_path)
    output_configs = {_OUTPUT_NAME: output}
    output_dir.mkdir(parents=True, exist_ok=True)

    adapters = resolve_output_adapters(output_configs)
    callbacks = resolve_output_callbacks(
        output_configs,
        context=RecommendationOutputContext(workload=search.workload),
    )
    result = Sweeper(
        runner_factory=DynamoReplayRunnerFactory(),
        show_progress=True,
    ).run(
        search,
        top_n=max_candidates,
        on_candidate=callbacks.on_candidate,
        on_round=callbacks.on_round,
    )
    write_output_adapters(
        adapters,
        output_configs,
        result=result,
        output_dir=output_dir,
    )


def main() -> int:
    """Parse the container command line and run the search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
