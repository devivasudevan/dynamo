# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from dynamo.aisimulate.output.dgdr_run import runner

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def test_load_input_separates_output_config_from_native_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native = {
        "search_space": {"backend": ["vllm"]},
        "workload": {"request_count": 8},
        "goal": {"metric": "throughput_per_gpu"},
        "sweep": {"max_rounds": 1},
        "dgdr_run": {
            "name": "run-1",
            "runtime_image": "example.com/planner:1",
            "num_gpus_per_node": 8,
            "max_candidates": 3,
        },
    }
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(native))
    validated: list[dict[str, Any]] = []
    search = SimpleNamespace(workload=object())

    def validate(value: dict[str, Any]) -> Any:
        validated.append(value)
        return search

    monkeypatch.setattr(runner.SmartSearchConfig, "model_validate", validate)

    got_search, output, max_candidates = runner._load_input(path)

    assert got_search is search
    assert validated == [
        {key: value for key, value in native.items() if key != "dgdr_run"}
    ]
    assert output == native["dgdr_run"]
    assert max_candidates == 3


def test_load_input_requires_dgdr_output_config(tmp_path: Path) -> None:
    path = tmp_path / "input.yaml"
    path.write_text("search_space: {}\n")

    with pytest.raises(TypeError, match="must contain a 'dgdr_run' mapping"):
        runner._load_input(path)
