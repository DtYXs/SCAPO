# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
from typing import Any

PERTURBATION_TYPES = ("paraphrase", "typo_noise", "scenario_wrap", "irrelevant_context")


def semifactual_key(problem: Any, answer: Any) -> tuple[str, str]:
    return str(problem), str(answer)


def _extract_perturbations(row: dict[str, Any]) -> dict[str, str]:
    mapping = row.get("perturbations", row.get("semifactual_perturbations"))
    if isinstance(mapping, dict):
        return {str(key): str(value) for key, value in mapping.items() if key and value}

    items = row.get("irrelevant_perturbations", [])
    return {
        str(item["perturbation_type"]): str(item["perturbed_question"])
        for item in items
        if isinstance(item, dict) and item.get("perturbation_type") and item.get("perturbed_question")
    }


def load_semifactual_jsonl(path: str) -> dict[tuple[str, str], dict[str, str]]:
    semifactuals: dict[tuple[str, str], dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            if row.get("status", "success") != "success":
                continue
            if "problem" not in row or "answer" not in row:
                raise ValueError(f"Missing problem/answer at {path}:{line_no}.")

            key = semifactual_key(row["problem"], row["answer"])
            if key in semifactuals:
                raise ValueError(f"Duplicate semifactual row for problem/answer at {path}:{line_no}.")

            perturbations = _extract_perturbations(row)
            missing = [name for name in PERTURBATION_TYPES if name not in perturbations]
            if missing:
                raise ValueError(f"Missing SCAPO perturbations {missing!r} at {path}:{line_no}.")
            semifactuals[key] = {name: perturbations[name] for name in PERTURBATION_TYPES}

    return semifactuals
