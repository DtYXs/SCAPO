from __future__ import annotations

from typing import Any

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError as exc:
    raise ImportError(
        "The eval reward requires Math-Verify. Install it with `python -m pip install math-verify`."
    ) from exc

REWARD_NAME = "math_verify"
REWARD_TYPE = "batch"

_MATH_VERIFY = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)


def math_verify_accuracy(response: str, ground_truth: Any) -> float:
    gold = rf"\boxed{{{str(ground_truth).strip()}}}"
    try:
        score, _ = _MATH_VERIFY([gold], [response])
        return float(score)
    except TimeoutException:
        return 0.0
    except Exception:
        return 0.0


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for the Math-Verify eval reward function.")

    scores = []
    for reward_input in reward_inputs:
        score = math_verify_accuracy(reward_input["response"], reward_input["ground_truth"])
        scores.append(
            {
                "overall": score,
                "accuracy": score,
                "math_verify_accuracy": score,
            }
        )
    return scores
