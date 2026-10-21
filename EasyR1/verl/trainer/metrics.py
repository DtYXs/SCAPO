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

from typing import Any

import numpy as np
import torch

from ..protocol import DataProto

_FIPO_DIAGNOSTIC_NAMES = (
    "valid_count",
    "future_kl_sum",
    "influence_weight_sum",
    "upper_clip_count",
    "lower_clip_count",
    "extreme_filter_count",
    "safety_count",
    "sequence_count",
    "sequence_drop_count",
)


def _reduce_fipo_diagnostics(
    metrics: dict[str, list[Any]],
    reduced: dict[str, Any],
    eps: float,
) -> None:
    keys = {name: f"actor/fipo_diag_{name}" for name in _FIPO_DIAGNOSTIC_NAMES}
    if all(key in metrics for key in keys.values()):
        valid_counts = np.asarray(metrics[keys["valid_count"]], dtype=np.float64)
        valid_count = float(valid_counts.sum())
        sequence_count = float(np.sum(metrics[keys["sequence_count"]]))

        def ratio(numerator: float, denominator: float) -> float:
            return numerator / denominator if denominator > eps else 0.0

        reduced["actor/fipo_future_kl_mean"] = ratio(float(np.sum(metrics[keys["future_kl_sum"]])), valid_count)
        reduced["actor/fipo_influence_weight_mean"] = ratio(
            float(np.sum(metrics[keys["influence_weight_sum"]])), valid_count
        )
        reduced["actor/fipo_influence_clipfrac_upper"] = ratio(
            float(np.sum(metrics[keys["upper_clip_count"]])), valid_count
        )
        reduced["actor/fipo_influence_clipfrac_lower"] = ratio(
            float(np.sum(metrics[keys["lower_clip_count"]])), valid_count
        )
        reduced["actor/fipo_extreme_ratio_filterfrac"] = ratio(
            float(np.sum(metrics[keys["extreme_filter_count"]])), valid_count
        )
        reduced["actor/fipo_safety_frac"] = ratio(float(np.sum(metrics[keys["safety_count"]])), valid_count)
        reduced["actor/fipo_sequence_drop_frac"] = ratio(
            float(np.sum(metrics[keys["sequence_drop_count"]])), sequence_count
        )

    for key in keys.values():
        reduced.pop(key, None)


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    reduced = {key: np.mean(value) for key, value in metrics.items()}

    clip_count_keys = {
        "high": "actor/pg_clip_high_count",
        "low": "actor/pg_clip_low_count",
        "valid": "actor/pg_valid_token_count",
    }
    eps = 1.0e-12

    if all(key in metrics for key in clip_count_keys.values()):
        counts = {name: float(np.sum(metrics[key])) for name, key in clip_count_keys.items()}
        reduced["actor/pg_clipfrac_higher"] = (counts["high"] + counts["low"]) / (counts["valid"] + eps)

    for key in clip_count_keys.values():
        reduced.pop(key, None)

    _reduce_fipo_diagnostics(metrics, reduced, eps)

    return reduced


def compute_length_metrics(batch: DataProto) -> dict[str, Any]:
    max_response_length = batch.batch["responses"].size(-1)
    max_prompt_length = batch.batch["attention_mask"].size(-1) - max_response_length

    prompt_length = batch.batch["attention_mask"][:, :-max_response_length].sum(-1).float()
    response_length = batch.batch["attention_mask"][:, -max_response_length:].sum(-1).float()

    return {

        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.eq(response_length, max_response_length).float().mean().detach().item(),

        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.eq(prompt_length, max_prompt_length).float().mean().detach().item(),
    }


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    return {

        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),

        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),

        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),

        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {

                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),

                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        **compute_length_metrics(batch),
    }


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], num_gpus: int) -> dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }
