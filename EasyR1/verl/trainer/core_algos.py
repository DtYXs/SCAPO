# Copyright 2022 The HuggingFace Team
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

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF

if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        ...


class AdaptiveKLController(KLController):
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class AdvantageEstimator(str, Enum):
    GAE = "gae"
    GRPO = "grpo"
    GRPO_PASSK = "grpo_passk"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


ADV_ESTIMATOR_MAP: dict[str, Any] = {}


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


def resolve_semifactual_lambda(
    *,
    base_lambda: float,
    current_step: int,
    n_steps: int,
) -> tuple[float, bool]:
    if current_step <= 0:
        raise ValueError(f"current_step must be positive, got: {current_step}.")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got: {n_steps}.")
    active = current_step <= n_steps
    value = float(base_lambda) if active else 0.0
    return value, bool(active and value > 0.0)


def masked_group_zscore(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any,
    eps: float = 1e-6,
) -> torch.Tensor:
    if isinstance(index, torch.Tensor):
        index = index.detach().cpu().numpy()
    else:
        index = np.asarray(index)

    normalized = torch.zeros_like(values)
    for uid in np.unique(index):
        row_indices = torch.as_tensor(np.where(index == uid)[0], device=values.device, dtype=torch.long)
        group_values = values[row_indices]
        group_mask = response_mask[row_indices].bool()
        valid_values = group_values[group_mask]
        if valid_values.numel() <= 1:
            continue

        std = valid_values.std(unbiased=False)
        if std.detach().item() <= eps:
            continue
        normalized[row_indices] = (group_values - valid_values.mean()) / (std + eps)

    return normalized * response_mask


def compute_semifactual_distance(
    base_log_probs: torch.Tensor,
    perturbed_log_probs: torch.Tensor,
) -> torch.Tensor:
    abs_log_ratio = (base_log_probs - perturbed_log_probs).abs()
    return 2.0 * torch.tanh(0.5 * abs_log_ratio)


@torch.no_grad()
def compute_semifactual_importance(
    base_log_probs: torch.Tensor,
    perturbed_log_probs_by_type: dict[str, torch.Tensor],
    response_mask: torch.Tensor,
    index: Any,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if not perturbed_log_probs_by_type:
        raise ValueError("At least one semifactual log-prob tensor is required.")

    type_scores = []
    type_distances = []
    for perturb_type, perturbed_log_probs in perturbed_log_probs_by_type.items():
        if perturbed_log_probs.shape != base_log_probs.shape:
            raise ValueError(
                f"Semifactual log-prob shape mismatch for {perturb_type!r}: "
                f"{tuple(perturbed_log_probs.shape)} != {tuple(base_log_probs.shape)}."
            )
        distance = compute_semifactual_distance(base_log_probs, perturbed_log_probs) * response_mask
        score = masked_group_zscore(-distance, response_mask, index, eps=eps)
        type_distances.append(distance)
        type_scores.append(score)

    aggregate_distance = torch.stack(type_distances, dim=0).mean(dim=0) * response_mask
    aggregate_score = torch.stack(type_scores, dim=0).mean(dim=0) * response_mask
    normalized_score = masked_group_zscore(aggregate_score, response_mask, index, eps=eps)

    metrics = {"distance/mean": VF.masked_mean(aggregate_distance, response_mask).detach().item()}
    return aggregate_score, normalized_score, metrics


def negative_semifactual_signal(
    normalized_score: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.minimum(normalized_score, torch.zeros_like(normalized_score)) * response_mask


def register_adv_estimator(name: AdvantageEstimator):
    def decorator(fn):
        wrapped_fn = torch.no_grad()(fn)
        ADV_ESTIMATOR_MAP[getattr(name, "value", name)] = wrapped_fn
        return wrapped_fn

    return decorator


def compute_advantage_return(name: AdvantageEstimator, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    return ADV_ESTIMATOR_MAP[getattr(name, "value", name)](**kwargs)


@register_adv_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    nextvalues = 0
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        gaelam = delta + gamma * lam * lastgaelam

        if response_mask[:, t]:
            nextvalues = values[:, t]
            lastgaelam = gaelam

        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)
    id2score = defaultdict(list)
    id2indices = defaultdict(list)

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
        id2indices[index[i]].append(i)

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        rewards = torch.tensor(id2score[idx])
        topk, topk_idx = torch.topk(rewards, k=2)
        r_max, r_second_max = topk[0], topk[1]
        i_max = id2indices[idx][topk_idx[0]]
        advantages[i_max] = (r_max - r_second_max) / (torch.std(torch.tensor(id2score[idx])) + eps)

    returns = advantages.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return

        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = (token_level_rewards.sum(dim=-1) - reward_baselines) * response_mask
    returns = (token_level_rewards * response_mask).flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    return advantages, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def average_loss(
    values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
) -> torch.Tensor:
    if mode == "token":
        return VF.masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")


@torch.no_grad()
def compute_fipo_future_kl(
    log_prob_shift: torch.Tensor,
    response_mask: torch.Tensor,
    decay_rate: float = 32.0,
    ratio_filter_threshold: float = 10.0,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    if log_prob_shift.shape != response_mask.shape:
        raise ValueError(
            f"log_prob_shift and response_mask must have the same shape, got "
            f"{log_prob_shift.shape} and {response_mask.shape}."
        )
    if log_prob_shift.ndim != 2:
        raise ValueError(f"FIPO expects 2D [batch, response_length] tensors, got {log_prob_shift.shape}.")
    if decay_rate <= 0:
        raise ValueError(f"fipo_decay_rate must be positive, got {decay_rate}.")
    if ratio_filter_threshold <= 1:
        raise ValueError(f"FIPO ratio_filter_threshold must be greater than 1, got {ratio_filter_threshold}.")
    if chunk_size <= 0:
        raise ValueError(f"fipo_chunk_size must be positive, got {chunk_size}.")

    shift = log_prob_shift.detach().float().clamp(min=-20.0, max=20.0)
    valid_mask = response_mask.bool()
    filter_log_threshold = float(np.log(ratio_filter_threshold))
    participation_mask = valid_mask & (shift <= filter_log_threshold)
    filtered_shift = shift * participation_mask

    batch_size, response_length = shift.shape
    if response_length == 0:
        return torch.zeros_like(shift), participation_mask

    block_size = min(int(chunk_size), response_length)
    gamma = float(np.exp2(-1.0 / decay_rate))
    positions = torch.arange(block_size, device=shift.device, dtype=torch.float32)
    distances = positions.unsqueeze(0) - positions.unsqueeze(1)
    decay_matrix = torch.where(distances >= 0, gamma**distances, torch.zeros_like(distances))

    blocks: list[torch.Tensor] = []
    carry = torch.zeros(batch_size, device=shift.device, dtype=torch.float32)
    for block_end in range(response_length, 0, -block_size):
        block_start = max(0, block_end - block_size)
        width = block_end - block_start
        block_shift = filtered_shift[:, block_start:block_end]
        local_future = block_shift @ decay_matrix[:width, :width].T
        tail_decay = gamma ** torch.arange(width, 0, -1, device=shift.device, dtype=torch.float32)
        block_future = local_future + carry.unsqueeze(-1) * tail_decay
        blocks.append(block_future)
        carry = block_future[:, 0]

    future_kl = torch.cat(list(reversed(blocks)), dim=-1)
    return future_kl * valid_mask, participation_mask


@torch.no_grad()
def compute_fipo_influence_weights(
    log_prob_shift: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    decay_rate: float = 32.0,
    influence_clip_low: float = 0.0,
    influence_clip_high: float = 0.2,
    ratio_filter_threshold: float = 10.0,
    safety_threshold: float = 10.0,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if advantages.shape != log_prob_shift.shape:
        raise ValueError(
            f"advantages and log_prob_shift must have the same shape, got "
            f"{advantages.shape} and {log_prob_shift.shape}."
        )
    if not 0 <= influence_clip_low < 1:
        raise ValueError(f"fipo_influence_clip_low must be in [0, 1), got {influence_clip_low}.")
    if influence_clip_high < 0:
        raise ValueError(f"fipo_influence_clip_high must be non-negative, got {influence_clip_high}.")
    if safety_threshold <= 1:
        raise ValueError(f"fipo_safety_threshold must be greater than 1, got {safety_threshold}.")

    future_kl, participation_mask = compute_fipo_future_kl(
        log_prob_shift=log_prob_shift,
        response_mask=response_mask,
        decay_rate=decay_rate,
        ratio_filter_threshold=ratio_filter_threshold,
        chunk_size=chunk_size,
    )
    valid_mask = response_mask.bool()
    lower_bound = 1.0 - influence_clip_low
    upper_bound = 1.0 + influence_clip_high

    clipped_future_kl = future_kl.clamp(min=float(np.log(lower_bound)), max=float(np.log(upper_bound)))
    influence_weights = torch.exp(clipped_future_kl)
    influence_weights = torch.where(valid_mask, influence_weights, torch.ones_like(influence_weights))

    bounded_shift = log_prob_shift.detach().float().clamp(min=-20.0, max=20.0)
    ratio = torch.exp(bounded_shift)
    safety_mask = valid_mask & (advantages < 0) & (ratio > safety_threshold)
    safe_weights = influence_weights.clamp(min=lower_bound, max=1.0)
    influence_weights = torch.where(safety_mask, safe_weights, influence_weights)

    upper_clip_mask = valid_mask & (influence_weights >= upper_bound - 1.0e-7)
    lower_clip_mask = valid_mask & (influence_weights <= lower_bound + 1.0e-7)
    extreme_filter_mask = valid_mask & ~participation_mask
    valid_count = valid_mask.sum()
    valid_future_kl = future_kl[valid_mask]
    valid_weights = influence_weights[valid_mask]
    zero = torch.zeros((), device=future_kl.device, dtype=torch.float32)
    diagnostics = {
        "fipo_diag_valid_count": valid_count,
        "fipo_diag_future_kl_sum": valid_future_kl.sum() if valid_future_kl.numel() else zero,
        "fipo_diag_influence_weight_sum": valid_weights.sum() if valid_weights.numel() else zero,
        "fipo_diag_upper_clip_count": upper_clip_mask.sum(),
        "fipo_diag_lower_clip_count": lower_clip_mask.sum(),
        "fipo_diag_extreme_filter_count": extreme_filter_mask.sum(),
        "fipo_diag_safety_count": safety_mask.sum(),
    }
    return future_kl, influence_weights.to(dtype=advantages.dtype), diagnostics


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    tau_positive: float,
    tau_negative: float,
    loss_type: Literal["default", "gspo", "gspo_token", "sapo", "cf_grpo", "fipo"],
    loss_avg_mode: Literal["token", "seq"],
    fipo_decay_rate: float = 32.0,
    fipo_influence_clip_low: float = 0.0,
    fipo_influence_clip_high: float = 0.2,
    fipo_safety_threshold: float = 10.0,
    fipo_chunk_size: int = 128,
    loss_weights: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    if loss_type == "cf_grpo" and loss_weights is None:
        raise ValueError("CF-GRPO requires response-token weights before the actor update.")
    negative_approx_kl = log_probs - old_log_probs
    if loss_type in ["gspo", "gspo_token"]:
        negative_approx_kl_in_seq = VF.masked_mean(negative_approx_kl, response_mask, dim=-1)

        if loss_type == "gspo_token":
            log_importance_ratio = negative_approx_kl_in_seq.detach().unsqueeze(-1) + log_probs - log_probs.detach()
        else:
            log_importance_ratio = negative_approx_kl_in_seq.unsqueeze(-1) * response_mask
    elif loss_type == "fipo":
        log_importance_ratio = negative_approx_kl.clamp(min=-20.0, max=20.0)
    else:
        log_importance_ratio = negative_approx_kl

    ratio = torch.exp(torch.clamp(log_importance_ratio, -20.0, 20.0))
    clipped_ratio = torch.exp(
        torch.clamp(log_importance_ratio, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    metrics = {"ppo_kl": -log_importance_ratio if loss_type == "fipo" else -negative_approx_kl}

    metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode)
    clip_count_metrics = {}
    fipo_diagnostics = {}
    effective_advantages = advantages
    if loss_type == "fipo":
        _, influence_weights, fipo_diagnostics = compute_fipo_influence_weights(
            log_prob_shift=negative_approx_kl,
            advantages=advantages,
            response_mask=response_mask,
            decay_rate=fipo_decay_rate,
            influence_clip_low=fipo_influence_clip_low,
            influence_clip_high=fipo_influence_clip_high,
            ratio_filter_threshold=clip_ratio_dual,
            safety_threshold=fipo_safety_threshold,
            chunk_size=fipo_chunk_size,
        )
        effective_advantages = advantages * influence_weights

    if loss_type == "sapo":
        positive_token_mask = (advantages >= 0).float()
        negative_token_mask = (advantages < 0).float()
        gate_negative = 4.0 / tau_negative * torch.sigmoid(tau_negative * (ratio - 1.0))
        gate_positive = 4.0 / tau_positive * torch.sigmoid(tau_positive * (ratio - 1.0))
        final_pg_loss = -advantages * (positive_token_mask * gate_positive + negative_token_mask * gate_negative)
    else:
        pg_loss = -effective_advantages * ratio
        pg_loss2 = -effective_advantages * clipped_ratio
        pg_loss3 = -effective_advantages * clip_ratio_dual

        clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)
        primary_clip_mask = pg_loss < pg_loss2
        valid_response_mask = response_mask.bool()
        positive_adv_mask = (advantages > 0) & valid_response_mask
        negative_adv_mask = (advantages < 0) & valid_response_mask
        high_clip_mask = primary_clip_mask & positive_adv_mask
        low_clip_mask = primary_clip_mask & negative_adv_mask

        metrics["pg_clipfrac_higher"] = primary_clip_mask.float()

        clip_count_metrics = {
            "pg_clip_high_count": high_clip_mask.sum(),
            "pg_clip_low_count": low_clip_mask.sum(),
            "pg_valid_token_count": valid_response_mask.sum(),
        }

        clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)
        final_pg_loss = torch.where(effective_advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
        metrics["pg_clipfrac_lower"] = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_response_mask = response_mask
    if loss_type == "fipo":
        dual_clip_mask = (advantages < 0) & (clipped_pg_loss_higher > pg_loss3) & response_mask.bool()
        sequence_drop_mask = dual_clip_mask.sum(dim=-1) > 1
        final_response_mask = response_mask * (~sequence_drop_mask).unsqueeze(-1)
        fipo_diagnostics.update(
            {
                "fipo_diag_sequence_count": torch.tensor(
                    response_mask.size(0), device=response_mask.device, dtype=torch.long
                ),
                "fipo_diag_sequence_drop_count": sequence_drop_mask.sum(),
            }
        )

    if loss_weights is not None:
        if loss_weights.shape != final_pg_loss.shape:
            raise ValueError(
                f"loss_weights shape {tuple(loss_weights.shape)} does not match policy loss shape "
                f"{tuple(final_pg_loss.shape)}."
            )
        final_pg_loss = final_pg_loss * loss_weights

    final_pg_loss = average_loss(final_pg_loss, final_response_mask, mode=loss_avg_mode)
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}
    metrics.update({k: v.detach().item() for k, v in clip_count_metrics.items()})
    metrics.update({k: v.detach().item() for k, v in fipo_diagnostics.items()})
    return final_pg_loss, metrics


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_avg_mode: Literal["token", "seq"],
) -> tuple[torch.Tensor, dict[str, float]]:
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    clipped_vf_losses = torch.max(vf_loss1, vf_loss2)
    vf_loss = 0.5 * average_loss(clipped_vf_losses, response_mask, mode=loss_avg_mode)
    metrics = {
        "vf_clipfrac": VF.masked_mean((vf_loss1 < vf_loss2).float(), response_mask).detach().item(),
        "vpred_mean": VF.masked_mean(vpreds, response_mask).detach().item(),
    }
    return vf_loss, metrics


def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
) -> torch.Tensor:
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    if kl_penalty == "low_var_kl":
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
