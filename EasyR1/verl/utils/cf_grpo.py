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
"""
Token-level policy-loss weights for Counterfactual GRPO (CF-GRPO).

Detect reasoning spans, replace each span with an equal-length placeholder,
measure the answer log-probability drop under teacher forcing, and weight the
token-level policy-gradient loss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F


_ANSWER_MARKERS = (
    re.compile(r"(?i)\banswer\s*:"),
    re.compile(r"(?i)\bfinal\s+answer\b"),
    re.compile(r"####"),
    re.compile(r"\\boxed\s*\{"),
)
_ARITHMETIC_RE = re.compile(
    r"[-+]?\d+(?:\.\d+)?"
    r"(?:\s*(?:[+\-*/]|x|X|\\times|\\div|×|÷)\s*[-+]?\d+(?:\.\d+)?)+"
    r"\s*=\s*[-+]?\d+(?:\.\d+)?"
)


@dataclass(frozen=True)
class AnswerRegion:
    reasoning_start: int
    reasoning_end: int
    answer_start: int
    answer_end: int
    marker_text: str


@dataclass
class CFGRPOResult:
    weights: torch.Tensor
    skipped_zero_advantage: torch.Tensor
    skipped_no_answer: torch.Tensor
    skipped_no_spans: torch.Tensor
    span_counts: torch.Tensor
    cf_grpo_forward_count: int


def _decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(token_ids)


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return _decode_ids(tokenizer, [int(token_id)])


def _to_int_list(token_ids: torch.Tensor | list[int]) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        return [int(x) for x in token_ids.detach().cpu().tolist()]
    return [int(x) for x in token_ids]


def _valid_response_len(response_mask: Optional[torch.Tensor], max_response_len: int) -> int:
    if response_mask is None:
        return max_response_len

    valid_len = int(response_mask.detach().to(torch.long).sum().item())
    return max(0, min(valid_len, max_response_len))


def _find_answer_marker(text: str) -> Optional[re.Match[str]]:
    best_match = None
    for pattern in _ANSWER_MARKERS:
        match = pattern.search(text)
        if match is None:
            continue
        if best_match is None or match.start() < best_match.start():
            best_match = match

    return best_match


def _char_to_token_index(token_texts: list[str], char_pos: int) -> int:
    cursor = 0
    for idx, token_text in enumerate(token_texts):
        cursor_next = cursor + len(token_text)
        if char_pos < cursor_next:
            return idx
        cursor = cursor_next

    return max(0, len(token_texts) - 1)


def find_answer_region(
    tokenizer: Any,
    input_ids: torch.Tensor | list[int],
    prompt_length: int,
    response_mask: Optional[torch.Tensor] = None,
) -> Optional[AnswerRegion]:
    """Locate the final-answer region inside a generated response.

    Returns None if no explicit answer marker is present. This avoids assigning
    CF-GRPO weights to arbitrary response tails when the model did not
    follow the answer format.
    """
    token_ids = _to_int_list(input_ids)
    response_start = prompt_length
    max_response_len = max(0, len(token_ids) - response_start)
    response_len = _valid_response_len(response_mask, max_response_len)
    if response_len <= 0:
        return None

    response_ids = token_ids[response_start : response_start + response_len]
    response_text = _decode_ids(tokenizer, response_ids)
    marker = _find_answer_marker(response_text)
    if marker is None:
        return None

    response_token_texts = [_decode_token(tokenizer, token_id) for token_id in response_ids]
    marker_token_offset = _char_to_token_index(response_token_texts, marker.start())
    answer_start = response_start + marker_token_offset
    answer_end = response_start + response_len
    if answer_start >= answer_end:
        return None

    return AnswerRegion(
        reasoning_start=response_start,
        reasoning_end=answer_start,
        answer_start=answer_start,
        answer_end=answer_end,
        marker_text=marker.group(0),
    )


def _span_from_char_offsets(
    token_texts: list[str],
    base_index: int,
    char_start: int,
    char_end: int,
) -> Optional[tuple[int, int]]:
    cursor = 0
    token_start = None
    token_end = None

    for offset, token_text in enumerate(token_texts):
        cursor_next = cursor + len(token_text)
        if token_start is None and char_start < cursor_next:
            token_start = base_index + offset
        if token_start is not None and char_end <= cursor_next:
            token_end = base_index + offset + 1
            break
        cursor = cursor_next

    if token_start is None or token_end is None or token_start >= token_end:
        return None
    return token_start, token_end


def _detect_arithmetic_spans(
    token_texts: list[str],
    reasoning_start: int,
    reasoning_end: int,
    max_spans: int,
) -> list[tuple[int, int]]:
    reasoning_tokens = token_texts[reasoning_start:reasoning_end]
    reasoning_text = "".join(reasoning_tokens)
    spans = []

    for match in _ARITHMETIC_RE.finditer(reasoning_text):
        span = _span_from_char_offsets(reasoning_tokens, reasoning_start, match.start(), match.end())
        if span is not None:
            spans.append(span)
        if len(spans) >= max_spans:
            break

    return spans


def _detect_sentence_spans(
    token_texts: list[str],
    reasoning_start: int,
    reasoning_end: int,
    max_spans: int,
    min_tokens: int,
) -> list[tuple[int, int]]:
    spans = []
    sentence_start = reasoning_start

    for idx in range(reasoning_start, reasoning_end):
        token_text = token_texts[idx]
        stripped = token_text.strip()
        is_boundary = False
        if stripped.endswith((".", "!", "?")):
            is_boundary = True
        elif "\n\n" in token_text:
            is_boundary = True
        elif "\n" in token_text and idx + 1 < reasoning_end and token_texts[idx + 1].strip():
            is_boundary = True

        if not is_boundary:
            continue

        sentence_end = idx + 1
        if sentence_end - sentence_start >= min_tokens:
            spans.append((sentence_start, sentence_end))
        sentence_start = sentence_end
        if len(spans) >= max_spans:
            break

    if len(spans) < max_spans and reasoning_end - sentence_start >= min_tokens:
        spans.append((sentence_start, reasoning_end))

    return spans


def _overlaps(span: tuple[int, int], other: tuple[int, int]) -> bool:
    return not (span[1] <= other[0] or span[0] >= other[1])


def detect_reasoning_spans(
    token_texts: list[str],
    reasoning_start: int,
    reasoning_end: int,
    max_spans: int = 10,
    min_tokens: int = 5,
    max_tokens: int = 50,
) -> list[tuple[int, int]]:
    """Detect arithmetic spans first, then sentence spans as fallback coverage."""
    if reasoning_end <= reasoning_start or max_spans <= 0:
        return []

    arithmetic_spans = _detect_arithmetic_spans(
        token_texts=token_texts,
        reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
        max_spans=max(1, max_spans // 2),
    )
    sentence_spans = _detect_sentence_spans(
        token_texts=token_texts,
        reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
        max_spans=max_spans,
        min_tokens=min_tokens,
    )

    merged = list(arithmetic_spans)
    for span in sentence_spans:
        if any(_overlaps(span, arithmetic_span) for arithmetic_span in arithmetic_spans):
            continue
        merged.append(span)

    clipped = []
    for start, end in sorted(merged, key=lambda item: item[0]):
        if end - start > max_tokens:
            continue
        clipped.append((start, end))
        if len(clipped) >= max_spans:
            break

    return clipped


def build_weights_from_span_scores(
    response_length: int,
    prompt_length: int,
    answer_start: int,
    answer_end: int,
    spans: list[tuple[int, int]],
    span_scores: list[float],
    min_weight: float = 0.5,
    max_weight: float = 4.0,
    answer_weight: float = 1.5,
    eps: float = 1e-6,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Map span-level answer log-probability drops to CF-GRPO token weights.

    This mirrors the official CounterfactualGRPO weighting shape: every
    reasoning token starts with a small baseline importance, each span's answer
    log-probability drop is divided over its tokens, and the reasoning region is
    normalized before being mapped to soft weights.
    """
    if min_weight <= 0:
        raise ValueError("cf_grpo_min_weight must be positive.")
    if max_weight < min_weight:
        raise ValueError("cf_grpo_max_weight must be >= cf_grpo_min_weight.")

    weights = torch.ones(response_length, dtype=torch.float32, device=device)
    if response_length <= 0:
        return weights

    reasoning_rel_end = max(0, min(response_length, answer_start - prompt_length))
    if reasoning_rel_end > 0:
        importance = torch.full((reasoning_rel_end,), 0.2, dtype=torch.float32, device=device)
        for (span_start, span_end), span_score in zip(spans, span_scores):
            rel_start = max(0, span_start - prompt_length)
            rel_end = min(reasoning_rel_end, span_end - prompt_length)
            if rel_start >= rel_end:
                continue
            span_len = max(1, span_end - span_start)
            importance[rel_start:rel_end] += float(span_score) / span_len

        importance_min = torch.min(importance)
        importance_max = torch.max(importance)
        if torch.abs(importance_max - importance_min) > eps:
            normalized = (importance - importance_min) / (importance_max - importance_min + eps)
            normalized = 0.5 + 0.5 * normalized
        else:
            normalized = torch.full_like(importance, 0.5)
        weights[:reasoning_rel_end] = min_weight + normalized * (max_weight - min_weight)

    answer_rel_start = max(0, answer_start - prompt_length)
    answer_rel_end = min(response_length, answer_end - prompt_length)
    if answer_rel_start < answer_rel_end:
        weights[answer_rel_start:answer_rel_end] = answer_weight

    return weights


def _score_answer_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_start: int,
    answer_end: int,
) -> float:
    target_start = answer_start + 1
    if answer_start < 0 or answer_end <= target_start:
        return 0.0

    # Match the official CounterfactualGRPO implementation: reasoning_end is
    # the marker start, and scoring starts from the next teacher-forced token.
    logits_slice = logits[answer_start : answer_end - 1, :]
    target_ids = input_ids[target_start:answer_end]
    log_probs = F.log_softmax(logits_slice.float(), dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item())


def _distributed_max(value: int, device: torch.device) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value

    value_tensor = torch.tensor(value, dtype=torch.long, device=device)
    torch.distributed.all_reduce(value_tensor, op=torch.distributed.ReduceOp.MAX)
    return int(value_tensor.item())


@torch.no_grad()
def compute_cf_grpo_weights(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: Optional[torch.Tensor] = None,
    min_weight: float = 0.5,
    max_weight: float = 4.0,
    answer_weight: float = 1.5,
    max_spans: int = 10,
    span_min_tokens: int = 5,
    span_max_tokens: int = 50,
    cf_grpo_batch_size: int = 8,
    eps: float = 1e-6,
) -> CFGRPOResult:
    """Compute CF-GRPO response weights, skipping probes for zero-advantage samples."""
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    response_length = response_mask.shape[-1]
    prompt_length = seq_len - response_length
    weights = torch.ones((batch_size, response_length), dtype=torch.float32, device=device)
    skipped_zero_advantage = torch.zeros(batch_size, dtype=torch.float32, device=device)
    skipped_no_answer = torch.zeros(batch_size, dtype=torch.float32, device=device)
    skipped_no_spans = torch.zeros(batch_size, dtype=torch.float32, device=device)
    span_counts = torch.zeros(batch_size, dtype=torch.float32, device=device)

    sample_infos: dict[int, dict[str, Any]] = {}
    all_sequences = []
    all_attention_masks = []
    all_position_ids = []
    sequence_records = []
    cf_grpo_forward_count = 0
    cf_grpo_batch_size = max(1, int(cf_grpo_batch_size))
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    mask_token_id = pad_token_id if pad_token_id is not None else eos_token_id
    if mask_token_id is None:
        mask_token_id = 0

    for batch_idx in range(batch_size):
        valid_response_length = _valid_response_len(response_mask[batch_idx], response_length)
        if valid_response_length <= 0:
            skipped_no_answer[batch_idx] = 1.0
            continue

        if advantages is not None:
            valid_advantages = advantages[batch_idx, :valid_response_length]
            if valid_advantages.numel() == 0 or torch.max(torch.abs(valid_advantages)).item() <= eps:
                skipped_zero_advantage[batch_idx] = 1.0
                continue

        region = find_answer_region(
            tokenizer=tokenizer,
            input_ids=input_ids[batch_idx],
            prompt_length=prompt_length,
            response_mask=response_mask[batch_idx],
        )
        if region is None:
            skipped_no_answer[batch_idx] = 1.0
            continue

        valid_seq_len = prompt_length + valid_response_length
        token_texts = [_decode_token(tokenizer, token_id) for token_id in input_ids[batch_idx, :valid_seq_len]]
        spans = detect_reasoning_spans(
            token_texts=token_texts,
            reasoning_start=region.reasoning_start,
            reasoning_end=region.reasoning_end,
            max_spans=max_spans,
            min_tokens=span_min_tokens,
            max_tokens=span_max_tokens,
        )
        span_counts[batch_idx] = float(len(spans))

        if not spans:
            skipped_no_spans[batch_idx] = 1.0
            weights[batch_idx, :valid_response_length] = build_weights_from_span_scores(
                response_length=valid_response_length,
                prompt_length=prompt_length,
                answer_start=region.answer_start,
                answer_end=region.answer_end,
                spans=[],
                span_scores=[],
                min_weight=min_weight,
                max_weight=max_weight,
                answer_weight=answer_weight,
                eps=eps,
                device=device,
            )
            continue

        sample_infos[batch_idx] = {"region": region, "spans": spans}
        all_sequences.append(input_ids[batch_idx].clone())
        all_attention_masks.append(attention_mask[batch_idx].clone())
        all_position_ids.append(position_ids[batch_idx].clone())
        sequence_records.append((batch_idx, -1))

        for span_idx, (span_start, span_end) in enumerate(spans):
            masked_ids = input_ids[batch_idx].clone()
            masked_ids[span_start:span_end] = mask_token_id
            all_sequences.append(masked_ids)
            all_attention_masks.append(attention_mask[batch_idx].clone())
            all_position_ids.append(position_ids[batch_idx].clone())
            sequence_records.append((batch_idx, span_idx))

    global_sequence_count = _distributed_max(len(all_sequences), device)
    if global_sequence_count > 0:
        scores: dict[int, dict[str, Any]] = {}
        if batch_size <= 0:
            dummy_input_ids = torch.zeros(seq_len, dtype=input_ids.dtype, device=device)
            dummy_attention_mask = torch.zeros(seq_len, dtype=attention_mask.dtype, device=device)
            dummy_position_ids = position_ids.new_zeros(position_ids.shape[1:] or (seq_len,))
        else:
            dummy_input_ids = input_ids[0].clone()
            dummy_attention_mask = attention_mask[0].clone()
            dummy_position_ids = position_ids[0].clone()

        for chunk_start in range(0, global_sequence_count, cf_grpo_batch_size):
            chunk_end = min(global_sequence_count, chunk_start + cf_grpo_batch_size)
            local_end = min(len(all_sequences), chunk_end)
            local_sequences = all_sequences[chunk_start:local_end]
            local_attention_masks = all_attention_masks[chunk_start:local_end]
            local_position_ids = all_position_ids[chunk_start:local_end]
            chunk_records: list[tuple[Optional[int], Optional[int]]] = list(sequence_records[chunk_start:local_end])

            pad_count = (chunk_end - chunk_start) - len(local_sequences)
            if pad_count > 0:
                local_sequences.extend(dummy_input_ids.clone() for _ in range(pad_count))
                local_attention_masks.extend(dummy_attention_mask.clone() for _ in range(pad_count))
                local_position_ids.extend(dummy_position_ids.clone() for _ in range(pad_count))
                chunk_records.extend((None, None) for _ in range(pad_count))

            chunk_input_ids = torch.stack(local_sequences, dim=0)
            chunk_attention_mask = torch.stack(local_attention_masks, dim=0)
            chunk_position_ids = torch.stack(local_position_ids, dim=0)

            outputs = model(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
                position_ids=chunk_position_ids,
                use_cache=False,
            )
            cf_grpo_forward_count += 1

            for local_idx, (batch_idx, span_idx) in enumerate(chunk_records):
                if batch_idx is None or span_idx is None:
                    continue
                region = sample_infos[batch_idx]["region"]
                score = _score_answer_logprob(
                    logits=outputs.logits[local_idx],
                    input_ids=input_ids[batch_idx],
                    answer_start=region.answer_start,
                    answer_end=region.answer_end,
                )
                if batch_idx not in scores:
                    scores[batch_idx] = {"baseline": None, "masked": {}}
                if span_idx == -1:
                    scores[batch_idx]["baseline"] = score
                else:
                    scores[batch_idx]["masked"][span_idx] = score

            del outputs

        for batch_idx, info in sample_infos.items():
            if batch_idx not in scores or scores[batch_idx]["baseline"] is None:
                continue

            valid_response_length = _valid_response_len(response_mask[batch_idx], response_length)
            baseline_score = scores[batch_idx]["baseline"]
            span_scores = []
            for span_idx in range(len(info["spans"])):
                masked_score = scores[batch_idx]["masked"].get(span_idx, baseline_score)
                span_scores.append(baseline_score - masked_score)

            weights[batch_idx, :valid_response_length] = build_weights_from_span_scores(
                response_length=valid_response_length,
                prompt_length=prompt_length,
                answer_start=info["region"].answer_start,
                answer_end=info["region"].answer_end,
                spans=info["spans"],
                span_scores=span_scores,
                min_weight=min_weight,
                max_weight=max_weight,
                answer_weight=answer_weight,
                eps=eps,
                device=device,
            )

    return CFGRPOResult(
        weights=weights,
        skipped_zero_advantage=skipped_zero_advantage,
        skipped_no_answer=skipped_no_answer,
        skipped_no_spans=skipped_no_spans,
        span_counts=span_counts,
        cf_grpo_forward_count=cf_grpo_forward_count,
    )
