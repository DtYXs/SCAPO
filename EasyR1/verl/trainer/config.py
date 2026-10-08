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

import math
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: Any = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: Any = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16
    semifactual_train_file: Optional[str] = None
    semifactual_unmatched_sample_policy: str = "error"

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")
        self.semifactual_train_file = get_abs_path(self.semifactual_train_file, prompt="Semifactual train file")


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "grpo"
    disable_kl: bool = False
    use_kl_loss: bool = False
    kl_penalty: str = "kl"
    kl_coef: float = 1e-3
    kl_type: str = "fixed"
    kl_horizon: float = 10000.0
    kl_target: float = 0.1
    online_filtering: bool = False
    filter_key: str = "overall"
    filter_low: float = 0.01
    filter_high: float = 0.99
    causal_credit_enabled: bool = False
    causal_credit_weighting_mode: str = "counterfactual"
    causal_credit_min_weight: float = 0.5
    causal_credit_max_weight: float = 4.0
    causal_credit_answer_weight: float = 1.5
    causal_credit_max_spans: int = 10
    causal_credit_span_min_tokens: int = 5
    causal_credit_span_max_tokens: int = 50
    causal_credit_cf_batch_size: int = 8
    causal_credit_mean_normalize: bool = False
    causal_credit_skip_zero_advantage: bool = True
    semifactual_enabled: bool = False
    semifactual_lambda: float = 0.01
    semifactual_n_steps: int = 60
    semifactual_norm_eps: float = 1e-6

    def post_init(self):
        if self.causal_credit_weighting_mode not in {"counterfactual", "inverted", "random", "uniform"}:
            raise ValueError("Unknown algorithm.causal_credit_weighting_mode.")
        for name in ("min_weight", "max_weight", "answer_weight"):
            value = getattr(self, f"causal_credit_{name}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"algorithm.causal_credit_{name} must be finite and positive.")
        if self.causal_credit_max_weight < self.causal_credit_min_weight:
            raise ValueError("causal_credit_max_weight must be >= causal_credit_min_weight.")
        if self.causal_credit_max_spans < 0 or self.causal_credit_cf_batch_size <= 0:
            raise ValueError("causal_credit_max_spans must be non-negative and cf_batch_size positive.")
        if not 0 < self.causal_credit_span_min_tokens <= self.causal_credit_span_max_tokens:
            raise ValueError("Causal-credit span lengths must satisfy 0 < min_tokens <= max_tokens.")
        if self.causal_credit_enabled and self.semifactual_enabled:
            raise ValueError("CF-GRPO and SCAPO must be enabled separately.")
        if self.causal_credit_enabled and self.adv_estimator != "grpo":
            raise ValueError("CF-GRPO requires algorithm.adv_estimator='grpo'.")
        if not math.isfinite(self.semifactual_lambda) or self.semifactual_lambda < 0:
            raise ValueError("algorithm.semifactual_lambda must be finite and non-negative.")
        if self.semifactual_n_steps < 0:
            raise ValueError("algorithm.semifactual_n_steps must be non-negative.")
        if not math.isfinite(self.semifactual_norm_eps) or self.semifactual_norm_eps <= 0:
            raise ValueError("algorithm.semifactual_norm_eps must be finite and positive.")
        if not self.semifactual_enabled:
            return

        adv_estimator = getattr(self.adv_estimator, "value", self.adv_estimator)
        if str(adv_estimator).strip().lower() != "grpo":
            raise ValueError("SCAPO currently requires algorithm.adv_estimator='grpo'.")


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    max_steps: Optional[int] = None
    project_name: str = "easy_r1"
    experiment_name: str = "demo"
    logger: Tuple[str] = ("console", "file")
    nnodes: int = 1
    n_gpus_per_node: int = 8
    max_try_make_batch: int = 20
    critic_warmup: int = 0
    actor_start_step: int = 0
    val_freq: int = -1
    val_before_train: bool = True
    val_only: bool = False
    val_generations_to_log: int = 0
    save_freq: int = -1
    save_limit: int = -1
    save_model_only: bool = False
    save_checkpoint_path: Optional[str] = None
    load_checkpoint_path: Optional[str] = None
    ray_timeline: Optional[str] = None
    find_last_checkpoint: bool = True

    def post_init(self):
        self.actor_start_step = max(self.actor_start_step, self.critic_warmup)
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        self.worker.actor.causal_credit_enabled = self.algorithm.causal_credit_enabled
        self.worker.actor.causal_credit_weighting_mode = self.algorithm.causal_credit_weighting_mode
        self.worker.actor.causal_credit_min_weight = self.algorithm.causal_credit_min_weight
        self.worker.actor.causal_credit_max_weight = self.algorithm.causal_credit_max_weight
        self.worker.actor.causal_credit_answer_weight = self.algorithm.causal_credit_answer_weight
        self.worker.actor.causal_credit_max_spans = self.algorithm.causal_credit_max_spans
        self.worker.actor.causal_credit_span_min_tokens = self.algorithm.causal_credit_span_min_tokens
        self.worker.actor.causal_credit_span_max_tokens = self.algorithm.causal_credit_span_max_tokens
        self.worker.actor.causal_credit_cf_batch_size = self.algorithm.causal_credit_cf_batch_size
        self.worker.actor.causal_credit_mean_normalize = self.algorithm.causal_credit_mean_normalize
        self.worker.actor.causal_credit_skip_zero_advantage = self.algorithm.causal_credit_skip_zero_advantage
        if self.algorithm.causal_credit_enabled:
            if self.worker.actor.loss_type != "default":
                raise ValueError("CF-GRPO requires worker.actor.loss_type='default'.")
            if self.worker.actor.ulysses_size != 1:
                raise ValueError("CF-GRPO currently requires worker.actor.ulysses_size=1.")

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
