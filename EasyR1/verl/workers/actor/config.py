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
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LoraConfig:
    rank: int = 0
    alpha: int = 64
    target_modules: str = "all-linear"
    exclude_modules: Optional[str] = None

    def post_init(self):
        if not isinstance(self.target_modules, str):
            raise TypeError("lora.target_modules must be a string like 'all-linear' or 'q_proj,k_proj,v_proj,o_proj'.")

        self.target_modules = self.target_modules.strip()
        if self.exclude_modules is not None:
            if not isinstance(self.exclude_modules, str):
                raise TypeError("lora.exclude_modules must be a string like '.*visual.*'.")

            self.exclude_modules = self.exclude_modules.strip()


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    freeze_vision_tower: bool = False
    lora: LoraConfig = field(default_factory=LoraConfig)

    def post_init(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.model_path is not None and os.path.exists(self.model_path):
            self.model_path = os.path.abspath(self.model_path)

        if self.tokenizer_path is not None and os.path.exists(self.tokenizer_path):
            self.tokenizer_path = os.path.abspath(self.tokenizer_path)


@dataclass
class OptimConfig:
    lr: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    strategy: str = "adamw"
    lr_warmup_ratio: float = 0.0
    lr_warmup_steps: Optional[int] = None
    lr_ramp_ratio: float = 0.0
    lr_ramp_steps: Optional[int] = None
    min_lr_ratio: Optional[float] = None
    lr_scheduler_type: str = "constant"
    training_steps: int = field(default=-1, init=False)

    def post_init(self):
        # Preserve EasyR1's warmup configuration names alongside the paper's ramp names.
        if self.lr_warmup_steps is not None:
            self.lr_ramp_steps = self.lr_warmup_steps
        elif self.lr_warmup_ratio != 0.0:
            self.lr_ramp_ratio = self.lr_warmup_ratio


@dataclass
class FSDPConfig:
    enable_full_shard: bool = True
    enable_cpu_offload: bool = False
    enable_rank0_init: bool = True
    use_orig_params: bool = False
    torch_dtype: Optional[str] = None
    fsdp_size: int = -1
    mp_param_dtype: str = "bf16"
    mp_reduce_dtype: str = "fp32"
    mp_buffer_dtype: str = "fp32"


@dataclass
class OffloadConfig:
    offload_params: bool = False
    offload_optimizer: bool = False


@dataclass
class ActorConfig:
    strategy: str = "fsdp"
    global_batch_size: int = 256
    micro_batch_size_per_device_for_update: int = 4
    micro_batch_size_per_device_for_experience: int = 16
    max_grad_norm: float = 1.0
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.3
    clip_ratio_dual: float = 3.0
    loss_avg_mode: str = "token"
    loss_type: str = "default"
    cf_grpo_min_weight: float = 0.5
    cf_grpo_max_weight: float = 4.0
    cf_grpo_answer_weight: float = 1.5
    cf_grpo_max_spans: int = 10
    cf_grpo_span_min_tokens: int = 5
    cf_grpo_span_max_tokens: int = 50
    cf_grpo_batch_size: int = 8
    fipo_decay_rate: float = 32.0
    fipo_influence_clip_low: float = 0.0
    fipo_influence_clip_high: float = 0.2
    fipo_safety_threshold: float = 10.0
    fipo_chunk_size: int = 128
    ppo_epochs: int = 1
    padding_free: bool = True
    dynamic_batching: bool = True
    ulysses_size: int = 1
    use_torch_compile: bool = True
    tau_positive: float = 1.0
    tau_negative: float = 1.05
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    global_batch_size_per_device: int = field(default=-1, init=False)
    disable_kl: bool = field(default=False, init=False)
    use_kl_loss: bool = field(default=False, init=False)
    kl_penalty: str = field(default="kl", init=False)
    kl_coef: float = field(default=0.0, init=False)

    def post_init(self):
        if self.loss_type == "cf_grpo":
            for name in ("min_weight", "max_weight", "answer_weight"):
                value = getattr(self, f"cf_grpo_{name}")
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"worker.actor.cf_grpo_{name} must be finite and positive.")
            if self.cf_grpo_max_weight < self.cf_grpo_min_weight:
                raise ValueError("cf_grpo_max_weight must be >= cf_grpo_min_weight.")
            if self.cf_grpo_max_spans < 0 or self.cf_grpo_batch_size <= 0:
                raise ValueError("cf_grpo_max_spans must be non-negative and cf_grpo_batch_size positive.")
            if not 0 < self.cf_grpo_span_min_tokens <= self.cf_grpo_span_max_tokens:
                raise ValueError("CF-GRPO span lengths must satisfy 0 < min_tokens <= max_tokens.")
            if self.ulysses_size != 1:
                raise ValueError("CF-GRPO currently requires worker.actor.ulysses_size=1.")
            return
        if self.loss_type == "sapo":
            if self.tau_positive <= 0:
                raise ValueError(f"tau_positive must be positive for SAPO, got {self.tau_positive}.")
            if self.tau_negative <= 0:
                raise ValueError(f"tau_negative must be positive for SAPO, got {self.tau_negative}.")
            return
        if self.loss_type != "fipo":
            return
        if self.fipo_decay_rate <= 0:
            raise ValueError(f"fipo_decay_rate must be positive, got {self.fipo_decay_rate}.")
        if not 0 <= self.fipo_influence_clip_low < 1:
            raise ValueError(f"fipo_influence_clip_low must be in [0, 1), got {self.fipo_influence_clip_low}.")
        if self.fipo_influence_clip_high < 0:
            raise ValueError(f"fipo_influence_clip_high must be non-negative, got {self.fipo_influence_clip_high}.")
        if self.fipo_safety_threshold <= 1:
            raise ValueError(f"fipo_safety_threshold must be greater than 1, got {self.fipo_safety_threshold}.")
        if self.fipo_chunk_size <= 0:
            raise ValueError(f"fipo_chunk_size must be positive, got {self.fipo_chunk_size}.")
        if self.clip_ratio_dual <= 1:
            raise ValueError(f"clip_ratio_dual must be greater than 1 for FIPO, got {self.clip_ratio_dual}.")


@dataclass
class RefConfig:
    strategy: str = "fsdp"
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    micro_batch_size_per_device_for_experience: int = field(default=-1, init=False)
    padding_free: bool = field(default=False, init=False)
    dynamic_batching: bool = field(default=False, init=False)
    ulysses_size: int = field(default=1, init=False)
    use_torch_compile: bool = field(default=True, init=False)
