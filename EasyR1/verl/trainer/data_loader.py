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

from collections.abc import Mapping
from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


def create_dataloader(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
) -> tuple[StatefulDataLoader, dict[str, StatefulDataLoader]]:
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        semifactual_file=config.semifactual_train_file,
        semifactual_unmatched_sample_policy=config.semifactual_unmatched_sample_policy,
    )

    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataloaders = _create_val_dataloaders(config, tokenizer, processor)

    assert len(train_dataloader) >= 1
    for name, val_dataloader in val_dataloaders.items():
        assert len(val_dataloader) >= 1, f"Validation dataloader '{name}' is empty"

    print(f"Size of train dataloader: {len(train_dataloader)}")
    for name, val_dataloader in val_dataloaders.items():
        print(f"Size of val dataloader '{name}': {len(val_dataloader)}")

    return train_dataloader, val_dataloaders


def _create_val_dataloaders(
    config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]
) -> dict[str, StatefulDataLoader]:
    if isinstance(config.val_files, str):
        val_files = {"default": config.val_files}
    elif isinstance(config.val_files, Mapping):
        val_files = dict(config.val_files)
    else:
        raise ValueError(f"val_files must be str or dict, got {type(config.val_files)}")

    if isinstance(config.val_batch_size, (int, float)):
        val_batch_sizes = {name: int(config.val_batch_size) for name in val_files}
    elif isinstance(config.val_batch_size, Mapping):
        val_batch_sizes = dict(config.val_batch_size)
    else:
        raise ValueError(f"val_batch_size must be int or dict, got {type(config.val_batch_size)}")

    val_dataloaders = {}
    for name, data_path in val_files.items():
        val_dataset = RLHFDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
        )

        batch_size = val_batch_sizes.get(name, val_batch_sizes.get("default", -1))
        val_batch_size = len(val_dataset) if batch_size == -1 else int(batch_size)
        val_dataloaders[str(name)] = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

    return val_dataloaders
