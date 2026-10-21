from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "EasyR1"))

from omegaconf import OmegaConf
from verl.trainer.config import PPOConfig

METHODS = ("grpo", "gspo", "sapo", "fipo", "cf_grpo", "scapo")


def build_config(method, size, *, model_path=None, output_dir=None, gpus=8, max_steps=None):
    if method not in METHODS or size not in {"4b", "1.7b"}:
        raise ValueError("Choose a released method and model size 4b or 1.7b")
    data = ROOT / "data"
    steps = max_steps if max_steps is not None else (300 if size == "4b" else 500)
    output = Path(output_dir or ROOT / "outputs" / f"{method}_{size}").resolve()
    model = model_path or f"Qwen/Qwen3-{'4B' if size == '4b' else '1.7B'}-Base"
    val_files = {f"aime{year}": str(data / f"eval/aime{year}.parquet") for year in (2024, 2025, 2026)}
    overrides = {
        "data": {
            "train_files": str(data / "train/dapo17k.parquet"),
            "val_files": val_files,
            "prompt_key": "problem",
            "answer_key": "answer",
            "format_prompt": str(ROOT / "EasyR1/examples/format_prompt/r1.jinja"),
            "max_prompt_length": 1024,
            "max_response_length": 16384,
            "rollout_batch_size": 128,
            "val_batch_size": 1024,
            "shuffle": True,
            "seed": 1,
            "filter_overlong_prompts": False,
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "disable_kl": True,
            "use_kl_loss": False,
            "online_filtering": False,
            "semifactual_enabled": method == "scapo",
            "semifactual_lambda": 0.01,
            "semifactual_n_steps": 60 if size == "4b" else 100,
            "semifactual_norm_eps": 1e-6,
        },
        "worker": {
            "actor": {
                "global_batch_size": 64,
                "micro_batch_size_per_device_for_update": 2,
                "micro_batch_size_per_device_for_experience": 4,
                "max_grad_norm": 1.0,
                "clip_ratio_low": 3e-4 if method == "gspo" else 0.2,
                "clip_ratio_high": 4e-4 if method == "gspo" else 0.28,
                "clip_ratio_dual": 10.0,
                "loss_type": {"grpo": "default", "scapo": "default", "gspo": "gspo_token"}.get(method, method),
                "loss_avg_mode": "token",
                "ppo_epochs": 1,
                "tau_positive": 1.0,
                "tau_negative": 1.05,
                "fipo_decay_rate": 32.0,
                "fipo_influence_clip_low": 0.0,
                "fipo_influence_clip_high": 0.2,
                "fipo_safety_threshold": 10.0,
                "padding_free": True,
                "dynamic_batching": True,
                "ulysses_size": 1,
                "model": {"model_path": model, "trust_remote_code": False, "enable_gradient_checkpointing": True},
                "optim": {
                    "lr": 1e-6,
                    "betas": [0.9, 0.999],
                    "weight_decay": 0.01,
                    "strategy": "adamw",
                    "lr_ramp_ratio": 0.0,
                    "lr_scheduler_type": "constant",
                },
                "fsdp": {"enable_full_shard": True, "enable_cpu_offload": False, "enable_rank0_init": True},
                "offload": {"offload_params": False, "offload_optimizer": False},
            },
            "rollout": {
                "n": 8,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "seed": 1,
                "limit_images": 0,
                "gpu_memory_utilization": 0.8,
                "enforce_eager": False,
                "enable_chunked_prefill": False,
                "tensor_parallel_size": 1,
                "disable_tqdm": True,
                "max_num_batched_tokens": 17408,
                "val_override_config": {
                    name: {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "n": 16,
                    }
                    for name in val_files
                },
            },
            "ref": {
                "fsdp": {"enable_full_shard": True, "enable_cpu_offload": True, "enable_rank0_init": True},
                "offload": {"offload_params": False},
            },
            "reward": {"reward_function": str(ROOT / "EasyR1/examples/reward_function/r1.py") + ":compute_score"},
        },
        "trainer": {
            "total_epochs": 999,
            "max_steps": steps,
            "project_name": "scapo",
            "experiment_name": f"{method}_{size}",
            "logger": ["file"],
            "nnodes": 1,
            "n_gpus_per_node": gpus,
            "val_before_train": True,
            "val_freq": 20,
            "val_generations_to_log": 0,
            "save_freq": 999,
            "save_limit": -1,
            "save_model_only": True,
            "save_checkpoint_path": str(output),
            "find_last_checkpoint": False,
        },
    }
    if method == "scapo":
        overrides["data"]["semifactual_train_file"] = str(data / "train/dapo17k_semifactual.jsonl")
        overrides["data"]["semifactual_unmatched_sample_policy"] = "error"
    return OmegaConf.merge(OmegaConf.structured(PPOConfig()), overrides)


def runtime_versions():
    packages = (
        "torch",
        "vllm",
        "transformers",
        "flash-attn",
        "ray",
        "mathruler",
        "math-verify",
        "tensordict",
        "torchdata",
    )
    result = {"python": sys.version.split()[0]}
    for name in packages:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main():
    parser = argparse.ArgumentParser(description='Paper training recipes and final-checkpoint merging.')
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("size", choices=("4b", "1.7b"))
    parser.add_argument("--model-path", help="Local pretrained model directory or HF model ID")
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1,2,3,4,5,6,7"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, help="Override rollout steps for a smoke run")
    args = parser.parse_args()
    gpu_ids = args.gpus.split(",")
    if any(not x.isdigit() for x in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids) or 512 % (2 * len(gpu_ids)):
        parser.error("GPU ids must be distinct nonnegative integers; 512 must be divisible by twice their count")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    config = build_config(
        args.method,
        args.size,
        model_path=args.model_path,
        output_dir=args.output_dir,
        gpus=len(gpu_ids),
        max_steps=args.max_steps,
    )
    obj = OmegaConf.to_object(config)
    obj.deep_post_init()
    output = Path(config.trainer.save_checkpoint_path)
    if output.exists() and any(p.name not in {"config.yaml", "runtime_versions.json"} for p in output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(OmegaConf.to_container(config, resolve=False, enum_to_str=True), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output / "runtime_versions.json").write_text(json.dumps(runtime_versions(), indent=2) + "\n")
    print(
        f"{args.method} / {args.size}: {config.trainer.max_steps} rollout steps, "
        f"{2 * config.trainer.max_steps} policy updates; config: {config_path}",
        flush=True,
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, TOKENIZERS_PARALLELISM="false")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "EasyR1"), env.get("PYTHONPATH"))))
    subprocess.run([sys.executable, "-m", "verl.trainer.main", f"config={config_path}"], cwd=ROOT, env=env, check=True)
    actor = output / f"global_step_{config.trainer.max_steps}" / "actor"
    subprocess.run(
        [sys.executable, str(ROOT / "EasyR1/scripts/model_merger.py"), "--local_dir", str(actor)], cwd=ROOT, check=True
    )


if __name__ == "__main__":
    main()
