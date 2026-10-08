from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TEST_DATA_ROOT = REPO_ROOT / "data" / "eval"

DATASET_REGISTRY = {
    "aime2024": "aime2024.parquet",
    "aime2025": "aime2025.parquet",
    "aime2026": "aime2026.parquet",
    "amc23": "amc23.parquet",
    "amc2024": "amc2024.parquet",
    "amc2025": "amc2025.parquet",
    "minerva": "minerva.parquet",
    "olympiadbench": "olympiadbench.parquet",
    "omni_math_rule": "omni_math_rule.parquet",
    "hmmt_feb_2025": "hmmt_feb_2025.parquet",
    "hmmt_feb_2026": "hmmt_feb_2026.parquet",
    "brumo_2025": "brumo_2025.parquet",
    "smt_2025": "smt_2025.parquet",
    "noop_aime24": "noop_aime2024.parquet",
    "thinkbench_aime2024": "thinkbench_aime2024.parquet",
    "noop_aime25": "noop_aime2025.parquet",
    "thinkbench_aime2025": "thinkbench_aime2025.parquet",
    "noop_aime26": "noop_aime2026.parquet",
    "thinkbench_aime2026": "thinkbench_aime2026.parquet",
}

DATASET_ALIASES = {
    "aime24": "aime2024",
    "aime25": "aime2025",
    "aime26": "aime2026",
    "amc24": "amc2024",
    "amc25": "amc2025",
    "hmmt25": "hmmt_feb_2025",
    "hmmt26": "hmmt_feb_2026",
    "brumo25": "brumo_2025",
    "smt25": "smt_2025",
    "omni_math": "omni_math_rule",
}

HF_WEIGHT_PATTERNS = (
    "*.safetensors",
    "pytorch_model*.bin",
    "model*.bin",
)


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    n: int
    rows: int


@dataclasses.dataclass(frozen=True)
class ModelResolution:
    input_path: str
    runtime_model_path: str
    load_checkpoint_path: str | None
    source_kind: str
    source_config_path: str | None
    source_config: dict[str, Any] | None
    checkpoint_world_size: int | None
    merge_performed: bool


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def _temperature(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _float_0_1(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def _canonical_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    return DATASET_ALIASES.get(normalized, normalized)


def _split_optional_n(value: str, default: int = 1) -> tuple[str, int]:
    head, separator, tail = value.rpartition(":")
    if separator and tail.isdigit():
        n = int(tail)
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        return head, n

    return value, default


def _expand_raw_specs(raw_specs: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for raw in raw_specs:
        expanded.extend(item.strip() for item in raw.split(",") if item.strip())
    return expanded


def _dataset_row_count_and_schema(path: Path) -> tuple[int, set[str]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to inspect parquet test sets") from exc

        parquet_file = pq.ParquetFile(path)
        return parquet_file.metadata.num_rows, set(parquet_file.schema.names)

    if suffix in {".json", ".jsonl"}:
        rows = 0
        columns: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            if suffix == ".json":
                payload = json.load(handle)
                if not isinstance(payload, list):
                    raise ValueError(f"JSON test set must contain a list: {path}")
                rows = len(payload)
                if payload:
                    columns = set(payload[0])
            else:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if rows == 0:
                        columns = set(item)
                    rows += 1
        return rows, columns

    raise ValueError(f"Unsupported test-set format '{suffix}': {path}")


def parse_dataset_specs(raw_specs: Iterable[str]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    seen: set[str] = set()

    for raw in _expand_raw_specs(raw_specs):
        if "=" in raw:
            raw_name, rhs = raw.split("=", 1)
            canonical_name = _canonical_dataset_name(raw_name)
            if rhs.isdigit() and canonical_name in DATASET_REGISTRY:
                path = TEST_DATA_ROOT / DATASET_REGISTRY[canonical_name]
                n = int(rhs)
            else:
                raw_path, n = _split_optional_n(rhs)
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = REPO_ROOT / path
        else:
            raw_name, n = _split_optional_n(raw)
            canonical_name = _canonical_dataset_name(raw_name)
            if canonical_name not in DATASET_REGISTRY:
                raise ValueError(
                    f"Unknown registered test set '{raw_name}'. For a custom file use NAME=data/eval/custom.parquet:N."
                )
            path = TEST_DATA_ROOT / DATASET_REGISTRY[canonical_name]

        canonical_name = _canonical_dataset_name(raw_name)
        if not re.fullmatch(r"[a-z0-9_.-]+", canonical_name):
            raise ValueError(f"Invalid dataset name '{canonical_name}'. Use only letters, digits, '.', '_' or '-'.")
        if n < 1:
            raise ValueError(f"n must be >= 1 for {canonical_name}, got {n}")
        if canonical_name in seen:
            raise ValueError(f"Duplicate test-set name: {canonical_name}")

        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Test set does not exist: {path}")
        rows, columns = _dataset_row_count_and_schema(path)
        missing = {"problem", "answer"} - columns
        if missing:
            raise ValueError(f"Test set {path} is missing columns: {sorted(missing)}")
        if rows == 0:
            raise ValueError(f"Test set is empty: {path}")

        specs.append(DatasetSpec(name=canonical_name, path=path, n=n, rows=rows))
        seen.add(canonical_name)

    if not specs:
        raise ValueError("At least one --test-set is required.")
    return specs


def _has_hf_weights(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    return any(any(path.glob(pattern)) for pattern in HF_WEIGHT_PATTERNS)


def _checkpoint_world_size(actor_dir: Path) -> int | None:
    sizes = set()
    for path in actor_dir.glob("model_world_size_*_rank_*.pt"):
        match = re.fullmatch(r"model_world_size_(\d+)_rank_(\d+)\.pt", path.name)
        if match:
            sizes.add(int(match.group(1)))
    if not sizes:
        return None
    if len(sizes) != 1:
        raise ValueError(f"Actor checkpoint contains inconsistent world sizes: {sorted(sizes)}")
    return sizes.pop()


def _has_complete_checkpoint_state(actor_dir: Path, world_size: int) -> bool:
    for rank in range(world_size):
        if not (actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt").is_file():
            return False
        if not (actor_dir / f"optim_world_size_{world_size}_rank_{rank}.pt").is_file():
            return False
        if not (actor_dir / f"extra_state_world_size_{world_size}_rank_{rank}.pt").is_file():
            return False
    return True


def _load_config_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return payload


def _find_source_config(step_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = (
        step_dir.parent / "config.yaml",
        step_dir.parent / "experiment_config.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate, _load_config_file(candidate)
    return None, None


def _nested_get(mapping: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _checkpoint_step_and_actor(path: Path) -> tuple[Path | None, Path | None]:
    if path.name.startswith("global_step_") and (path / "actor").is_dir():
        return path, path / "actor"
    if path.name == "actor" and path.parent.name.startswith("global_step_"):
        return path.parent, path
    if (
        path.name == "huggingface"
        and path.parent.name == "actor"
        and path.parent.parent.name.startswith("global_step_")
    ):
        return path.parent.parent, path.parent
    return None, None


def _resolve_tracker_root(path: Path) -> Path:
    tracker_path = path / "checkpoint_tracker.json"
    if not tracker_path.is_file():
        return path

    with tracker_path.open(encoding="utf-8") as handle:
        tracker = json.load(handle)
    last_actor = tracker.get("last_actor_path")
    if last_actor and Path(last_actor).is_dir():
        return Path(last_actor).resolve()
    last_step = tracker.get("last_global_step")
    if last_step is None:
        raise ValueError(f"No last checkpoint recorded in {tracker_path}")
    candidate = path / f"global_step_{last_step}"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Tracked checkpoint does not exist: {candidate}")
    return candidate.resolve()


def _merge_actor_checkpoint(actor_dir: Path) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "EasyR1" / "scripts" / "model_merger.py"),
        "--local_dir",
        str(actor_dir),
    ]
    print("Merging checkpoint for vLLM: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def resolve_model(
    model_input: str,
    *,
    num_gpus: int,
    auto_merge: bool,
) -> ModelResolution:
    expanded = Path(model_input).expanduser()
    if not expanded.exists():
        if expanded.is_absolute():
            raise FileNotFoundError(f"Model path does not exist: {expanded}")
        return ModelResolution(
            input_path=model_input,
            runtime_model_path=model_input,
            load_checkpoint_path=None,
            source_kind="huggingface_repo",
            source_config_path=None,
            source_config=None,
            checkpoint_world_size=None,
            merge_performed=False,
        )

    path = _resolve_tracker_root(expanded.resolve())
    if not path.is_dir():
        raise ValueError(f"Model must be a directory or Hugging Face repo id: {path}")

    if _has_hf_weights(path):
        step_dir, _ = _checkpoint_step_and_actor(path)
        source_config_path, source_config = _find_source_config(step_dir) if step_dir else (None, None)
        return ModelResolution(
            input_path=model_input,
            runtime_model_path=str(path),
            load_checkpoint_path=None,
            source_kind="huggingface_model",
            source_config_path=str(source_config_path) if source_config_path else None,
            source_config=source_config,
            checkpoint_world_size=None,
            merge_performed=False,
        )

    step_dir, actor_dir = _checkpoint_step_and_actor(path)
    if step_dir is None or actor_dir is None:
        raise ValueError(
            f"No Hugging Face weights or recognized verl checkpoint found at {path}. "
            "Pass a HF model directory, global_step_*, global_step_*/actor, or actor/huggingface."
        )

    world_size = _checkpoint_world_size(actor_dir)
    if world_size is None:
        raise ValueError(f"No actor model shards found in {actor_dir}")
    source_config_path, source_config = _find_source_config(step_dir)
    bootstrap_model = _nested_get(source_config, "worker", "actor", "model", "model_path")
    bootstrap_available = bool(bootstrap_model)
    if bootstrap_available and Path(str(bootstrap_model)).is_absolute():
        bootstrap_available = Path(str(bootstrap_model)).exists()

    complete_state = _has_complete_checkpoint_state(actor_dir, world_size)
    if complete_state and world_size == num_gpus and bootstrap_available:
        return ModelResolution(
            input_path=model_input,
            runtime_model_path=str(bootstrap_model),
            load_checkpoint_path=str(step_dir),
            source_kind="verl_checkpoint",
            source_config_path=str(source_config_path) if source_config_path else None,
            source_config=source_config,
            checkpoint_world_size=world_size,
            merge_performed=False,
        )

    hf_path = actor_dir / "huggingface"
    if _has_hf_weights(hf_path):
        return ModelResolution(
            input_path=model_input,
            runtime_model_path=str(hf_path),
            load_checkpoint_path=None,
            source_kind="merged_verl_checkpoint",
            source_config_path=str(source_config_path) if source_config_path else None,
            source_config=source_config,
            checkpoint_world_size=world_size,
            merge_performed=False,
        )

    reasons = []
    if world_size != num_gpus:
        reasons.append(f"checkpoint world_size={world_size}, requested GPUs={num_gpus}")
    if not complete_state:
        reasons.append("optimizer/extra-state shards are absent")
    if not bootstrap_available:
        reasons.append("the original bootstrap model path is unavailable")
    reason = "; ".join(reasons) or "dense HF weights are absent"
    if not auto_merge:
        raise ValueError(
            f"Checkpoint must be merged before this evaluation ({reason}). Re-run without --no-auto-merge."
        )

    _merge_actor_checkpoint(actor_dir)
    if not _has_hf_weights(hf_path):
        raise RuntimeError(f"Checkpoint merger finished but no HF weights were created in {hf_path}")
    return ModelResolution(
        input_path=model_input,
        runtime_model_path=str(hf_path),
        load_checkpoint_path=None,
        source_kind="merged_verl_checkpoint",
        source_config_path=str(source_config_path) if source_config_path else None,
        source_config=source_config,
        checkpoint_world_size=world_size,
        merge_performed=True,
    )


def _resolve_repo_path(value: str | None, fallback: Path) -> Path:
    if not value:
        return fallback.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _model_config(
    resolution: ModelResolution,
    *,
    trust_remote_code: bool | None,
) -> dict[str, Any]:
    if resolution.load_checkpoint_path and resolution.source_config:
        source_model = _nested_get(resolution.source_config, "worker", "actor", "model", default={})
        model_config = copy.deepcopy(source_model) if isinstance(source_model, dict) else {}
    else:
        model_config = {}

        model_config["lora"] = {"rank": 0, "alpha": 64, "target_modules": "all-linear", "exclude_modules": None}

    model_config["model_path"] = resolution.runtime_model_path
    model_config["tokenizer_path"] = resolution.runtime_model_path
    model_config["enable_gradient_checkpointing"] = False
    model_config["freeze_vision_tower"] = False
    if trust_remote_code is not None:
        model_config["trust_remote_code"] = trust_remote_code
    else:
        model_config.setdefault("trust_remote_code", False)
    return model_config


def build_eval_config(
    args: argparse.Namespace,
    datasets: list[DatasetSpec],
    resolution: ModelResolution,
    output_dir: Path,
    gpu_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = resolution.source_config
    source_data = _nested_get(source, "data", default={})
    if not isinstance(source_data, dict):
        source_data = {}

    max_prompt_length = args.max_prompt_length or int(source_data.get("max_prompt_length", 1024))
    max_response_length = args.max_response_length or int(source_data.get("max_response_length", 16384))
    val_batch_size = args.val_batch_size or int(source_data.get("val_batch_size", 1024))
    format_prompt = _resolve_repo_path(
        args.format_prompt or source_data.get("format_prompt"),
        REPO_ROOT / "EasyR1" / "examples" / "format_prompt" / "r1.jinja",
    )
    if not format_prompt.is_file():
        raise FileNotFoundError(f"Prompt template does not exist: {format_prompt}")

    reward_function_value = args.reward_function
    reward_function = _resolve_repo_path(reward_function_value, SCRIPT_DIR / "math_verify_verifier.py")
    reward_entrypoint = str(reward_function)
    if reward_function_value and ":" in str(reward_function_value):
        raw_path, function_name = str(reward_function_value).rsplit(":", 1)
        reward_path = _resolve_repo_path(raw_path, reward_function)
        reward_entrypoint = f"{reward_path}:{function_name}"
    else:
        reward_entrypoint = f"{reward_function}:compute_score"
    if not Path(reward_entrypoint.rsplit(":", 1)[0]).is_file():
        raise FileNotFoundError(f"Reward function does not exist: {reward_entrypoint}")

    tensor_parallel_size = args.tensor_parallel_size
    if len(gpu_ids) % tensor_parallel_size != 0:
        raise ValueError(
            f"GPU count ({len(gpu_ids)}) must be divisible by tensor parallel size ({tensor_parallel_size})."
        )
    data_parallel_size = len(gpu_ids) // tensor_parallel_size

    source_rollout = _nested_get(source, "worker", "rollout", default={})
    if not isinstance(source_rollout, dict):
        source_rollout = {}
    seed = args.seed if args.seed is not None else int(source_rollout.get("seed", 1))
    gpu_memory_utilization = (
        args.gpu_memory_utilization
        if args.gpu_memory_utilization is not None
        else float(source_rollout.get("gpu_memory_utilization", 0.8))
    )

    val_files = {spec.name: str(spec.path) for spec in datasets}
    val_override_config = {
        spec.name: {"temperature": args.temperature, "top_p": args.top_p, "n": spec.n} for spec in datasets
    }
    base_n = max(2, max(spec.n for spec in datasets))

    source_fsdp = _nested_get(source, "worker", "actor", "fsdp", default={})
    fsdp_config = copy.deepcopy(source_fsdp) if isinstance(source_fsdp, dict) else {}
    fsdp_config.setdefault("enable_full_shard", True)
    fsdp_config.setdefault("enable_cpu_offload", False)
    fsdp_config.setdefault("enable_rank0_init", True)

    fsdp_config.setdefault("torch_dtype", "bf16")

    run_name = output_dir.name
    val_generations_to_log = max(spec.rows * spec.n for spec in datasets) if args.save_generations else 0
    config = {
        "data": {
            "train_files": str(datasets[0].path),
            "val_files": val_files,
            "prompt_key": "problem",
            "answer_key": "answer",
            "image_key": "images",
            "video_key": "videos",
            "image_dir": None,
            "video_fps": 2.0,
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
            "rollout_batch_size": len(gpu_ids),
            "mini_rollout_batch_size": None,
            "val_batch_size": val_batch_size,
            "format_prompt": str(format_prompt),
            "override_chat_template": source_data.get("override_chat_template"),
            "shuffle": False,
            "seed": seed,
            "min_pixels": int(source_data.get("min_pixels", 262144)),
            "max_pixels": int(source_data.get("max_pixels", 4194304)),
            "filter_overlong_prompts": bool(source_data.get("filter_overlong_prompts", False)),
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "disable_kl": True,
            "use_kl_loss": False,
            "online_filtering": False,
            "semifactual_enabled": False,
        },
        "worker": {
            "actor": {
                "global_batch_size": len(gpu_ids),
                "micro_batch_size_per_device_for_update": 1,
                "micro_batch_size_per_device_for_experience": 1,
                "padding_free": True,
                "dynamic_batching": True,
                "ulysses_size": 1,
                "model": _model_config(resolution, trust_remote_code=args.trust_remote_code),
                "optim": {"lr": 1.0e-6, "weight_decay": 1.0e-2, "strategy": "adamw"},
                "fsdp": fsdp_config,
                "offload": {"offload_params": True, "offload_optimizer": False},
            },
            "rollout": {
                "n": base_n,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": seed,
                "limit_images": 0,
                "gpu_memory_utilization": gpu_memory_utilization,
                "enforce_eager": args.enforce_eager,
                "enable_chunked_prefill": args.enable_chunked_prefill,
                "tensor_parallel_size": tensor_parallel_size,
                "disable_tqdm": args.disable_tqdm,
                "val_override_config": val_override_config,
                "max_num_batched_tokens": max_prompt_length + max_response_length,
            },
            "ref": {
                "fsdp": {"enable_full_shard": True, "enable_cpu_offload": True, "enable_rank0_init": True},
                "offload": {"offload_params": False},
            },
            "reward": {"reward_function": reward_entrypoint},
        },
        "trainer": {
            "total_epochs": 1,
            "max_steps": 1,
            "project_name": "scapo-eval",
            "experiment_name": run_name,
            "logger": ["file"],
            "nnodes": 1,
            "n_gpus_per_node": len(gpu_ids),
            "val_freq": -1,
            "val_before_train": True,
            "val_only": True,
            "val_generations_to_log": val_generations_to_log,
            "save_freq": -1,
            "save_limit": -1,
            "save_model_only": False,
            "save_checkpoint_path": str(output_dir),
            "load_checkpoint_path": resolution.load_checkpoint_path,
            "find_last_checkpoint": False,
        },
    }

    derived = {
        "data_parallel_size": data_parallel_size,
        "tensor_parallel_size": tensor_parallel_size,
        "max_prompt_length": max_prompt_length,
        "max_response_length": max_response_length,
        "format_prompt": str(format_prompt),
        "reward_function": reward_entrypoint,
        "seed": seed,
        "gpu_memory_utilization": gpu_memory_utilization,
        "val_generations_to_log": val_generations_to_log,
    }
    return config, derived


def _atomic_json_dump(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _flatten_metrics(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        child_prefix = f"{prefix}/{key}" if prefix else str(key)
        flattened.update(_flatten_metrics(item, child_prefix))
    return flattened


def _read_eval_metrics(log_path: Path) -> tuple[int, dict[str, Any], dict[str, Any]]:
    entries = []
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    if not entries:
        raise RuntimeError(f"No evaluation metrics found in {log_path}")
    payload = entries[-1]
    step = int(payload.get("step", 0))
    nested_metrics = {key: value for key, value in payload.items() if key != "step"}
    return step, nested_metrics, _flatten_metrics(nested_metrics)


def _dataset_results(datasets: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = {}
    for dataset in datasets:
        prefix = f"val/{dataset['name']}/"
        score = metrics.get(prefix + "math_verify_accuracy_reward")
        if score is None:
            score = metrics.get(prefix + "accuracy_reward")
        results[dataset["name"]] = {
            "mean_accuracy": float(score) if score is not None else None,
            "rows": dataset["rows"],
            "n": dataset["n"],
            "samples": dataset["rows"] * dataset["n"],
        }
    return results


def _print_result_table(summary: dict[str, Any]) -> None:
    print("\nEvaluation results")
    print(f"{'dataset':<24} {'rows':>6} {'n':>4} {'generations':>12} {'Math-Verify':>12}")
    print("-" * 68)
    metrics = summary["metrics"]
    for dataset in summary["datasets"]:
        prefix = f"val/{dataset['name']}/"
        math_verify = metrics.get(prefix + "math_verify_accuracy_reward")
        if math_verify is None:
            math_verify = metrics.get(prefix + "accuracy_reward")
        math_verify_text = f"{float(math_verify):.6f}" if math_verify is not None else "N/A"
        print(
            f"{dataset['name']:<24} {dataset['rows']:>6} {dataset['n']:>4} "
            f"{dataset['rows'] * dataset['n']:>12} {math_verify_text:>12}"
        )
    overall = metrics.get("val/overall_reward")
    if overall is not None:
        print("-" * 68)
        print(f"{'val/overall_reward (Math-Verify)':<50} {float(overall):>18.6f}")


def _default_gpu_ids() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return visible or "0,1,2,3,4,5,6,7"


def _parse_gpu_ids(value: str) -> list[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"--gpus contains duplicates: {value}")
    return gpu_ids


def _safe_model_name(model: str) -> str:
    value = Path(model.rstrip("/")).name or "model"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a HF/verl checkpoint with the same val-only vLLM path used during training. "
            "The default reward scores each rollout with Math-Verify only. "
            "Registered syntax: --test-set aime2024=16. Custom syntax: "
            "--test-set my_set=data/eval/custom.parquet:4."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", help="HF model/repo, experiment root, global_step_*, actor, or actor/huggingface")
    parser.add_argument(
        "--test-set",
        "--test-sets",
        action="append",
        default=[],
        help="Repeatable test-set spec; comma-separated specs are also accepted",
    )
    parser.add_argument("--temperature", type=_temperature, default=0.7)
    parser.add_argument("--top-p", type=_probability, default=0.9)
    parser.add_argument("--gpus", default=_default_gpu_ids(), help="Physical GPU ids exposed to the eval process")
    parser.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the source training config, then 1")
    parser.add_argument("--max-prompt-length", type=_positive_int, default=None)
    parser.add_argument("--max-response-length", type=_positive_int, default=None)
    parser.add_argument("--val-batch-size", type=_positive_int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=_float_0_1, default=None)
    parser.add_argument(
        "--format-prompt", default=None,
        help="Defaults to source config, then EasyR1/examples/format_prompt/r1.jinja",
    )
    parser.add_argument(
        "--reward-function",
        default=None,
        help=(
            "PATH:FUNCTION; defaults to eval/math_verify_verifier.py:compute_score. Pass a different "
            "entrypoint only for an explicit legacy or diagnostic evaluation"
        ),
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--auto-merge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--disable-tqdm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-generations", action="store_true", help="Save all evaluated prompts and responses")
    parser.add_argument("--list-test-sets", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_test_sets:
        for name, filename in DATASET_REGISTRY.items():
            print(f"{name:<24} {TEST_DATA_ROOT / filename}")
        return 0
    if not args.model:
        parser.error("--model is required unless --list-test-sets is used")

    try:
        gpu_ids = _parse_gpu_ids(args.gpus)
        datasets = parse_dataset_specs(args.test_set)
        resolution = resolve_model(
            args.model,
            num_gpus=len(gpu_ids),
            auto_merge=args.auto_merge,
        )

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else REPO_ROOT / "outputs" / "eval" / f"{_safe_model_name(args.model)}_{timestamp}"
        )
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        output_dir = output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already contains an evaluation: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        config, derived = build_eval_config(args, datasets, resolution, output_dir, gpu_ids)
        config_path = output_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

        manifest = {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "experiment": output_dir.name,
            "verifier": derived["reward_function"],
            "repo_root": str(REPO_ROOT),
            "model": dataclasses.asdict(resolution) | {"source_config": None},
            "datasets": [
                {"name": spec.name, "path": str(spec.path), "n": spec.n, "rows": spec.rows} for spec in datasets
            ],
            "sampling": {"temperature": args.temperature, "top_p": args.top_p},
            "gpus": gpu_ids,
            "parallelism": {
                "data_parallel_size": derived["data_parallel_size"],
                "tensor_parallel_size": derived["tensor_parallel_size"],
            },
            "resolved": derived,
            "config_path": str(config_path),
        }
        _atomic_json_dump(output_dir / "manifest.json", manifest)

        print(f"Model input:      {args.model}")
        print(f"Runtime model:    {resolution.runtime_model_path}")
        print(f"Checkpoint load:  {resolution.load_checkpoint_path or 'none (HF weights)'}")
        print(
            f"Parallelism:      DP={derived['data_parallel_size']}, TP={derived['tensor_parallel_size']}, "
            f"GPUs={','.join(gpu_ids)}"
        )
        print(f"Sampling:         temperature={args.temperature}, top_p={args.top_p}")
        for spec in datasets:
            print(f"Test set:         {spec.name} rows={spec.rows} n={spec.n} path={spec.path}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO_ROOT / "EasyR1"), env.get("PYTHONPATH"))))
        env.setdefault("PYTHONUNBUFFERED", "1")
        command = [sys.executable, "-m", "verl.trainer.main", f"config={config_path}"]
        print("Launching: " + " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

        metrics_log = output_dir / "experiment_log.jsonl"
        step, nested_metrics, flat_metrics = _read_eval_metrics(metrics_log)
        summary = {
            "experiment": manifest["experiment"],
            "created_at": manifest["created_at"],
            "verifier": manifest["verifier"],
            "model": manifest["model"],
            "datasets": manifest["datasets"],
            "sampling": manifest["sampling"],
            "parallelism": manifest["parallelism"],
            "step": step,
            "metrics": flat_metrics,
            "nested_metrics": nested_metrics,
            "results": _dataset_results(manifest["datasets"], flat_metrics),
            "artifacts": {
                "config": str(config_path),
                "manifest": str(output_dir / "manifest.json"),
                "experiment_log": str(metrics_log),
                "val_generations": (
                    str(output_dir / "val_generations.jsonl") if derived["val_generations_to_log"] > 0 else None
                ),
            },
        }
        summary_path = output_dir / "summary.json"
        _atomic_json_dump(summary_path, summary)
        _print_result_table(summary)
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
