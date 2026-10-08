from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

from gpqa_protocol import extract_mcq, gpqa_samples, prompt_choices

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/eval/gpqa_diamond.csv"


def sample_index(sample):
    return 24 * sample["metadata"]["source_row"] + sample["metadata"]["permutation_id"]


def sample_seed(sample, seed=1):
    return seed + sample_index(sample)


def shard_for(sample, replicas):
    return sample_index(sample) % replicas


def load_samples(data=DATA):
    data = Path(data)
    if not data.is_file():
        raise FileNotFoundError(
            "GPQA-Diamond is not bundled. Download an authorized copy from "
            "https://huggingface.co/datasets/Idavidrein/gpqa and pass --gpqa-file."
        )
    with data.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("GPQA CSV is missing required question/answer columns")
        rows = list(reader)
    if any(not row.get(key) for row in rows for key in required):
        raise ValueError("GPQA CSV contains an empty question or answer")
    if len(rows) != 198:
        raise ValueError("GPQA-Diamond requires 198 original questions")
    samples = list(gpqa_samples(rows))
    if len(samples) != 4752 or len({s["id"] for s in samples}) != 4752:
        raise ValueError("Expected 4,752 unique question/permutation pairs")
    return samples


def prediction_record(sample, prompt, answer, save_generations=False):
    prediction = extract_mcq(answer.text, 4, prompt_choices(sample["prompt"], 4))
    row = {
        "id": sample["id"], "question_id": sample["question_id"],
        "permutation": sample["metadata"]["permutation"], "seed": sample_seed(sample),
        "target": sample["target"], "prediction": prediction,
        "score": float(prediction == sample["target"]),
        "finish_reason": answer.finish_reason, "stop_reason": answer.stop_reason,
        "output_tokens": len(answer.token_ids),
    }
    if save_generations:
        row.update(prompt=prompt, response=answer.text)
    return row


def resolve_dense_model(model_input, num_gpus):
    from eval_vllm_dp import _has_hf_weights, _merge_actor_checkpoint, resolve_model

    model = resolve_model(model_input, num_gpus=num_gpus, auto_merge=True)
    if model.load_checkpoint_path is not None:
        actor = Path(model.load_checkpoint_path) / "actor"
        dense = actor / "huggingface"
        if not _has_hf_weights(dense):
            _merge_actor_checkpoint(actor)
            if not _has_hf_weights(dense):
                raise RuntimeError("The checkpoint merger produced no dense weights")

        model = replace(model, runtime_model_path=str(dense), load_checkpoint_path=None)
    return model


def worker(spec):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(spec["gpus"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(spec["model"], trust_remote_code=False)
    model_config = AutoConfig.from_pretrained(spec["model"], trust_remote_code=False)
    context = getattr(model_config, "max_position_embeddings", None)
    if not context:
        raise ValueError("Cannot infer model context length")
    engine = LLM(
        model=spec["model"],
        tokenizer=spec["model"],
        dtype="auto",
        tensor_parallel_size=len(spec["gpus"]),
        max_model_len=context,
        gpu_memory_utilization=0.85,
        seed=1,
        trust_remote_code=False,
        max_num_seqs=32,
        enforce_eager=False,
    )
    samples = [s for s in load_samples(spec["gpqa_file"]) if shard_for(s, spec["replicas"]) == spec["rank"]]
    with Path(spec["output"]).open("w", encoding="utf-8") as handle:
        for start in range(0, len(samples), 32):
            batch = samples[start : start + 32]
            rendered, prompts, params = [], [], []
            for sample in batch:
                prompt = sample["prompt"]
                if tokenizer.chat_template:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                    )
                tokens = tokenizer.encode(prompt, add_special_tokens=not bool(tokenizer.chat_template))
                if len(tokens) + 16384 > context:
                    raise ValueError(
                        f"Context overflow for {sample['id']}; the protocol does not truncate GPQA prompts"
                    )
                prompts.append({"prompt_token_ids": tokens})
                rendered.append(prompt)
                params.append(
                    SamplingParams(temperature=0.7, top_p=0.9, max_tokens=16384, n=1, seed=sample_seed(sample))
                )
            predictions = engine.generate(prompts, sampling_params=params, use_tqdm=False)
            for sample, prompt, prediction in zip(batch, rendered, predictions, strict=True):
                if len(prediction.outputs) != 1:
                    raise RuntimeError("Expected one generation per permutation")
                answer = prediction.outputs[0]
                row = prediction_record(sample, prompt, answer, spec["save_generations"])
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"GPQA rank {spec['rank']}: {min(start + 32, len(samples))}/{len(samples)}", flush=True)


def summarize(output, data=DATA):
    rows = [json.loads(line) for p in sorted(output.glob("rank-*.jsonl")) for line in p.read_text().splitlines()]
    expected = {s["id"] for s in load_samples(data)}
    if len(rows) != 4752 or {r["id"] for r in rows} != expected:
        raise ValueError("Incomplete or duplicated GPQA generations")
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row["score"])
    if len(by_question) != 198 or any(len(v) != 24 for v in by_question.values()):
        raise ValueError("Each of the 198 questions must have exactly 24 permutations")
    score = sum(sum(v) / 24 for v in by_question.values()) / 198
    result = {
        "benchmark": "gpqa_diamond",
        "accuracy": score,
        "score_percent": 100 * score,
        "num_questions": 198,
        "num_generations": 4752,
        "permutations": 24,
        "parse_failures": sum(r["prediction"] is None for r in rows),
        "truncated": sum(r["finish_reason"] == "length" for r in rows),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate all 24 option permutations of GPQA-Diamond with vLLM.')
    parser.add_argument("--model")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpqa-file", type=Path, default=DATA)
    parser.add_argument("--save-generations", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker.read_text()))
        return
    if not args.model or not args.output_dir:
        parser.error("--model and --output-dir are required")
    gpu_ids = args.gpus.split(",")
    if any(not item.isdigit() for item in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error("GPU ids must be distinct nonnegative integers")
    if args.tensor_parallel_size <= 0 or len(gpu_ids) % args.tensor_parallel_size:
        parser.error("GPU count must be divisible by tensor parallel size")
    data = args.gpqa_file.expanduser().resolve()
    samples = load_samples(data)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty GPQA output directory")
    model = resolve_dense_model(args.model, len(gpu_ids))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": model.runtime_model_path,
        "questions": 198,
        "inputs": len(samples),
        "samples_per_permutation": 1,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 16384,
        "seed": 1,
        "save_generations": args.save_generations,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    groups = [gpu_ids[i : i + args.tensor_parallel_size] for i in range(0, len(gpu_ids), args.tensor_parallel_size)]
    processes = []
    try:
        for rank, group in enumerate(groups):
            spec = {
                "model": model.runtime_model_path,
                "gpus": group,
                "rank": rank,
                "replicas": len(groups),
                "output": str(output / f"rank-{rank:03}.jsonl"),
                "gpqa_file": str(data),
                "save_generations": args.save_generations,
            }
            path = output / f"worker-{rank:03}.json"
            path.write_text(json.dumps(spec))
            processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(path)]))
        failures = [p.wait() for p in processes]
        if any(failures):
            raise RuntimeError(f"GPQA worker exit codes: {failures}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
    summarize(output, data)


if __name__ == "__main__":
    main()
