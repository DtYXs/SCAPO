from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from gpqa_eval import DATA as DEFAULT_GPQA_FILE, load_samples

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MATH = json.loads((HERE / "benchmarks.json").read_text())
GROUPS = {
    "AIME 24-26": ["aime2024", "aime2025", "aime2026"],
    "AMC 23-25": ["amc2023", "amc2024", "amc2025"],
    "HMMT 25-26": ["hmmt2025", "hmmt2026"],
    "BRUMO 25": ["brumo_2025"],
    "SMT 25": ["smt_2025"],
    "Omni-Math": ["omni_math_rule"],
    "Minerva": ["minerva"],
    "Olympiad": ["olympiadbench"],
    "NoOp-AIME 24-26": ["noop_aime2024", "noop_aime2025", "noop_aime2026"],
    "ThinkBench-AIME 24-26": ["thinkbench_aime2024", "thinkbench_aime2025", "thinkbench_aime2026"],
}


def paper_table(math_summary, gpqa_summary=None):
    scores = {}
    weights = {d["name"]: d["rows"] * d["n"] for d in math_summary["datasets"]}
    for label, datasets in GROUPS.items():
        values = [math_summary["metrics"][f"val/{name}/math_verify_accuracy_reward"] for name in datasets]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError(f"Invalid accuracy in {label}")
        if any(weights[name] <= 0 for name in datasets):
            raise ValueError(f"Invalid sample count in {label}")
        scores[label] = sum(value * weights[name] for value, name in zip(values, datasets)) / sum(
            weights[name] for name in datasets
        )
    scores["ID Avg."] = sum(scores[k] for k in list(GROUPS)[:8]) / 8
    if gpqa_summary is not None:
        scores["GPQA-Diamond"] = gpqa_summary["accuracy"]
        scores["OOD Avg."] = (scores["NoOp-AIME 24-26"] + scores["ThinkBench-AIME 24-26"] + scores["GPQA-Diamond"]) / 3
    order = list(GROUPS)[:8] + ["ID Avg."] + list(GROUPS)[8:]
    if gpqa_summary is not None:
        order += ["GPQA-Diamond", "OOD Avg."]
    return {name: scores[name] for name in order}


def run_logged(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description='Evaluate available paper benchmarks, or a single benchmark.')
    parser.add_argument("--model", required=True, help="HF model ID, merged model, or training checkpoint directory")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--benchmark", choices=(*MATH, "gpqa"),
                           help="Evaluate one benchmark (default: all available paper benchmarks)")
    selection.add_argument("--suite", choices=("all", "math", "gpqa"), default="all", help=argparse.SUPPRESS)
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1,2,3,4,5,6,7"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, help="New directory for persistent evaluation results")
    parser.add_argument("--gpqa-file", type=Path, default=DEFAULT_GPQA_FILE,
                        help="Locally downloaded, authorized GPQA-Diamond CSV")
    parser.add_argument("--save-generations", action="store_true", help="Save full prompts and responses")
    args = parser.parse_args()
    ids = args.gpus.split(",")
    if any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids):
        parser.error("--gpus must list distinct nonnegative integers")
    if args.tensor_parallel_size <= 0 or len(ids) % args.tensor_parallel_size:
        parser.error("GPU count must be divisible by tensor parallel size")
    suite = args.suite if args.benchmark is None else ("gpqa" if args.benchmark == "gpqa" else "math")
    math_benchmarks = MATH if suite != "gpqa" else {}
    if args.benchmark in MATH:
        math_benchmarks = {args.benchmark: MATH[args.benchmark]}
    include_gpqa = suite in {"all", "gpqa"}
    gpqa_file = args.gpqa_file.expanduser().resolve()
    if include_gpqa and suite == "all" and not gpqa_file.exists():
        print(f"Skipping GPQA-Diamond: {gpqa_file} not found.", flush=True)
        include_gpqa = False
        suite = "math"
    if include_gpqa:
        load_samples(gpqa_file)
    benchmarks = list(math_benchmarks) + (["gpqa"] if include_gpqa else [])
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model).strip("._") or "model"
    output = (args.output_dir or ROOT / "outputs/eval" / f"{model_name}_{timestamp}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "vllm", "transformers", "math-verify"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {
        "model": args.model, "suite": suite, "benchmarks": benchmarks, "gpus": ids,
        "tensor_parallel_size": args.tensor_parallel_size,
        "save_generations": args.save_generations,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime_versions": versions, "status": "running",
    }
    write_json(output / "manifest.json", manifest)
    print(f"Evaluation output: {output}", flush=True)
    try:
        common = ["--model", args.model, "--gpus", args.gpus, "--tensor-parallel-size", str(args.tensor_parallel_size)]
        save = ["--save-generations"] if args.save_generations else []
        if math_benchmarks:
            command = [
                sys.executable,
                str(HERE / "eval_vllm_dp.py"),
                *common,
                "--output-dir",
                str(output / "math"),
                "--temperature",
                "0.7",
                "--top-p",
                "0.9",
                "--seed",
                "1",
                "--max-prompt-length",
                "1024",
                "--max-response-length",
                "16384",
                "--val-batch-size",
                "1024",
                *save,
            ]
            for name, n in math_benchmarks.items():
                command.extend(["--test-set", f"{name}={n}"])
            run_logged(command, output / "math.log")
        if include_gpqa:
            run_logged(
                [sys.executable, str(HERE / "gpqa_eval.py"), *common, "--output-dir", str(output / "gpqa"),
                 "--gpqa-file", str(gpqa_file), *save],
                output / "gpqa.log",
            )
        gpqa = json.loads((output / "gpqa/summary.json").read_text()) if include_gpqa else None
        if not math_benchmarks:
            scores = {"GPQA-Diamond": gpqa["accuracy"]}
        else:
            math_summary = json.loads((output / "math/summary.json").read_text())
            if args.benchmark in MATH:
                score = math_summary["metrics"][f"val/{args.benchmark}/math_verify_accuracy_reward"]
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Invalid accuracy in {args.benchmark}")
                scores = {args.benchmark: score}
            else:
                scores = paper_table(math_summary, gpqa)
        write_json(output / "summary.json", {
            "suite": suite, "benchmarks": benchmarks, "scores": scores, "unit": "accuracy_fraction",
        })
        table = "| Benchmark | Accuracy (%) |\n| --- | ---: |\n"
        table += "".join(f"| {name} | {100 * value:.2f} |\n" for name, value in scores.items())
        (output / "scores.md").write_text(table, encoding="utf-8")
        print("\n" + table)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
