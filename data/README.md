# Data

Training and evaluation data used in the
[SCAPO paper](https://arxiv.org/abs/2609.40360).

## Training

- `train/dapo17k.parquet`: 17,242 rows, with string fields `problem` and `answer`,
  derived from [DAPO-Math-17K](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k).
- `train/dapo17k_semifactual.jsonl`: 17,242 aligned records constructed for
  SCAPO. Each record has the same `problem` and `answer`, and a `perturbations`
  object containing `paraphrase`, `typo_noise`, `scenario_wrap`, and
  `irrelevant_context`.

## Mathematical evaluation

The 19 math evaluation Parquet files contain only `problem` and `answer` columns.
Math sampling counts follow [benchmarks.json](../scripts/evaluation/benchmarks.json).

Use `--benchmark <name>` to select a benchmark from the table below, or `--benchmark gpqa` for GPQA-Diamond.
Omit `--benchmark` to evaluate all math benchmarks and include GPQA-Diamond when its CSV is available.

| Files | Benchmark names (`--benchmark`) | Rows per file |
| --- | --- | --- |
| `eval/aime2024.parquet`, `eval/aime2025.parquet`, `eval/aime2026.parquet` | `aime2024`, `aime2025`, `aime2026` | 30 / 30 / 30 |
| `eval/amc2023.parquet`, `eval/amc2024.parquet`, `eval/amc2025.parquet` | `amc2023`, `amc2024`, `amc2025` | 40 / 45 / 42 |
| `eval/hmmt2025.parquet`, `eval/hmmt2026.parquet` | `hmmt2025`, `hmmt2026` | 30 / 33 |
| `eval/brumo_2025.parquet` | `brumo_2025` | 30 |
| `eval/smt_2025.parquet` | `smt_2025` | 53 |
| `eval/omni_math_rule.parquet` | `omni_math_rule` | 2,821 |
| `eval/minerva.parquet` | `minerva` | 272 |
| `eval/olympiadbench.parquet` | `olympiadbench` | 674 |
| `eval/noop_aime2024.parquet`, `eval/noop_aime2025.parquet`, `eval/noop_aime2026.parquet` | `noop_aime2024`, `noop_aime2025`, `noop_aime2026` | 30 / 30 / 30 |
| `eval/thinkbench_aime2024.parquet`, `eval/thinkbench_aime2025.parquet`, `eval/thinkbench_aime2026.parquet` | `thinkbench_aime2024`, `thinkbench_aime2025`, `thinkbench_aime2026` | 120 / 120 / 120 |

## GPQA-Diamond evaluation

To comply with GPQA's access conditions, we do not include the original questions.
Download `gpqa_diamond.csv` from the [official GPQA dataset](https://huggingface.co/datasets/Idavidrein/gpqa)
and place it in `data/eval/` at the repository root:

```text
data/eval/gpqa_diamond.csv
```

The evaluator requires 198 questions and these columns: `Question`,
`Correct Answer`, `Incorrect Answer 1`, `Incorrect Answer 2`, and
`Incorrect Answer 3`. We enumerate all 24
option permutations for each question and check that all 4,752 records are
present exactly once before computing the score.

## Evaluation protocol

| Benchmarks | Samples per problem | Scoring |
| --- | ---: | --- |
| Competition math & NoOp | 16 | Math-Verify 0.9.0 |
| ThinkBench | 4 | Math-Verify 0.9.0 |
| Omni-Math, Minerva & OlympiadBench | 1 | Math-Verify 0.9.0 |
| GPQA-Diamond | 24 option permutations | Multiple-choice accuracy |

GPQA uses the deterministic seed `1 + 24 × question_index + permutation_index`, with zero-based indices.
Scores are averaged over the 24 permutations for each question, then over all 198 questions.

Multi-year math scores are weighted by problem count × samples. ID/OOD averages are
unweighted means of 8 ID / 3 OOD benchmark groups and are reported only when all
required benchmarks have been evaluated. A single-benchmark run reports only its own accuracy.

Results include `summary.json`, `scores.md`, configurations, and logs.
