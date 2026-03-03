#!/usr/bin/env bash
# Run pairwise evaluation on reward benchmarks.
#
# Usage:
#   bash run_eval.sh --benchmark rewardbench --prompt_type direct_judge \
#                    --model_path /path/to/model --test_parquet /path/to/data.parquet
#
# Supported benchmarks:  rewardbench | rmbench | rmb
# Supported prompt types: direct_judge | rubric_judge

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

# ---- Default values (override via environment or CLI flags) ----
MODEL_PATH_DEFAULT="${MODEL_PATH_DEFAULT:-}"
REWARDBENCH_PARQUET_DEFAULT="${REWARDBENCH_PARQUET_DEFAULT:-eval_dataset/reward-bench/data/filtered-00000-of-00001.parquet}"
RMBENCH_JSON_DEFAULT="${RMBENCH_JSON_DEFAULT:-eval_dataset/RM-Bench/total_dataset.json}"
RMB_JSON_DEFAULT="${RMB_JSON_DEFAULT:-eval_dataset/RMB_dataset/Pairwise_set}"

TP_DEFAULT="${TP_DEFAULT:-8}"
BATCH_SIZE_DEFAULT="${BATCH_SIZE_DEFAULT:-128}"
MAX_TOKENS_DEFAULT="${MAX_TOKENS_DEFAULT:-8192}"
GPU_MEM_UTIL_DEFAULT="${GPU_MEM_UTIL_DEFAULT:-0.9}"
SEED_DEFAULT="${SEED_DEFAULT:-42}"
OUTPUT_ROOT_DEFAULT="${OUTPUT_ROOT_DEFAULT:-./eval_results}"

# ---- Usage function ----
usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Required:
  --model_path PATH           Path to judge model
  --benchmark NAME            Benchmark name: rewardbench | rmbench | rmb
  --prompt_type TYPE          Prompt type: direct_judge | rubric_judge

Benchmark-specific data files (required based on --benchmark):
  --test_parquet PATH         For rewardbench: path to test parquet file
  --rmbench_jsonl PATH        For rmbench: path to RM-Bench JSON file
  --rmb_json PATH             For rmb: path to RMB JSON file or directory

Conditional:
  --rubrics_file PATH         Required when --prompt_type=rubric_judge

Optional:
  --output_root PATH          Output root directory (default: ./eval_results)
  --batch_size N              Batch size (default: ${BATCH_SIZE_DEFAULT})
  --tensor_parallel_size N     Tensor parallel size (default: ${TP_DEFAULT})
  --gpu_memory_utilization F  GPU memory utilization (default: ${GPU_MEM_UTIL_DEFAULT})
  --max_tokens N              Max tokens to generate (default: ${MAX_TOKENS_DEFAULT})
  --seed N                    Random seed (default: ${SEED_DEFAULT})
  --shuffle                   Shuffle pairs before processing
  --max_samples N             Limit number of samples (0 = no limit, default: 0)
  --rmb_json_dir              Treat --rmb_json as directory (for rmb benchmark)

Examples:
  # Direct judge on RewardBench
  $0 --model_path /path/to/model \\
     --benchmark rewardbench \\
     --prompt_type direct_judge \\
     --test_parquet eval_dataset/reward-bench/data/filtered-00000-of-00001.parquet

  # Rubric judge on RM-Bench
  $0 --model_path /path/to/model \\
     --benchmark rmbench \\
     --prompt_type rubric_judge \\
     --rmbench_jsonl eval_dataset/RM-Bench/total_dataset.json \\
     --rubrics_file ./rubrics/rubrics_rmbench.jsonl

  # Rubric judge on RMB
  $0 --model_path /path/to/model \\
     --benchmark rmb \\
     --prompt_type rubric_judge \\
     --rmb_json eval_dataset/RMB_dataset/Pairwise_set \\
     --rubrics_file ./rubrics/rubrics_rmb.jsonl \\
     --rmb_json_dir
EOF
}

# ---- Mutable state ----
BENCHMARK=""
PROMPT_TYPE=""
MODEL_PATH="${MODEL_PATH_DEFAULT}"
RUBRICS_FILE=""
TEST_PARQUET="${REWARDBENCH_PARQUET_DEFAULT}"
RMBENCH_JSON="${RMBENCH_JSON_DEFAULT}"
RMB_JSON="${RMB_JSON_DEFAULT}"
RMB_JSON_DIR="0"
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT}"
BATCH_SIZE="${BATCH_SIZE_DEFAULT}"
TP="${TP_DEFAULT}"
MAX_TOKENS="${MAX_TOKENS_DEFAULT}"
GPU_MEM_UTIL="${GPU_MEM_UTIL_DEFAULT}"
SEED="${SEED_DEFAULT}"
SHUFFLE="0"
MAX_SAMPLES="0"

# ---- Parse command line arguments ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    --prompt_type)
      PROMPT_TYPE="$2"
      shift 2
      ;;
    --rubrics_file)
      RUBRICS_FILE="$2"
      shift 2
      ;;
    --test_parquet)
      TEST_PARQUET="$2"
      shift 2
      ;;
    --rmbench_jsonl)
      RMBENCH_JSON="$2"
      shift 2
      ;;
    --rmb_json)
      RMB_JSON="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --tensor_parallel_size)
      TP="$2"
      shift 2
      ;;
    --gpu_memory_utilization)
      GPU_MEM_UTIL="$2"
      shift 2
      ;;
    --max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --shuffle)
      SHUFFLE="1"
      shift
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --rmb_json_dir)
      RMB_JSON_DIR="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# ---- Validation ----
if [[ -z "${MODEL_PATH}" ]]; then
  echo "Error: --model_path is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "${BENCHMARK}" ]]; then
  echo "Error: --benchmark is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "${PROMPT_TYPE}" ]]; then
  echo "Error: --prompt_type is required" >&2
  usage >&2
  exit 1
fi

valid_benchmarks="rewardbench rmbench rmb"
if [[ ! " ${valid_benchmarks} " =~ " ${BENCHMARK} " ]]; then
  echo "Error: unsupported benchmark: ${BENCHMARK}" >&2
  echo "Supported: ${valid_benchmarks}" >&2
  usage >&2
  exit 1
fi

valid_prompt_types="direct_judge rubric_judge"
if [[ ! " ${valid_prompt_types} " =~ " ${PROMPT_TYPE} " ]]; then
  echo "Error: unsupported prompt_type: ${PROMPT_TYPE}" >&2
  echo "Supported: ${valid_prompt_types}" >&2
  usage >&2
  exit 1
fi

if [[ "${PROMPT_TYPE}" == "rubric_judge" && -z "${RUBRICS_FILE}" ]]; then
  echo "Error: --rubrics_file is required for prompt_type=rubric_judge" >&2
  echo "Hint: run generate_rubrics.py first to create a rubrics file" >&2
  usage >&2
  exit 1
fi

# ---- Run evaluation ----
OUT_DIR="${OUTPUT_ROOT}/${BENCHMARK}/${PROMPT_TYPE}"
mkdir -p "${OUT_DIR}"

echo "==> Running evaluation"
echo "    benchmark=${BENCHMARK}  prompt_type=${PROMPT_TYPE}"
echo "    model_path=${MODEL_PATH}"
[[ -n "${RUBRICS_FILE}" ]] && echo "    rubrics_file=${RUBRICS_FILE}"
echo "    output_dir=${OUT_DIR}"

cmd=(
  "${PYTHON_BIN}" evaluate.py
  --benchmark    "${BENCHMARK}"
  --prompt_type  "${PROMPT_TYPE}"
  --model_path   "${MODEL_PATH}"
  --output_dir   "${OUT_DIR}"
  --batch_size   "${BATCH_SIZE}"
  --tensor_parallel_size "${TP}"
  --gpu_memory_utilization "${GPU_MEM_UTIL}"
  --max_tokens   "${MAX_TOKENS}"
  --seed         "${SEED}"
)

# Benchmark-specific data path
if [[ "${BENCHMARK}" == "rewardbench" ]]; then
  [[ -z "${TEST_PARQUET}" ]] && { echo "Error: --test_parquet is required for benchmark=rewardbench" >&2; exit 1; }
  echo "    test_parquet=${TEST_PARQUET}"
  cmd+=(--test_parquet "${TEST_PARQUET}")
elif [[ "${BENCHMARK}" == "rmb" ]]; then
  [[ -z "${RMB_JSON}" ]] && { echo "Error: --rmb_json is required for benchmark=rmb" >&2; exit 1; }
  echo "    rmb_json=${RMB_JSON}"
  cmd+=(--rmb_json "${RMB_JSON}")
  [[ "${RMB_JSON_DIR}" == "1" ]] && cmd+=(--rmb_json_dir)
else  # rmbench
  [[ -z "${RMBENCH_JSON}" ]] && { echo "Error: --rmbench_jsonl is required for benchmark=rmbench" >&2; exit 1; }
  echo "    rmbench_jsonl=${RMBENCH_JSON}"
  cmd+=(--rmbench_jsonl "${RMBENCH_JSON}")
fi

# Optional flags
[[ "${SHUFFLE}" == "1" ]] && cmd+=(--shuffle_pairs)
[[ "${MAX_SAMPLES}" != "0" ]] && cmd+=(--max_samples "${MAX_SAMPLES}")
[[ "${PROMPT_TYPE}" == "rubric_judge" ]] && cmd+=(--rubrics_file "${RUBRICS_FILE}")

"${cmd[@]}" | tee "${OUT_DIR}/stdout.log"

echo ""
echo "Done. Results saved to: ${OUTPUT_ROOT}/${BENCHMARK}"
