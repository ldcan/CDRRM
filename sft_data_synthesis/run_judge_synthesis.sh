#!/usr/bin/env bash

set -euo pipefail

INPUT_JSONL="${1:-../rubric_synthesis/data/judge_synthesis_sampled_data.jsonl}"
RUBRICS_JSONL="${2:-../eval/rubrics/rubrics_openrubrics.jsonl}"
OUTPUT_DIR="${3:-output/judge_sft_data}"

if [[ -z "$INPUT_JSONL" ]] || [[ -z "$RUBRICS_JSONL" ]]; then
    echo "Error: input_jsonl and rubrics_jsonl are required"
    echo "Usage: $0 <input_jsonl> <rubrics_jsonl> [output_dir]"
    echo "  input_jsonl:   raw pairwise data (instruction, response_a, response_b, winner), no id fields required"
    echo "  rubrics_jsonl: rubrics file from eval pipeline (contains formatted_rubric, sid corresponds to input line index)"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=== Synthesize judge SFT data ==="
python judge_synthesis.py \
    --input_file   "${INPUT_JSONL}" \
    --rubrics_file "${RUBRICS_JSONL}" \
    --output_dir   "${OUTPUT_DIR}" \
    --batch_size   "${JUDGE_BATCH:-50}" \
    --task_id      "judge_pairwise"

echo "Done. Judge synthesis SFT data written to: ${OUTPUT_DIR}"
