#!/usr/bin/env bash
# Rubric Generation SFT Data Preparation Pipeline
#
# Step 1: Preprocess sampled pairwise data (assign missing IDs) +
#         Filter rubrics with inconsistent StepB predictions
# Step 2: Generate rubric-generation SFT data (train/val split)
#
# Usage:
#   cd cdrrm/sft_data_synthesis
#   bash run_rubric_generation.sh \
#       <sampled_jsonl>      \   # raw pairwise data (e.g. rubric_pipeline/data/sampled.jsonl)
#       <rubric_pair_jsonl>  \   # rubric_pipeline output: rubric_pair.jsonl
#       [output_dir]             # default: output/rubric_generation_sft_data
#
# Output:
#   <output_dir>/preprocessed_sampled.jsonl
#   <output_dir>/rubric_pair_filtered.jsonl
#   <output_dir>/train.jsonl
#   <output_dir>/val.jsonl

set -euo pipefail

SAMPLED_JSONL="${1:-}"
RUBRIC_PAIR_JSONL="${2:-}"
OUTPUT_DIR="${3:-output/rubric_generation_sft_data}"

if [[ -z "$SAMPLED_JSONL" ]] || [[ -z "$RUBRIC_PAIR_JSONL" ]]; then
    echo "Error: sampled_jsonl and rubric_pair_jsonl are required"
    echo "Usage: $0 <sampled_jsonl> <rubric_pair_jsonl> [output_dir]"
    exit 1
fi

PREPROCESSED_JSONL="${OUTPUT_DIR}/preprocessed_sampled.jsonl"
FILTERED_PAIR_JSONL="${OUTPUT_DIR}/rubric_pair_filtered.jsonl"

mkdir -p "${OUTPUT_DIR}"

echo "=== Step 1: Preprocess + filter inconsistent rubrics ==="
python filter_inconsistent_rubrics.py \
    --input_file        "${SAMPLED_JSONL}" \
    --preprocessed_file "${PREPROCESSED_JSONL}" \
    --rubric_pair_jsonl "${RUBRIC_PAIR_JSONL}" \
    --output_file       "${FILTERED_PAIR_JSONL}"

echo "=== Step 2: Generate rubric-generation SFT data ==="
python prepare_sft_data.py \
    --rubric_pair_jsonl "${FILTERED_PAIR_JSONL}" \
    --pairs_jsonl       "${PREPROCESSED_JSONL}" \
    --output_dir        "${OUTPUT_DIR}" \
    --train_ratio       1.0 \
    --seed              42

echo "Done. Rubric generation SFT data written to: ${OUTPUT_DIR}"
