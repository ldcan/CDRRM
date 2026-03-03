#!/usr/bin/env bash
# Rubric Generation Pipeline: StepA (diagnosis) -> StepB (rubric generation)
#
# Usage:
#   cd cdrrm/rubric_pipeline
#   bash run_pipeline.sh data/sampled.jsonl output/ [BATCH_SIZE]
#
# Requires:
#   OPENAI_API_KEY  – OpenAI-compatible API key
#   OPENAI_BASE_URL – API base URL (optional, defaults to https://api.openai.com/v1)
#   MODEL_NAME      – Model to use for both StepA and StepB

set -euo pipefail

INPUT_FILE="${1:-data/rubric_generation_sampled_data.jsonl}"
OUTPUT_BASE_DIR="${2:-output}"
BATCH_SIZE="${3:-25}"

DIAGNOSIS_JSONL="${OUTPUT_BASE_DIR}/diagnosis.jsonl"
RUBRIC_PAIR_JSONL="${OUTPUT_BASE_DIR}/rubric_pair.jsonl"

echo "=========================================="
echo "Rubric Generation Pipeline"
echo "=========================================="
echo "Input:       ${INPUT_FILE}"
echo "Output:      ${OUTPUT_BASE_DIR}"
echo "Batch size:  ${BATCH_SIZE}"
echo ""

# StepA: Structured Diagnosis
echo "=========================================="
echo "StepA: Structured Diagnosis Generation"
echo "=========================================="
python diagnosis_generation.py \
    --input_file   "${INPUT_FILE}" \
    --output_jsonl "${DIAGNOSIS_JSONL}" \
    --batch_size   "${BATCH_SIZE}"

echo ""

# StepB: Discriminative Rubric Generation
echo "=========================================="
echo "StepB: Discriminative Rubric Generation"
echo "=========================================="
python rubric_generation.py \
    --diagnosis_jsonl "${DIAGNOSIS_JSONL}" \
    --pairs_file      "${INPUT_FILE}" \
    --output_jsonl    "${RUBRIC_PAIR_JSONL}" \
    --batch_size      "${BATCH_SIZE}"

echo ""
echo "Done. Outputs in: ${OUTPUT_BASE_DIR}"
