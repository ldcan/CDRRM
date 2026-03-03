#!/usr/bin/env bash

set -euo pipefail

# Default values
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-../models/Rubric-Generator-8B}"
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model <path> --dataset <path> --output_dir <path> [options]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]] || [[ -z "$DATASET_PATH" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: MODEL_PATH, DATASET_PATH, and OUTPUT_DIR must be set (via env vars or --model/--dataset/--output_dir)"
    echo "Usage: $0 --model <path> --dataset <path> --output_dir <path> [options]"
    echo "  Or set environment variables: MODEL_PATH, DATASET_PATH, OUTPUT_DIR"
    exit 1
fi

# NCCL configuration (can be overridden via environment variables)
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

# Run training
NNODES="$NNODES" \
NODE_RANK=0 \
MASTER_PORT="$MASTER_PORT" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --attn_impl flash_attn \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --eval_steps 0 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --logging_dir "$OUTPUT_DIR" \
    --logging_steps 10 \
    --max_length 25000 \
    --num_train_epochs 2 \
    --packing False \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.0 \
    --torch_dtype bfloat16 \
    --train_type full \
    --use_hf True \
    --warmup_ratio 0.05
