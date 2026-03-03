#!/usr/bin/env python3
"""
Generate rubrics for pairwise benchmark data using a fine-tuned SFT model.

Supported benchmarks:
  - rewardbench:  RewardBench parquet file
  - rmbench:      RM-Bench JSONL/JSON file
  - rmb:          RMB JSON file or directory
  - openrubrics:  Custom JSONL file with instruction/response_a/response_b/winner fields

Output: a JSONL file with fields:
  sid, benchmark, instruction, response_a, response_b,
  raw_rubric_output, formatted_rubric, extra_info
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np  
from tqdm import tqdm  
from vllm import LLM, SamplingParams  

# Shared modules live one level up (cdrrm/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts import RUBRIC_GEN_SYSTEM, RUBRIC_GEN_USER_TEMPLATE  
from evaluate import (  
    PairwiseSample,
    load_rewardbench,
    load_rmbench,
    load_rmb,
    build_chat_prompt,
)


# ============================================================================
# OpenRubrics data loader
# ============================================================================

def load_openrubrics(path: str, shuffle_pairs: bool, seed: int) -> List[PairwiseSample]:
    """
    Load OpenRubrics format JSONL file.

    Expected fields per line:
      instruction (str), response_a (str), response_b (str),
      winner (str: "A" or "B"), pair_id (str, optional)
    """
    rng = np.random.default_rng(seed)
    samples: List[PairwiseSample] = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            instruction = row.get("instruction", "")
            response_a = row.get("response_a", "")
            response_b = row.get("response_b", "")
            winner = row.get("winner", "").upper()
            if not instruction or not response_a or not response_b:
                continue

            # Determine chosen/rejected based on the winner field
            if winner in {"A", "RESPONSE_A", "A_WIN"}:
                chosen, rejected, gt = response_a, response_b, "A"
            elif winner in {"B", "RESPONSE_B", "B_WIN"}:
                chosen, rejected, gt = response_b, response_a, "B"
            else:
                chosen, rejected, gt = response_a, response_b, "A"  # default

            resp_a, resp_b = chosen, rejected
            if shuffle_pairs and float(rng.random()) > 0.5:
                resp_a, resp_b, gt = rejected, chosen, "B" if gt == "A" else "A"

            sid = row.get("pair_id") or row.get("instruction_id") or str(row.get("id", idx))
            samples.append(PairwiseSample(
                sid=str(sid),
                instruction=instruction,
                response_a=resp_a,
                response_b=resp_b,
                ground_truth=gt,
                data_source="openrubrics",
                extra_info={
                    "dataset": "openrubrics",
                    "instruction_id": row.get("instruction_id", ""),
                    "pair_id": row.get("pair_id", ""),
                    "original_winner": winner,
                },
            ))
    return samples


# ============================================================================
# Rubric parsing
# ============================================================================

def _parse_rubric_json(rubric_text: str) -> str:
    """
    Parse a JSON rubric from model output and format it as plain text
    suitable for use in the judge prompt.

    Supports rubric dicts with 'hard_rules' and 'principles' arrays.
    """
    # Try to extract a JSON object containing 'hard_rules'
    json_str: Optional[str] = None
    for pattern in [
        r'\{[^{}]*"hard_rules"[^{}]*\}',
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'\{.*"hard_rules".*\}',
    ]:
        m = re.search(pattern, rubric_text, re.DOTALL)
        if m:
            json_str = m.group(1) if m.lastindex else m.group(0)
            break

    if json_str is None:
        return "Hard Rules:\n1. (Failed to parse rubric)\n\nPrinciples:\nNone"

    try:
        rubric = json.loads(json_str)
    except json.JSONDecodeError:
        return "Hard Rules:\n1. (Failed to parse rubric)\n\nPrinciples:\nNone"

    lines = ["--- Rubric Generation ---"]

    hard_rules = rubric.get("hard_rules", [])
    if hard_rules:
        lines.append("Hard Rules:")
        for i, rule in enumerate(hard_rules, 1):
            text = rule.get("criterion", rule.get("rule", str(rule))) if isinstance(rule, dict) else str(rule)
            lines.append(f"{i}. {text}")
    else:
        lines.append("Hard Rules:\nNone")

    lines.append("")

    principles = rubric.get("principles", [])
    if principles:
        lines.append("Principles:")
        for i, p in enumerate(principles, 1):
            text = p.get("description", p.get("principle", str(p))) if isinstance(p, dict) else str(p)
            lines.append(f"{i}. {text}")
    else:
        lines.append("Principles:\nNone")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate pairwise rubrics using a fine-tuned SFT model")

    # Model
    parser.add_argument("--sft_model_path", type=str, required=True, help="Path to the rubric-generation SFT model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096)

    # Data
    parser.add_argument(
        "--benchmark", type=str, default="rewardbench",
        choices=["rewardbench", "rmbench", "rmb", "openrubrics"],
    )
    parser.add_argument("--test_parquet", type=str, default="", help="RewardBench parquet path")
    parser.add_argument("--rmbench_jsonl", type=str, default="", help="RM-Bench data path")
    parser.add_argument("--rmb_json", type=str, default="", help="RMB data path or directory")
    parser.add_argument("--rmb_json_dir", action="store_true", help="Treat --rmb_json as a directory")
    parser.add_argument("--openrubrics_jsonl", type=str, default="", help="OpenRubrics JSONL path")
    parser.add_argument("--shuffle_pairs", action="store_true", help="Randomly swap A/B positions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0, help="Limit samples (0 = all)")

    # Output
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file path")

    args = parser.parse_args()

    # --- Load dataset ---
    if args.benchmark == "rewardbench":
        if not args.test_parquet:
            raise ValueError("--test_parquet is required for benchmark=rewardbench")
        samples = load_rewardbench(args.test_parquet, args.shuffle_pairs, args.seed)
    elif args.benchmark == "rmb":
        if not args.rmb_json:
            raise ValueError("--rmb_json is required for benchmark=rmb")
        is_dir = args.rmb_json_dir or os.path.isdir(args.rmb_json)
        samples = load_rmb(args.rmb_json, args.shuffle_pairs, args.seed, is_dir=is_dir)
    elif args.benchmark == "openrubrics":
        if not args.openrubrics_jsonl:
            raise ValueError("--openrubrics_jsonl is required for benchmark=openrubrics")
        samples = load_openrubrics(args.openrubrics_jsonl, args.shuffle_pairs, args.seed)
    else:  # rmbench
        if not args.rmbench_jsonl:
            raise ValueError("--rmbench_jsonl is required for benchmark=rmbench")
        samples = load_rmbench(args.rmbench_jsonl, args.shuffle_pairs, args.seed)

    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"Loaded {len(samples)} samples")

    # --- Load SFT model ---
    print(f"Loading SFT model: {args.sft_model_path}")
    llm = LLM(
        model=args.sft_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # --- Build prompts ---
    prompts: List[str] = []
    for s in samples:
        user = RUBRIC_GEN_USER_TEMPLATE.format(
            instruction=s.instruction,
            response_a=s.response_a,
            response_b=s.response_b,
        )
        prompts.append(build_chat_prompt(tokenizer, RUBRIC_GEN_SYSTEM, user))

    # --- Generate rubrics ---
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    raw_outputs: List[str] = []
    formatted: List[str] = []

    print("Generating rubrics...")
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="generate rubrics"):
        for o in llm.generate(prompts[i : i + args.batch_size], sp):
            raw = o.outputs[0].text
            raw_outputs.append(raw)
            formatted.append(_parse_rubric_json(raw))

    # --- Save output ---
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for s, raw, fmt in zip(samples, raw_outputs, formatted):
            f.write(json.dumps({
                "sid": s.sid,
                "benchmark": args.benchmark,
                "instruction": s.instruction,
                "response_a": s.response_a,
                "response_b": s.response_b,
                "raw_rubric_output": raw,
                "formatted_rubric": fmt,
                "extra_info": s.extra_info or {},
            }, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} rubrics to: {args.output_file}")


if __name__ == "__main__":
    main()
