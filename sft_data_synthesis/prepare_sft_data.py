#!/usr/bin/env python3
"""
Generate SFT training data for the rubric-generation model.

For each rubric in rubric_pair_filtered.jsonl, paired with its corresponding
pairwise sample, produces one training record:
  system    – RUBRIC_GEN_SYSTEM
  user      – instruction + Response A + Response B
  assistant – cleaned rubric JSON (hard_rules + principles)

Input:
  --rubric_pair_jsonl  Filtered rubric_pair JSONL (from filter_inconsistent_rubrics.py)
  --pairs_jsonl        JSONL with pairwise data (instruction, response_a/b, winner)

Output:
  <output_dir>/train.jsonl
  <output_dir>/val.jsonl
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts import RUBRIC_GEN_SYSTEM, RUBRIC_GEN_USER_TEMPLATE
from utils import load_jsonl, save_jsonl, init_dirs, setup_logging, logger


# ============================================================================
# Rubric cleaning
# ============================================================================

def _clean_rubric(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip internal-only fields (derived_from, merged_rule_ids) from a rubric.

    Keeps for hard rules:  rule_id, type, criterion, test, rationale
    Keeps for principles:  principle_id, description, rationale
    """
    hard_rules = []
    for rule in output.get("hard_rules", []):
        if not isinstance(rule, dict):
            continue
        cleaned = {
            "rule_id": rule.get("rule_id", ""),
            "type": rule.get("type", ""),
            "criterion": rule.get("criterion", ""),
            "test": rule.get("test", ""),
            "rationale": rule.get("rationale", ""),
        }
        if cleaned["rule_id"] and cleaned["criterion"]:
            hard_rules.append(cleaned)

    principles = []
    for p in output.get("principles", []):
        if not isinstance(p, dict):
            continue
        cleaned = {
            "principle_id": p.get("principle_id", ""),
            "description": p.get("description", ""),
            "rationale": p.get("rationale", ""),
        }
        if cleaned["principle_id"] and cleaned["description"]:
            principles.append(cleaned)

    return {"hard_rules": hard_rules, "principles": principles}


# ============================================================================
# Main
# ============================================================================

def build_samples(
    rubric_pair_jsonl: str,
    pairs_jsonl: str,
) -> List[Dict[str, str]]:
    """
    Load rubrics and pairs, then build one training sample per matched rubric.

    After loading, response_a is always the chosen (winner) response:
      - If original winner == "A": keep as-is
      - If original winner == "B": swap response_a / response_b
    """
    rubrics = load_jsonl(rubric_pair_jsonl)
    pairs = load_jsonl(pairs_jsonl)

    # Index rubrics by instruction_id
    rubric_index: Dict[str, Dict[str, Any]] = {}
    for r in rubrics:
        iid = r.get("instruction_id", "")
        if iid:
            rubric_index[iid] = r
    logger.info(f"Loaded {len(rubric_index)} rubrics")

    # Index first pair per instruction_id (normalize so response_a = chosen)
    pair_index: Dict[str, Dict[str, Any]] = {}
    for p in pairs:
        iid = p.get("instruction_id") or p.get("id", "")
        if not iid or iid in pair_index:
            continue
        response_a = p.get("response_a", "")
        response_b = p.get("response_b", "")
        winner = p.get("winner", "").upper()
        if not response_a or not response_b:
            continue
        if winner in {"B", "RESPONSE_B", "B_WIN"}:
            response_a, response_b = response_b, response_a
        pair_index[iid] = {
            "instruction": p.get("instruction", ""),
            "response_a": response_a,
            "response_b": response_b,
        }
    logger.info(f"Loaded {len(pair_index)} pairs")

    # Build training samples
    samples: List[Dict[str, str]] = []
    skipped = 0
    for iid, rubric in rubric_index.items():
        pair = pair_index.get(iid)
        if pair is None:
            skipped += 1
            continue
        instruction = pair["instruction"] or rubric.get("instruction", "")
        if not instruction or not pair["response_a"] or not pair["response_b"]:
            skipped += 1
            continue

        user = RUBRIC_GEN_USER_TEMPLATE.format(
            instruction=instruction,
            response_a=pair["response_a"],
            response_b=pair["response_b"],
        )
        assistant = json.dumps(
            _clean_rubric({"hard_rules": rubric.get("hard_rules", []), "principles": rubric.get("principles", [])}),
            ensure_ascii=False,
            indent=2,
        )
        samples.append({"system": RUBRIC_GEN_SYSTEM, "user": user, "assistant": assistant})

    logger.info(f"Built {len(samples)} samples ({skipped} skipped due to missing pairs/instruction)")
    return samples


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Generate rubric-generation SFT training data")
    parser.add_argument("--rubric_pair_jsonl", type=str, required=True,
                        help="Filtered rubric_pair JSONL (output of filter_inconsistent_rubrics.py)")
    parser.add_argument("--pairs_jsonl", type=str, required=True,
                        help="Pairwise data JSONL (instruction, response_a/b, winner)")
    parser.add_argument("--output_dir", type=str, default="output/sft_data")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Fraction of data used for training (rest goes to val)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    init_dirs(args.output_dir)

    samples = build_samples(args.rubric_pair_jsonl, args.pairs_jsonl)
    if not samples:
        logger.error("No samples generated; check rubric and pairs files")
        return

    random.shuffle(samples)
    split = int(len(samples) * args.train_ratio)
    train, val = samples[:split], samples[split:]

    save_jsonl(os.path.join(args.output_dir, "train.jsonl"), train)
    save_jsonl(os.path.join(args.output_dir, "val.jsonl"), val)

    logger.info("=" * 50)
    logger.info(f"Done  train={len(train)}  val={len(val)}")
    logger.info(f"  -> {args.output_dir}/train.jsonl")
    logger.info(f"  -> {args.output_dir}/val.jsonl")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
