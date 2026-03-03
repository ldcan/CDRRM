#!/usr/bin/env python3
"""
Preprocess and filter pairwise data for the SFT pipeline.

Step 1 – Preprocess sampled.jsonl: assign instruction_id / pair_id by line number
          for any records that are missing them.
Step 2 – Filter rubric_pair.jsonl: remove every instruction_id that contains at
          least one pair with a failing StepB consistency check.  Only consistent
          pairs are written to the output file.

Usage:
  python filter_inconsistent_rubrics.py \
      --input_file        data/sampled.jsonl \
      --preprocessed_file output/sft_data/preprocessed_sampled.jsonl \
      --rubric_pair_jsonl ../rubric_pipeline/output/rubric_pair.jsonl \
      --output_file       output/sft_data/rubric_pair_filtered.jsonl
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_jsonl, save_jsonl, iter_jsonl, setup_logging, logger


# ============================================================================
# Step 1: Preprocess sampled data
# ============================================================================

def preprocess_sampled_data(input_file: str, output_file: str) -> Dict[str, int]:
    """Add line-number-based instruction_id / pair_id where missing."""
    data = load_jsonl(input_file)
    if not data:
        logger.error("Input data is empty")
        return {"total": 0, "processed": 0, "skipped": 0}

    stats = {"total": len(data), "processed": 0, "skipped": 0}
    for idx, item in enumerate(data, 1):
        if item.get("instruction_id") or item.get("id"):
            stats["skipped"] += 1
        else:
            item["instruction_id"] = f"inst_{idx}"
            stats["processed"] += 1
        if not item.get("pair_id"):
            item["pair_id"] = f"line_{idx}"

    save_jsonl(output_file, data)
    return stats


# ============================================================================
# Step 2: Filter inconsistent rubric pairs
# ============================================================================

def _is_consistent(rubric_pair: Dict[str, Any]) -> bool:
    """Return True if expected_winner matches rubric_predicts (or either is absent)."""
    cc = rubric_pair.get("pair_consistency_check") or {}
    if not cc:
        return True

    def _normalize(val: str) -> str:
        v = str(val).strip().upper()
        return {"CHOSEN": "A", "REJECTED": "B"}.get(v, v)

    expected = _normalize(cc.get("expected_winner", ""))
    predicted = _normalize(cc.get("rubric_predicts", ""))

    if not expected or not predicted:
        return True
    return expected == predicted


def filter_rubric_pairs(rubric_pair_jsonl: str, output_file: str) -> Dict[str, int]:
    """
    Filter rubric_pair.jsonl, removing any instruction_id that has at least one
    inconsistent pair_consistency_check.  Writes all consistent pairs to output_file.
    """
    if not os.path.exists(rubric_pair_jsonl):
        logger.error(f"File not found: {rubric_pair_jsonl}")
        return {"total_instructions": 0, "inconsistent_instructions": 0,
                "kept_instructions": 0, "kept_pairs": 0}

    by_instruction: Dict[str, List[Dict]] = defaultdict(list)
    for obj in iter_jsonl(rubric_pair_jsonl):
        iid = obj.get("instruction_id", "")
        if iid:
            by_instruction[iid].append(obj)

    total_pairs = sum(len(v) for v in by_instruction.values())
    logger.info(f"Loaded {total_pairs} pairs from {len(by_instruction)} instructions")

    inconsistent: Set[str] = {
        iid for iid, pairs in by_instruction.items()
        if any(not _is_consistent(p) for p in pairs)
    }
    logger.info(f"Found {len(inconsistent)} instructions with inconsistent pairs")

    kept: List[Dict] = [
        obj
        for iid, pairs in by_instruction.items()
        if iid not in inconsistent
        for obj in pairs
    ]
    save_jsonl(output_file, kept)

    return {
        "total_instructions": len(by_instruction),
        "inconsistent_instructions": len(inconsistent),
        "kept_instructions": len(by_instruction) - len(inconsistent),
        "kept_pairs": len(kept),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Preprocess pairwise data and filter inconsistent rubric pairs"
    )
    parser.add_argument("--input_file", required=True,
                        help="Raw sampled JSONL (preprocess step input)")
    parser.add_argument("--preprocessed_file", required=True,
                        help="Output JSONL after ID assignment")
    parser.add_argument("--rubric_pair_jsonl", required=True,
                        help="rubric_pair.jsonl from rubric_pipeline")
    parser.add_argument("--output_file", required=True,
                        help="Filtered rubric_pair JSONL (consistent pairs only)")
    args = parser.parse_args()

    logger.info("=== Step 1: Preprocess sampled data ===")
    s1 = preprocess_sampled_data(args.input_file, args.preprocessed_file)
    logger.info(
        f"total={s1['total']}  id_assigned={s1['processed']}  "
        f"already_had_id={s1['skipped']}"
    )
    logger.info(f"Output: {args.preprocessed_file}")

    logger.info("=== Step 2: Filter inconsistent rubric pairs ===")
    s2 = filter_rubric_pairs(args.rubric_pair_jsonl, args.output_file)
    logger.info(
        f"instructions total={s2['total_instructions']}  "
        f"inconsistent={s2['inconsistent_instructions']}  "
        f"kept={s2['kept_instructions']}  "
        f"kept_pairs={s2['kept_pairs']}"
    )
    logger.info(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
