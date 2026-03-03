#!/usr/bin/env python3
"""
StepB: Discriminative Rubric Generation.

Given diagnosis results from StepA, calls the OpenAI API to generate a rubric
that discriminates between the chosen and rejected responses.

Reads from:
  --diagnosis_jsonl  JSONL output of StepA (diagnosis_generation.py)
  --pairs_file       Original input JSONL (used to recover the instruction text)

Writes to:
  --output_jsonl     JSONL with RubricPairResult records

Resume support: already-processed pair_ids are skipped on restart.
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts import STEPB_RUBRIC_SYSTEM, STEPB_RUBRIC_USER_TEMPLATE
from utils import load_jsonl, init_dirs, append_jsonl, iter_jsonl, extract_json_from_text, setup_logging
from llm_client import llm_call, DiagnosisResult, RubricPairResult
from diagnosis_generation import normalize_pair_data  # sibling module

logger = logging.getLogger(__name__)

DIAGNOSIS_JSONL = "output/diagnosis.jsonl"
OUTPUT_JSONL = "output/rubric_pair.jsonl"


# ============================================================================
# Helpers
# ============================================================================

def _load_diagnosis_index(diagnosis_jsonl: str) -> Dict[str, Dict[str, DiagnosisResult]]:
    """Build index: pair_id -> {'chosen': DiagnosisResult, 'rejected': DiagnosisResult}."""
    index: Dict[str, Dict[str, DiagnosisResult]] = {}
    for obj in iter_jsonl(diagnosis_jsonl):
        pair_id = str(obj.get("pair_id", "")).strip()
        label = str(obj.get("label", "")).strip()
        if not pair_id or label not in {"chosen", "rejected"}:
            continue
        try:
            diag = DiagnosisResult(obj)
        except Exception:
            continue
        index.setdefault(pair_id, {})[label] = diag
    return index


def _redact_label_fields(diag: DiagnosisResult) -> Dict[str, Any]:
    """Remove fields that leak chosen/rejected labels before sending to StepB."""
    d = dict(diag.to_dict())
    for field in ("label", "pair_id", "instruction_id"):
        d.pop(field, None)
    return d


# ============================================================================
# Core async functions
# ============================================================================

async def generate_rubric_async(
    instruction: str,
    diagnosis_chosen: DiagnosisResult,
    diagnosis_rejected: DiagnosisResult,
    instruction_id: str = "",
    pair_id: str = "",
) -> Optional[RubricPairResult]:
    """Generate a single rubric via the LLM API."""
    diagnosis_a_str = json.dumps(_redact_label_fields(diagnosis_chosen), ensure_ascii=False, indent=2)
    diagnosis_b_str = json.dumps(_redact_label_fields(diagnosis_rejected), ensure_ascii=False, indent=2)

    user_prompt = STEPB_RUBRIC_USER_TEMPLATE.format(
        instruction=instruction,
        diagnosis_a=diagnosis_a_str,
        diagnosis_b=diagnosis_b_str,
    )

    try:
        response_text = await llm_call(STEPB_RUBRIC_SYSTEM, user_prompt, max_tokens=4096)
        rubric_dict = extract_json_from_text(response_text)
        if not rubric_dict:
            logger.error(f"Failed to extract JSON for pair {pair_id}")
            return None

        rubric_dict["instruction_id"] = instruction_id or rubric_dict.get("instruction_id", "")
        rubric_dict["pair_id"] = pair_id or rubric_dict.get("pair_id", "")

        # Normalize legacy 'chosen'/'rejected' consistency check fields to A/B
        cc = rubric_dict.get("pair_consistency_check") or {}
        if isinstance(cc, dict):
            for field in ("expected_winner", "rubric_predicts"):
                val = str(cc.get(field, "")).strip().lower()
                if val == "chosen":
                    cc[field] = "A"
                elif val == "rejected":
                    cc[field] = "B"
            rubric_dict["pair_consistency_check"] = cc

        rubric = RubricPairResult(rubric_dict)
        is_valid, errors = rubric.validate()
        if not is_valid:
            logger.error(f"Rubric validation failed ({pair_id}): {errors}")
            return None

        # Warn if consistency check fails
        consistency = rubric.pair_consistency_check
        if consistency:
            expected = consistency.get("expected_winner", "")
            predicted = consistency.get("rubric_predicts", "")
            if expected in {"A", "chosen"} and predicted not in {"A", "chosen", "tie"}:
                logger.warning(
                    f"Consistency check mismatch ({pair_id}): "
                    f"expected=A, predicted={predicted}. {consistency.get('notes', '')}"
                )

        return rubric

    except Exception as e:
        logger.error(f"Error generating rubric ({pair_id}): {e}")
        return None


async def generate_rubric_batch(
    pairs_data: List[Dict[str, Any]],
    diagnosis_jsonl: str,
    output_jsonl: str,
    batch_size: int = 10,
) -> Dict[str, Any]:
    """Batch-generate rubric results with resume support."""
    stats: Dict[str, Any] = {
        "total": 0, "processed": 0, "success": 0,
        "failed": 0, "skipped": 0, "missing_diagnosis": 0,
    }

    # Resume: collect already-processed pair_ids
    done_pairs: set = {
        str(obj.get("pair_id", "")).strip()
        for obj in iter_jsonl(output_jsonl)
        if obj.get("pair_id")
    }

    # Build pair_id -> instruction mapping from raw pairs
    pair_to_instruction: Dict[str, Dict[str, str]] = {}
    for idx, pair in enumerate(pairs_data):
        normalized = normalize_pair_data(pair, idx)
        if not normalized:
            continue
        pair_to_instruction[normalized["pair_id"]] = {
            "instruction": normalized["instruction"],
            "instruction_id": normalized.get("instruction_id", ""),
        }

    # Load diagnosis index
    logger.info(f"Loading diagnosis index: {diagnosis_jsonl}")
    diagnosis_index = _load_diagnosis_index(diagnosis_jsonl)
    all_pair_ids = sorted(diagnosis_index.keys())
    stats["total"] = len(all_pair_ids)
    logger.info(f"Found {len(all_pair_ids)} diagnosed pairs")

    # Build task list
    tasks: List[Tuple[str, DiagnosisResult, DiagnosisResult, str, str]] = []
    for pair_id in all_pair_ids:
        if pair_id in done_pairs:
            stats["skipped"] += 1
            continue

        diags = diagnosis_index[pair_id]
        chosen_diag = diags.get("chosen")
        rejected_diag = diags.get("rejected")
        if not chosen_diag or not rejected_diag:
            logger.warning(f"Missing chosen/rejected diagnosis for pair {pair_id}")
            stats["missing_diagnosis"] += 1
            continue

        inst_info = pair_to_instruction.get(pair_id, {})
        instruction = inst_info.get("instruction", "") or chosen_diag.raw_data.get("instruction", "")
        if not instruction:
            logger.warning(f"Cannot find instruction for pair {pair_id}")
            stats["missing_diagnosis"] += 1
            continue

        instruction_id = inst_info.get("instruction_id", pair_id)
        tasks.append((instruction, chosen_diag, rejected_diag, instruction_id, pair_id))

    logger.info(f"Tasks: {len(tasks)} to process, {stats['skipped']} already done")

    for i in tqdm(range(0, len(tasks), batch_size), desc="Generating rubrics"):
        batch = tasks[i : i + batch_size]
        results = await asyncio.gather(
            *[
                generate_rubric_async(
                    instruction=inst, diagnosis_chosen=cd,
                    diagnosis_rejected=rd, instruction_id=iid, pair_id=pid
                )
                for inst, cd, rd, iid, pid in batch
            ],
            return_exceptions=True,
        )
        for (_, _, _, _, pid), result in zip(batch, results):
            stats["processed"] += 1
            if isinstance(result, Exception):
                logger.error(f"Exception ({pid}): {repr(result)}")
                stats["failed"] += 1
            elif result:
                append_jsonl(output_jsonl, result.to_dict())
                stats["success"] += 1
            else:
                stats["failed"] += 1

    return stats


# ============================================================================
# Main
# ============================================================================

async def main_async(args: argparse.Namespace) -> None:
    init_dirs(os.path.dirname(args.output_jsonl) if os.path.dirname(args.output_jsonl) else ".")

    pairs_data: List[Dict[str, Any]] = []
    if args.pairs_file and os.path.exists(args.pairs_file):
        pairs_data = load_jsonl(args.pairs_file)
        logger.info(f"Loaded {len(pairs_data)} pairs from {args.pairs_file}")

    logger.info("Resume: already-processed pair_ids are skipped automatically")
    stats = await generate_rubric_batch(
        pairs_data, args.diagnosis_jsonl, args.output_jsonl, batch_size=args.batch_size
    )

    logger.info("=" * 50)
    logger.info("Rubric generation complete")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  output: {args.output_jsonl}")
    logger.info("=" * 50)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="StepB: Discriminative Rubric Generation")
    parser.add_argument("--diagnosis_jsonl", type=str, default=DIAGNOSIS_JSONL, help="StepA output JSONL")
    parser.add_argument("--pairs_file", type=str, default=None, help="Original pairwise input JSONL")
    parser.add_argument("--output_jsonl", type=str, default=OUTPUT_JSONL, help="Output JSONL")
    parser.add_argument("--batch_size", type=int, default=10, help="Concurrent API call batch size")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
