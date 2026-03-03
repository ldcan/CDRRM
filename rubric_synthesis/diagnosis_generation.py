#!/usr/bin/env python3
"""
StepA: Structured Diagnosis Generation.

Calls the OpenAI API to produce a structured diagnosis for each (chosen, rejected)
response pair, then saves results to a JSONL file.

Reads from:
  --input_file    JSONL with fields: instruction, chosen, rejected (or response_a/b + winner)

Writes to:
  --output_jsonl  JSONL with DiagnosisResult records

Resume support: already-processed (pair_id, label) pairs are skipped on restart.
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm  # type: ignore

# Shared utilities live one level up (cdrrm/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts import STEPA_DIAGNOSIS_SYSTEM, STEPA_DIAGNOSIS_USER_TEMPLATE
from utils import load_jsonl, init_dirs, append_jsonl, iter_jsonl, extract_json_from_text, setup_logging
from llm_client import llm_call, DiagnosisResult

logger = logging.getLogger(__name__)

OUTPUT_JSONL = "output/diagnosis.jsonl"


# ============================================================================
# Data normalization
# ============================================================================

def normalize_pair_data(pair: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """
    Normalize input pair to a standard dict with 'chosen' and 'rejected' fields.

    Supports two input formats:
      - Standard:     {'instruction', 'chosen', 'rejected', ...}
      - OpenRubrics:  {'instruction', 'response_a', 'response_b', 'winner', ...}
    """
    instruction = pair.get("instruction", "")
    if not instruction:
        return None

    normalized: Dict[str, Any] = {"instruction": instruction}

    if "chosen" in pair and "rejected" in pair:
        normalized["chosen"] = pair["chosen"]
        normalized["rejected"] = pair["rejected"]
    elif "response_a" in pair and "response_b" in pair:
        response_a = pair.get("response_a", "")
        response_b = pair.get("response_b", "")
        if not response_a or not response_b:
            return None
        winner = pair.get("winner", "").upper()
        if winner in {"A", "RESPONSE_A", "A_WIN"}:
            normalized["chosen"] = response_a
            normalized["rejected"] = response_b
        elif winner in {"B", "RESPONSE_B", "B_WIN"}:
            normalized["chosen"] = response_b
            normalized["rejected"] = response_a
        else:
            logger.warning(f"Pair {idx + 1}: invalid winner {winner!r}, defaulting response_a as chosen")
            normalized["chosen"] = response_a
            normalized["rejected"] = response_b
    else:
        return None

    if not normalized.get("chosen") or not normalized.get("rejected"):
        return None

    normalized["instruction_id"] = pair.get("instruction_id") or pair.get("id") or f"inst_{idx + 1}"
    normalized["pair_id"] = pair.get("pair_id") or pair.get("id") or f"line_{idx + 1}"
    normalized["instruction_keypoints"] = pair.get("instruction_keypoints")
    return normalized


# ============================================================================
# Core async functions
# ============================================================================

async def generate_diagnosis_async(
    instruction: str,
    answer: str,
    label: str,
    instruction_id: str = "",
    pair_id: str = "",
) -> Optional[DiagnosisResult]:
    """Generate a single diagnosis via the LLM API."""
    system_prompt = STEPA_DIAGNOSIS_SYSTEM
    user_prompt = STEPA_DIAGNOSIS_USER_TEMPLATE.format(
        instruction=instruction, answer=answer
    )

    try:
        response_text = await llm_call(system_prompt, user_prompt, max_tokens=4096)
        diagnosis_dict = extract_json_from_text(response_text)
        if not diagnosis_dict:
            logger.error(f"Failed to extract JSON for {pair_id}_{label}")
            logger.debug(f"Response snippet: {response_text[:500]}")
            return None

        diagnosis_dict.update(
            instruction_id=instruction_id, pair_id=pair_id, label=label
        )
        diagnosis = DiagnosisResult(diagnosis_dict)
        is_valid, errors = diagnosis.validate()
        if not is_valid:
            logger.error(f"Diagnosis validation failed ({pair_id}_{label}): {errors}")
            return None
        return diagnosis

    except Exception as e:
        logger.error(f"Error generating diagnosis ({pair_id}_{label}): {e}")
        return None


async def generate_diagnosis_batch(
    pairs: List[Dict[str, Any]],
    output_jsonl: str,
    batch_size: int = 10,
) -> Dict[str, Any]:
    """Batch-generate diagnosis results with resume support."""
    stats: Dict[str, Any] = {
        "total": len(pairs) * 2,
        "processed": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
    }

    # Resume: collect already-processed (pair_id, label) keys
    done_keys: set = set()
    for obj in iter_jsonl(output_jsonl):
        pid = str(obj.get("pair_id", "")).strip()
        lbl = str(obj.get("label", "")).strip()
        if pid and lbl:
            done_keys.add(f"{pid}::{lbl}")

    # Build task list
    tasks: List[Tuple[str, str, str, str, str]] = []
    for idx, pair in enumerate(pairs):
        normalized = normalize_pair_data(pair, idx)
        if not normalized:
            logger.warning(f"Skipping invalid pair at line {idx + 1}")
            stats["skipped"] += 2
            continue

        instruction = normalized["instruction"]
        pair_id = normalized["pair_id"]
        instruction_id = normalized["instruction_id"]

        for label, answer in [("chosen", normalized["chosen"]), ("rejected", normalized["rejected"])]:
            key = f"{pair_id}::{label}"
            if key in done_keys:
                stats["skipped"] += 1
            else:
                tasks.append((label, instruction, answer, instruction_id, pair_id))

    logger.info(f"Tasks: {len(tasks)} to process, {stats['skipped']} already done")

    for i in tqdm(range(0, len(tasks), batch_size), desc="Generating diagnoses"):
        batch = tasks[i : i + batch_size]
        results = await asyncio.gather(
            *[
                generate_diagnosis_async(
                    instruction=inst, answer=ans,
                    label=lbl, instruction_id=inst_id, pair_id=pid
                )
                for lbl, inst, ans, inst_id, pid in batch
            ],
            return_exceptions=True,
        )
        for (lbl, _, _, _, pid), result in zip(batch, results):
            stats["processed"] += 1
            if isinstance(result, Exception):
                logger.error(f"Exception ({pid}_{lbl}): {repr(result)}")
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

    logger.info(f"Loading input: {args.input_file}")
    pairs = load_jsonl(args.input_file)
    if not pairs:
        logger.error("Input data is empty")
        return

    logger.info(f"Loaded {len(pairs)} pairs (resume: already-processed pairs are skipped automatically)")
    stats = await generate_diagnosis_batch(pairs, args.output_jsonl, batch_size=args.batch_size)

    logger.info("=" * 50)
    logger.info("Diagnosis generation complete")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  output: {args.output_jsonl}")
    logger.info("=" * 50)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="StepA: Structured Diagnosis Generation")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_jsonl", type=str, default=OUTPUT_JSONL, help="Output JSONL file")
    parser.add_argument("--batch_size", type=int, default=10, help="Concurrent API call batch size")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
