#!/usr/bin/env python3
"""
Synthesize pairwise judge SFT data via async LLM API calls.

Input:
  - input_file:   Raw pairwise data (instruction, response_a, response_b, winner)
  - rubrics_file: Rubrics file from eval pipeline (contains formatted_rubric field)

The script matches rubrics to input data by instruction_id or sid, then calls
rubric-judge LLM to generate SFT training samples.

API credentials are read from environment variables (shared with rubric_pipeline):
  OPENAI_BASE_URL   – API base URL        (default: https://api.openai.com/v1)
  OPENAI_API_KEY    – API key             (required)
  MODEL_NAME        – LLM model name     (required)
  MAX_CONCURRENCY   – parallel calls     (default: 32)

Features: async batch processing, checkpoint/resume, filtered SFT data export.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import llm_call, MAX_CONCURRENCY  
from prompts import (  
    RUBRIC_JUDGE_SYSTEM,
    RUBRIC_JUDGE_USER_TEMPLATE,
)
from utils import init_dirs, save_json, save_jsonl, setup_logging  

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = ".judge_checkpoints"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(task_id: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{task_id}.checkpoint.json")


def save_ckpt(task_id: str, processed_idx: int, results: List[str]) -> None:
    """Persist checkpoint to disk."""
    init_dirs(CHECKPOINT_DIR)
    # Ensure results is a list of strings
    if not isinstance(results, list):
        logger.error(f"Results is not a list: {type(results)}")
        return
    # Validate that all results are strings
    results = [str(r) if not isinstance(r, str) else r for r in results]
    
    checkpoint_data = {
        "processed_idx": processed_idx,
        "results": results,
        "results_count": len(results),
        "timestamp": time.time(),
    }
    
    with open(_ckpt_path(task_id), "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False)
    logger.info(f"Checkpoint saved: {processed_idx + 1}/{len(results)} processed")


def load_ckpt(task_id: str) -> Tuple[int, List[str]]:
    """Load checkpoint; returns (-1, []) if not found or corrupted."""
    path = _ckpt_path(task_id)
    if not os.path.exists(path):
        return -1, []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed_idx = data.get("processed_idx", -1)
        results = data.get("results", [])
        
        # Validate results is a list
        if not isinstance(results, list):
            logger.warning(f"Checkpoint results is not a list: {type(results)}, reinitializing")
            return -1, []
        
        # Ensure all results are strings
        results = [str(r) if not isinstance(r, str) else r for r in results]
        
        saved_count = data.get("results_count", len(results))
        if saved_count != len(results):
            logger.warning(f"Checkpoint results count mismatch: saved={saved_count}, actual={len(results)}")
        
        logger.info(f"Loaded checkpoint: {processed_idx + 1}/{len(results)} processed")
        return processed_idx, results
    except Exception as e:
        logger.warning(f"Corrupted checkpoint, ignoring: {e}")
        return -1, []


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_input(input_file: str, rubrics_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load pairwise data from input_file and optionally match rubrics from rubrics_file.
    
    The matching is done by line index: rubrics_file中的sid对应原始数据的行号（索引，从0开始）。
    
    Args:
        input_file: Raw pairwise data file (JSONL or Parquet), no id fields required
        rubrics_file: Optional rubrics file with formatted_rubric field, sid corresponds to line index
        
    Returns:
        List of data items with formatted_rubric attached (if rubrics_file provided)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    ext = os.path.splitext(input_file)[1].lower()
    data: List[Dict[str, Any]] = []

    # Load input data
    if ext in (".jsonl", ".json"):
        with open(input_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {i}: {e}")
    elif ext == ".parquet":
        if pd is None:
            raise ImportError("pandas is required for Parquet files: pip install pandas")
        df = pd.read_parquet(input_file)
        data = df.to_dict("records")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    logger.info(f"Loaded {len(data)} samples from {input_file}")

    # Load and match rubrics if provided
    # rubrics_file中的sid对应原始数据的行号（索引）
    if rubrics_file and os.path.exists(rubrics_file):
        logger.info(f"Loading rubrics from {rubrics_file} ...")
        rubrics: Dict[int, str] = {}  # key: line index (sid), value: formatted_rubric
        
        with open(rubrics_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    sid = r.get("sid", "")
                    formatted = r.get("formatted_rubric", "")
                    
                    if formatted and sid:
                        # sid should be a string representation of line index
                        try:
                            idx = int(sid)
                            rubrics[idx] = formatted
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid sid in rubrics file: {sid}, skipping")
                except json.JSONDecodeError:
                    continue
        
        matched = 0
        for idx, item in enumerate(data):
            item["formatted_rubric"] = ""
            # Match by line index (sid corresponds to line number in input file)
            if idx in rubrics:
                item["formatted_rubric"] = rubrics[idx]
                matched += 1
        
        logger.info(f"Matched {matched}/{len(data)} samples with rubrics (by line index)")
    else:
        for item in data:
            item.setdefault("formatted_rubric", "")

    # Normalize winner field: extract from extra_info.original_winner if needed
    for item in data:
        if "winner" not in item or not item.get("winner"):
            extra_info = item.get("extra_info", {})
            if isinstance(extra_info, dict):
                original_winner = extra_info.get("original_winner", "")
                if original_winner:
                    item["winner"] = original_winner.upper()

    return data


# ---------------------------------------------------------------------------
# Message builders  (return (system, user) string tuples for llm_call)
# ---------------------------------------------------------------------------

def build_pairwise_message(item: Dict[str, Any]) -> Tuple[str, str]:
    """Build (system, user) strings for rubric-judge pairwise evaluation."""
    user = RUBRIC_JUDGE_USER_TEMPLATE.format(
        instruction=str(item.get("instruction") or "").strip(),
        response_a=str(item.get("response_a") or "").strip(),
        response_b=str(item.get("response_b") or "").strip(),
        rubric=str(item.get("formatted_rubric") or item.get("rubric") or "").strip(),
    )
    return RUBRIC_JUDGE_SYSTEM.strip(), user.strip()


# ---------------------------------------------------------------------------
# Async runner  (delegates to llm_call from llm_client)
# ---------------------------------------------------------------------------

async def _run_all(
    message_pairs: List[Tuple[str, str]],
    batch_size: int,
    task_id: str,
    max_tokens: int,
) -> List[str]:
    total = len(message_pairs)
    start_idx, results = load_ckpt(task_id)
    
    # Ensure results list has correct length
    if len(results) != total:
        logger.info(f"Checkpoint results length ({len(results)}) != total ({total}), reinitializing")
        results = [""] * total
        start_idx = -1
    
    start = start_idx + 1
    if start >= total:
        logger.info(f"All {total} samples already processed, skipping")
        return results

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    pbar = tqdm(total=total, initial=start, desc=f"[{task_id}]", unit="req")
    try:
        for i in range(start, total, batch_size):
            end = min(i + batch_size, total)
            tasks = [
                asyncio.create_task(
                    llm_call(system, user, max_tokens=max_tokens, semaphore=semaphore)
                )
                for system, user in message_pairs[i:end]
            ]
            batch_res = await asyncio.gather(*tasks)
            # Update results at correct indices
            for idx, resp in enumerate(batch_res):
                results[i + idx] = resp
            pbar.update(len(batch_res))
            # Save checkpoint after each batch
            save_ckpt(task_id, end - 1, results)
    finally:
        pbar.close()
    return results


def run_judge(
    message_pairs: List[Tuple[str, str]],
    batch_size: int,
    task_id: str,
    max_tokens: int = 4096,
) -> List[str]:
    """Run async LLM calls, reusing an existing event loop if available."""
    coro = _run_all(message_pairs, batch_size, task_id, max_tokens)
    try:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Result parsing and saving
# ---------------------------------------------------------------------------

def _parse_winner(text: str) -> Optional[str]:
    """Extract predicted winner ('A' or 'B') from judge response text."""
    for pattern in [
        r"Winner:\s*Response\s*([AB])\b",
        r"Winner:\s*([AB])\b",
        r"Final\s+Winner:\s*Response\s*([AB])\b",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def process_pairwise(
    items: List[Dict[str, Any]], responses: List[str], output_dir: str
) -> None:
    """
    Parse pairwise judge responses, compute accuracy, and export SFT data.

    Filtering: keep only samples where (a) the API call succeeded and
    (b) a winner was parsed. Accuracy is computed only on samples with a
    valid ground-truth winner label.
    """
    assert len(items) == len(responses)
    init_dirs(output_dir)

    all_rows: List[Dict] = []
    filtered_rows: List[Dict] = []
    sft_data: List[Dict] = []
    total = correct = f_total = f_correct = 0

    for idx, (item, resp) in enumerate(zip(items, responses)):
        winner_pred = _parse_winner(resp)
        # Get winner from top-level or extra_info.original_winner
        gt = str(item.get("winner", "")).upper().strip()
        if not gt:
            extra_info = item.get("extra_info", {})
            if isinstance(extra_info, dict):
                gt = str(extra_info.get("original_winner", "")).upper().strip()
        
        is_correct = gt in ("A", "B") and winner_pred == gt

        row = {
            "index": idx,
            "sid": item.get("sid", ""),
            "benchmark": item.get("benchmark", ""),
            "instruction": item.get("instruction", ""),
            "response_a": item.get("response_a", ""),
            "response_b": item.get("response_b", ""),
            "rubric": item.get("formatted_rubric", item.get("rubric", "")),
            "winner_gt": gt,
            "winner_pred": winner_pred,
            "is_correct": is_correct,
            "raw_response": resp,
        }
        all_rows.append(row)

        if gt in ("A", "B"):
            total += 1
            if is_correct:
                correct += 1
            if not resp.startswith("ERROR") and winner_pred in ("A", "B"):
                filtered_rows.append(row)
                f_total += 1
                if is_correct:
                    f_correct += 1

    # Generate SFT data from filtered rows
    for row in filtered_rows:
        user = RUBRIC_JUDGE_USER_TEMPLATE.format(
            instruction=row["instruction"],
            response_a=row["response_a"],
            response_b=row["response_b"],
            rubric=row["rubric"],
        )
        sft_data.append({
            "messages": [
                {"role": "system", "content": RUBRIC_JUDGE_SYSTEM.strip()},
                {"role": "user", "content": user.strip()},
                {"role": "assistant", "content": row["raw_response"].strip()},
            ],
            "winner_gt": row["winner_gt"],
            "winner_pred": row["winner_pred"],
            "is_correct": row["is_correct"],
        })

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "filtered_total": f_total,
        "filtered_correct": f_correct,
        "filtered_accuracy": f_correct / f_total if f_total else 0.0,
        "filtered_ratio": f_total / total if total else 0.0,
        "sft_samples": len(sft_data),
    }
    _save_outputs(output_dir, "pairwise", all_rows, filtered_rows, sft_data, metrics)
    acc = correct / total if total else 0.0
    f_acc = f_correct / f_total if f_total else 0.0
    logger.info(
        f"Pairwise: {total} GT-valid, accuracy={acc:.2%}; "
        f"filtered {f_total}, filtered_accuracy={f_acc:.2%}; "
        f"SFT samples: {len(sft_data)}; saved to {output_dir}"
    )


def _save_outputs(
    output_dir: str,
    mode: str,
    all_rows: List[Dict],
    filtered_rows: List[Dict],
    sft_data: List[Dict],
    metrics: Dict,
) -> None:
    save_jsonl(os.path.join(output_dir, f"results_all_{mode}.jsonl"), all_rows)
    save_jsonl(os.path.join(output_dir, f"results_filtered_{mode}.jsonl"), filtered_rows)
    save_jsonl(os.path.join(output_dir, f"sft_data_{mode}.jsonl"), sft_data)
    save_json(os.path.join(output_dir, f"metrics_{mode}.json"), metrics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize pairwise judge SFT data via LLM API calls. "
            "Loads raw pairwise data from input_file and matches rubrics from rubrics_file "
            "by line index (sid in rubrics_file corresponds to line number in input_file). "
            "API credentials are read from env vars: "
            "OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME, MAX_CONCURRENCY."
        )
    )
    parser.add_argument("--input_file", required=True,
                        help="Input JSONL or Parquet file (raw pairwise data)")
    parser.add_argument("--rubrics_file", default=None,
                        help="Rubrics file from eval pipeline (contains formatted_rubric, sid corresponds to input line index)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for output files")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Async batch size (also controls checkpoint frequency)")
    parser.add_argument("--task_id", default="judge_task",
                        help="Task ID for checkpoint file naming (enables resume)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap the number of input samples (useful for debugging)")
    args = parser.parse_args()

    try:
        data = load_input(args.input_file, args.rubrics_file)
        if args.max_samples:
            data = data[: args.max_samples]
        if not data:
            raise ValueError("Input data is empty")

        logger.info(f"Processing {len(data)} samples")
        message_pairs = [build_pairwise_message(item) for item in data]
        responses = run_judge(message_pairs, args.batch_size, args.task_id, max_tokens=4096)
        process_pairwise(data, responses, args.output_dir)
        logger.info("Done.")
    except Exception as e:
        logger.error(f"Failed: {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
