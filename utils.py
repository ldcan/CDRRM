#!/usr/bin/env python3
"""
Shared file I/O and checkpoint utilities for the CDRRM pipeline.

Functions:
  load_jsonl       – load all records from a JSONL file
  iter_jsonl       – iterate a JSONL file line-by-line (memory-efficient)
  save_jsonl       – write a list of dicts to a JSONL file
  append_jsonl     – append a single record to a JSONL file
  load_json        – load a JSON file
  save_json        – atomically save a JSON file
  init_dirs        – create directories
  load_checkpoint  – load a JSON checkpoint
  save_checkpoint  – save a JSON checkpoint
  extract_json_from_text – extract the first JSON object from free text
"""

import json
import logging
import os
import re
import traceback
from typing import Any, Dict, Generator, List, Optional

# Library modules should not configure the root logger; add a NullHandler
# so callers that don't configure logging won't see "No handler found" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic console logging. Call this from script entry points."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# ============================================================================
# JSONL I/O
# ============================================================================

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load all records from a JSONL file. Returns [] if the file doesn't exist."""
    data: List[Dict[str, Any]] = []
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error at line {line_num}: {e}")
    return data


def iter_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Iterate a JSONL file line-by-line, skipping blank/malformed lines."""
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error at line {line_num}: {e}")


def save_jsonl(file_path: str, data: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to a JSONL file, creating parent dirs as needed."""
    _ensure_parent(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(file_path: str, item: Dict[str, Any]) -> None:
    """Append a single record to a JSONL file (safe within a single process)."""
    _ensure_parent(file_path)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


# ============================================================================
# JSON I/O
# ============================================================================

def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file. Returns None if the file doesn't exist."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path: str, data: Dict[str, Any], indent: int = 2) -> None:
    """Atomically save a dict as a JSON file (write to tmp then rename)."""
    _ensure_parent(file_path)
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    os.replace(tmp_path, file_path)


# ============================================================================
# Checkpoint helpers
# ============================================================================

def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load a JSON checkpoint file. Returns {} if not found."""
    return load_json(checkpoint_file) or {}


def save_checkpoint(checkpoint_file: str, checkpoint: Dict[str, Any]) -> None:
    """Save a JSON checkpoint file."""
    save_json(checkpoint_file, checkpoint)


# ============================================================================
# Directory helpers
# ============================================================================

def init_dirs(*dirs: str) -> None:
    """Create directories (including parents), silently ignoring existing ones."""
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)


def _ensure_parent(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ============================================================================
# JSON extraction from free text
# ============================================================================

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from text that may contain prose or code fences.

    Tries in order:
      1. Direct JSON parse of the full string.
      2. JSON inside a ```json ... ``` or ``` ... ``` code block.
      3. First '{' to last '}' in the string.

    Returns the parsed dict, or None if all attempts fail.
    """
    # 1. Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Code block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None
