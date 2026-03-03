#!/usr/bin/env python3
"""
Async OpenAI-compatible LLM client and rubric pipeline data types.

Used by rubric_pipeline/ (StepA diagnosis + StepB rubric generation).

Environment variables:
  OPENAI_BASE_URL   – API base URL        (default: https://api.openai.com/v1)
  OPENAI_API_KEY    – API key             (required)
  MODEL_NAME        – Model name          (required)
  MAX_RETRIES       – Max retry attempts  (default: 5)
  RETRY_DELAY       – Base retry delay s  (default: 6.0)
  TIMEOUT           – Request timeout s   (default: 120)
  MAX_CONCURRENCY   – Max parallel calls  (default: 32)
"""

import asyncio
import logging
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("OPENAI_BASE_URL", "")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3-235B-A22B-Instruct-2507")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "6.0"))
TIMEOUT = int(os.getenv("TIMEOUT", "120"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "32"))

# Singleton async client
_ASYNC_CLIENT: Optional[AsyncOpenAI] = None
_CLIENT_LOCK = asyncio.Lock()


async def get_async_client() -> AsyncOpenAI:
    """Return a shared, lazily-initialized AsyncOpenAI client."""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is not None:
        return _ASYNC_CLIENT
    async with _CLIENT_LOCK:
        if _ASYNC_CLIENT is None:
            if not API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Please export OPENAI_API_KEY before running."
                )
            _ASYNC_CLIENT = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _ASYNC_CLIENT


# ============================================================================
# Async LLM call (with exponential-backoff retry)
# ============================================================================

async def llm_call(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    retry_count: int = 0,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> str:
    """
    Call the LLM API asynchronously with automatic retry on transient errors.

    Args:
        system_prompt:  System message content.
        user_prompt:    User message content.
        max_tokens:     Maximum tokens to generate.
        temperature:    Sampling temperature.
        retry_count:    Current retry count (used internally for recursion).
        semaphore:      Optional semaphore to limit concurrency.

    Returns:
        The model's text response, or an error message string on failure.
    """
    if not MODEL_NAME:
        raise ValueError(
            "MODEL_NAME environment variable is not set. "
            "Please export MODEL_NAME before running."
        )

    client = await get_async_client()
    if semaphore is None:
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        async with semaphore:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=TIMEOUT,
            )
        return response.choices[0].message.content

    except Exception as e:
        # Determine HTTP status code if available
        status_code = getattr(e, "status_code", None)
        if status_code is None:
            resp_obj = getattr(e, "response", None)
            if resp_obj:
                status_code = getattr(resp_obj, "status_code", None)

        # Retry on transient HTTP errors
        if status_code and retry_count < MAX_RETRIES and status_code in {429, 500, 502, 503, 504}:
            delay = RETRY_DELAY * (2 ** retry_count)
            logger.warning(
                f"API error ({status_code}), retrying in {delay:.1f}s "
                f"({retry_count + 1}/{MAX_RETRIES}): {e}"
            )
            await asyncio.sleep(delay)
            return await llm_call(system_prompt, user_prompt, max_tokens, temperature, retry_count + 1, semaphore)

        # Retry on network / timeout errors
        err_str = str(e).lower()
        if retry_count < MAX_RETRIES and any(kw in err_str for kw in ("timeout", "connection", "network")):
            delay = RETRY_DELAY * (2 ** retry_count)
            logger.warning(
                f"Network error ({type(e).__name__}), retrying in {delay:.1f}s "
                f"({retry_count + 1}/{MAX_RETRIES}): {e}"
            )
            await asyncio.sleep(delay)
            return await llm_call(system_prompt, user_prompt, max_tokens, temperature, retry_count + 1, semaphore)

        err_msg = f"Unhandled error: {e}"
        logger.error(f"{err_msg}\n{traceback.format_exc()}")
        return err_msg


# ============================================================================
# Data structures
# ============================================================================

class DiagnosisResult:
    """Encapsulates a single StepA diagnosis output."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.instruction_id: str = data.get("instruction_id", "")
        self.pair_id: str = data.get("pair_id", "")
        self.label: str = data.get("label", "")  # "chosen" or "rejected"
        self.criteria_candidates: List[str] = data.get("criteria_candidates", [])
        self.findings: List[Dict[str, Any]] = data.get("findings", [])
        self.overall_summary: str = data.get("overall_summary", "")
        self.confidence: float = data.get("confidence", 0.0)
        self.raw_data = data

    def to_dict(self) -> Dict[str, Any]:
        return self.raw_data

    def validate(self) -> Tuple[bool, List[str]]:
        """Return (is_valid, list_of_errors)."""
        errors: List[str] = []

        if not self.instruction_id:
            errors.append("Missing instruction_id")
        if self.label not in {"chosen", "rejected"}:
            errors.append(f"label must be 'chosen' or 'rejected', got: {self.label!r}")

        for idx, finding in enumerate(self.findings):
            if not isinstance(finding, dict):
                errors.append(f"finding[{idx}] is not a dict")
                continue
            status = finding.get("status", "")
            if status in {"fail", "partial"}:
                if not finding.get("evidence"):
                    errors.append(f"finding[{idx}] (status={status}) missing evidence")
                if finding.get("severity", -1) < 0:
                    errors.append(f"finding[{idx}] (status={status}) missing severity")
            if not finding.get("claim"):
                errors.append(f"finding[{idx}] missing claim")
            if not finding.get("instruction_anchor"):
                errors.append(f"finding[{idx}] missing instruction_anchor")

        return len(errors) == 0, errors


class RubricPairResult:
    """Encapsulates a single StepB rubric generation output."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.instruction_id: str = data.get("instruction_id", "")
        self.pair_id: str = data.get("pair_id", "")
        self.hard_rules: List[Dict[str, Any]] = data.get("hard_rules", [])
        self.principles: List[Dict[str, Any]] = data.get("principles", [])
        self.pair_consistency_check: Dict[str, Any] = data.get("pair_consistency_check", {})
        self.raw_data = data

    def to_dict(self) -> Dict[str, Any]:
        return self.raw_data

    def validate(self) -> Tuple[bool, List[str]]:
        """Return (is_valid, list_of_errors)."""
        errors: List[str] = []

        if not self.instruction_id:
            errors.append("Missing instruction_id")
        if not self.pair_id:
            errors.append("Missing pair_id")
        if not self.hard_rules:
            errors.append("Missing hard_rules")

        for idx, rule in enumerate(self.hard_rules):
            if not isinstance(rule, dict):
                errors.append(f"hard_rule[{idx}] is not a dict")
                continue
            if not rule.get("rule_id"):
                errors.append(f"hard_rule[{idx}] missing rule_id")
            if rule.get("type") not in {"must", "forbid"}:
                errors.append(f"hard_rule[{idx}] type must be 'must' or 'forbid'")
            if not rule.get("criterion"):
                errors.append(f"hard_rule[{idx}] missing criterion")
            if not rule.get("test"):
                errors.append(f"hard_rule[{idx}] missing test")

        for idx, principle in enumerate(self.principles):
            if not isinstance(principle, dict):
                errors.append(f"principle[{idx}] is not a dict")
                continue
            if not principle.get("principle_id"):
                errors.append(f"principle[{idx}] missing principle_id")
            if not principle.get("description"):
                errors.append(f"principle[{idx}] missing description")

        cc = self.pair_consistency_check
        if cc:
            expected = cc.get("expected_winner", "")
            # Accept both A/B (current format) and chosen/rejected (legacy)
            if expected not in {"A", "B", "chosen", "rejected", ""}:
                errors.append(
                    f"expected_winner must be 'A'/'B' (or legacy 'chosen'/'rejected'), got: {expected!r}"
                )

        return len(errors) == 0, errors
