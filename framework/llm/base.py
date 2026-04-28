"""
framework/llm/base.py — Provider-agnostic LLM caller base class.

Defines:
    LLMResponse       — structured response dataclass
    LLMCallerBase     — abstract base class for all provider implementations
    extract_sql       — provider-agnostic SQL extraction (preserves the
                        existing llm_caller.extract_sql logic)
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured response from any LLM provider call.

    Mirrors framework/llm_caller.LLMResponse for backwards compatibility
    with existing experiment_runner / metrics code, with one addition
    (provider) so multi-provider results can be disambiguated downstream.
    """
    sql: str                   # Extracted SQL query
    raw_response: str          # Full text returned by the model
    input_tokens: int          # Prompt tokens consumed
    output_tokens: int         # Completion tokens generated
    latency_ms: float          # Wall-clock latency in milliseconds
    model: str                 # Provider's canonical model id
    success: bool              # Whether the call succeeded
    error: str = ""            # Error message if success is False
    provider: str = ""         # "anthropic" | "openai" | "google"


# ---------------------------------------------------------------------------
# LLMCallerBase
# ---------------------------------------------------------------------------

class LLMCallerBase(ABC):
    """Abstract base class for provider-specific LLM callers.

    Concrete subclasses implement `call(prompt, system)` and report results
    in the provider-agnostic `LLMResponse` shape.

    Subclasses SHOULD:
      - Read API keys from environment variables (never hard-coded)
      - Implement exponential-backoff retry on rate-limit / transient errors
      - Use temperature=0.0 by default (deterministic for reproducibility)
      - Pin the model id provided at construction time
    """

    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MAX_RETRIES = 4
    DEFAULT_RETRY_BASE_DELAY = 1.5

    PROVIDER: str = ""  # Subclass MUST set

    def __init__(
        self,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    ) -> None:
        if not self.PROVIDER:
            raise NotImplementedError(
                f"{type(self).__name__} must set PROVIDER class attribute."
            )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        logger.info(
            "%s initialized: model=%s, max_tokens=%d, temperature=%.2f, max_retries=%d",
            type(self).__name__, self.model, self.max_tokens, self.temperature, self.max_retries,
        )

    @abstractmethod
    def call(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """Send `prompt` (with optional `system` message) and return a structured response."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Provider-agnostic SQL extraction
#
# Lifted verbatim (modulo formatting) from framework/llm_caller.py so that
# every provider returns SQL extracted using identical logic. This keeps
# cross-provider comparisons on the SQL extractor axis controlled.
# ---------------------------------------------------------------------------

def extract_sql(response: str) -> str:
    """Extract a SQL query from a raw model response.

    Handles the four common formats:
      1. Markdown code fences (```sql ... ``` or ``` ... ```)
      2. Raw SQL (response starts with SELECT, WITH, INSERT, ...)
      3. SQL embedded in explanatory text
      4. SQL without semicolon followed by natural-language explanation
    """
    if not response or not response.strip():
        return ""

    text = response.strip()

    # Strategy 1: markdown code fences
    fence_pattern = re.compile(
        r"```(?:sql|clickhouse|SQL)?\s*\n?(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    fence_matches = fence_pattern.findall(text)
    if fence_matches:
        sql = max(fence_matches, key=len).strip()
        if sql:
            return _clean_sql(sql)

    # Strategy 2: response starts with SQL
    sql_keywords = re.compile(
        r"^\s*(SELECT|WITH|INSERT|CREATE|ALTER|DROP|EXPLAIN|SHOW|DESCRIBE|SET)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    if sql_keywords.match(text):
        sql = _extract_leading_sql(text)
        return _clean_sql(sql)

    # Strategy 3: SQL embedded in text (terminated by semicolon)
    sql_block_pattern = re.compile(r"((?:SELECT|WITH)\b.*?;)", re.DOTALL | re.IGNORECASE)
    sql_blocks = sql_block_pattern.findall(text)
    if sql_blocks:
        sql = max(sql_blocks, key=len).strip()
        return _clean_sql(sql)

    # Strategy 4: SQL without semicolon
    sql_nosemi_pattern = re.compile(r"((?:SELECT|WITH)\b.+)", re.DOTALL | re.IGNORECASE)
    sql_nosemi = sql_nosemi_pattern.findall(text)
    if sql_nosemi:
        sql = sql_nosemi[0].strip()
        lines = sql.split("\n")
        sql_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and not _looks_like_sql_line(stripped):
                if sql_lines and any("SELECT" in l.upper() for l in sql_lines):
                    break
            sql_lines.append(line)
        sql = "\n".join(sql_lines).strip()
        return _clean_sql(sql)

    return _clean_sql(text)


def _clean_sql(sql: str) -> str:
    if not sql:
        return ""
    sql = sql.strip()
    sql = sql.rstrip(";").strip()
    sql = re.sub(r"\n{3,}", "\n\n", sql)
    return sql


def _extract_leading_sql(text: str) -> str:
    lines = text.split("\n")
    sql_lines: list[str] = []
    blank_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_count += 1
            if blank_count >= 2 and sql_lines:
                break
            sql_lines.append(line)
            continue
        blank_count = 0
        if _looks_like_sql_line(stripped):
            sql_lines.append(line)
        elif sql_lines:
            break
        else:
            sql_lines.append(line)
    return "\n".join(sql_lines).strip()


def _looks_like_sql_line(line: str) -> bool:
    sql_indicators = re.compile(
        r"^\s*("
        r"SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|CROSS|"
        r"ON|AND|OR|NOT|IN|EXISTS|BETWEEN|LIKE|IS|NULL|"
        r"GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|UNION|"
        r"WITH|AS|CASE|WHEN|THEN|ELSE|END|"
        r"INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|"
        r"COUNT|SUM|AVG|MIN|MAX|"
        r"to\w+\(|array\w+\(|if\(|multiIf\(|"
        r"--.*|"
        r"[(),;`'\"\d*]|"
        r"\w+\.\w+"
        r")",
        re.IGNORECASE,
    )
    if sql_indicators.match(line):
        return True
    continuation = re.compile(r"^\s*\w[\w.]*(?:\s+AS\s+\w+)?\s*,?\s*$", re.IGNORECASE)
    if continuation.match(line):
        return True
    return False
