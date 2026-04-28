"""
framework/llm/anthropic_caller.py — Anthropic SDK wrapper.

Reads ANTHROPIC_API_KEY from the environment. Refuses to use the Meta CLI
binary at /usr/local/bin/claude per memory feedback_no_meta_cli_for_datapup.md
— this caller talks directly to api.anthropic.com under personal credentials.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import anthropic

from .base import LLMCallerBase, LLMResponse, extract_sql

logger = logging.getLogger(__name__)


class AnthropicCaller(LLMCallerBase):
    """Anthropic Claude API wrapper for cross-provider evaluation."""

    PROVIDER = "anthropic"

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Set it to your personal Anthropic API key. "
                "Do NOT use the Meta CLI binary."
            )
        # Refuse to silently route through any non-Anthropic base URL.
        # Custom proxies are deliberately disallowed for this paper.
        if os.environ.get("ANTHROPIC_BASE_URL"):
            raise EnvironmentError(
                "ANTHROPIC_BASE_URL is set. The DataPup paper requires direct calls to "
                "api.anthropic.com under personal credentials. Unset ANTHROPIC_BASE_URL."
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    def call(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        request_kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request_kwargs["system"] = system

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                response = self.client.messages.create(**request_kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                raw_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        raw_text += block.text

                return LLMResponse(
                    sql=extract_sql(raw_text),
                    raw_response=raw_text,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=round(elapsed_ms, 2),
                    model=self.model,
                    success=True,
                    provider=self.PROVIDER,
                )

            except anthropic.RateLimitError as e:
                last_error = f"Rate limited (attempt {attempt}/{self.max_retries}): {e}"
                logger.warning(last_error)
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))

            except anthropic.InternalServerError as e:
                last_error = f"Server error (attempt {attempt}/{self.max_retries}): {e}"
                logger.warning(last_error)
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))

            except anthropic.APIStatusError as e:
                if getattr(e, "status_code", None) == 529:
                    last_error = f"API overloaded (attempt {attempt}/{self.max_retries}): {e}"
                    logger.warning(last_error)
                    if attempt < self.max_retries:
                        time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))
                else:
                    return _failed(self.model, self.PROVIDER, f"API error: {e}", start_time)

            except anthropic.APIConnectionError as e:
                last_error = f"Connection error (attempt {attempt}/{self.max_retries}): {e}"
                logger.warning(last_error)
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))

            except Exception as e:
                return _failed(self.model, self.PROVIDER, f"Unexpected error: {type(e).__name__}: {e}", start_time)

        return _failed(self.model, self.PROVIDER,
                       f"All {self.max_retries} retry attempts exhausted. {last_error}", time.perf_counter())


def _failed(model: str, provider: str, error: str, start_time: float) -> LLMResponse:
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.error(error)
    return LLMResponse(
        sql="", raw_response="", input_tokens=0, output_tokens=0,
        latency_ms=round(max(elapsed_ms, 0.0), 2), model=model,
        success=False, error=error, provider=provider,
    )
