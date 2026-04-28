"""
framework/llm/google_caller.py — Google Gemini SDK wrapper.

Reads GOOGLE_API_KEY (or GEMINI_API_KEY as fallback) from the environment.
Uses the modern google-genai SDK (`from google import genai`) — NOT the
legacy `google.generativeai` package.

Per memory feedback_no_meta_cli_for_datapup.md: no Meta CLI invocation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from .base import LLMCallerBase, LLMResponse, extract_sql

logger = logging.getLogger(__name__)


class GoogleCaller(LLMCallerBase):
    """Google Gemini API wrapper for cross-provider evaluation."""

    PROVIDER = "google"

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set."
            )
        self.client = genai.Client(api_key=api_key)

    def call(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        config_kwargs: dict = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system:
            config_kwargs["system_instruction"] = system
        config = genai_types.GenerateContentConfig(**config_kwargs)

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                raw_text = response.text or ""

                # Token usage from the response metadata
                usage = getattr(response, "usage_metadata", None)
                input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

                return LLMResponse(
                    sql=extract_sql(raw_text),
                    raw_response=raw_text,
                    input_tokens=int(input_tokens or 0),
                    output_tokens=int(output_tokens or 0),
                    latency_ms=round(elapsed_ms, 2),
                    model=self.model,
                    success=True,
                    provider=self.PROVIDER,
                )

            except genai_errors.ClientError as e:
                # 429 rate limit, 4xx client errors
                code = getattr(e, "code", None)
                if code == 429:
                    last_error = f"Rate limited (attempt {attempt}/{self.max_retries}): {e}"
                    logger.warning(last_error)
                    if attempt < self.max_retries:
                        time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))
                else:
                    return _failed(self.model, self.PROVIDER, f"Client error {code}: {e}", start_time)

            except genai_errors.ServerError as e:
                last_error = f"Server error (attempt {attempt}/{self.max_retries}): {e}"
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
