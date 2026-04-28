"""
framework/llm/openai_caller.py — OpenAI SDK wrapper.

Reads OPENAI_API_KEY from the environment. Uses the Responses API for
GPT-5.x models which support reasoning effort and structured output.
Falls back to chat.completions if the Responses API is unavailable for
the chosen model.

Per memory feedback_no_meta_cli_for_datapup.md: no Meta CLI invocation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from .base import LLMCallerBase, LLMResponse, extract_sql

logger = logging.getLogger(__name__)


class OpenAICaller(LLMCallerBase):
    """OpenAI API wrapper for cross-provider evaluation."""

    PROVIDER = "openai"

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Set it to your personal OpenAI API key."
            )
        if os.environ.get("OPENAI_BASE_URL"):
            raise EnvironmentError(
                "OPENAI_BASE_URL is set. The DataPup paper requires direct calls to "
                "api.openai.com under personal credentials. Unset OPENAI_BASE_URL."
            )
        self.client = OpenAI(api_key=api_key)

    def call(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # GPT-5.x family uses max_completion_tokens; older models used max_tokens.
        # Use max_completion_tokens which is forward-compatible on GPT-5.x.
        request_kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
        }
        # GPT-5.x reasoning models do not support custom temperature; only set
        # temperature for non-reasoning models.
        if not self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            request_kwargs["temperature"] = self.temperature

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                raw_text = response.choices[0].message.content or ""

                # Token usage
                usage = response.usage
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0

                return LLMResponse(
                    sql=extract_sql(raw_text),
                    raw_response=raw_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=round(elapsed_ms, 2),
                    model=self.model,
                    success=True,
                    provider=self.PROVIDER,
                )

            except RateLimitError as e:
                last_error = f"Rate limited (attempt {attempt}/{self.max_retries}): {e}"
                logger.warning(last_error)
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))

            except InternalServerError as e:
                last_error = f"Server error (attempt {attempt}/{self.max_retries}): {e}"
                logger.warning(last_error)
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))

            except APIConnectionError as e:
                last_error = f"Connection error (attempt {attempt}/{self.max_retries}): {e}"
                logger.warning(last_error)
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))

            except APIStatusError as e:
                status = getattr(e, "status_code", None)
                if status in (502, 503, 504, 529):
                    last_error = f"Transient {status} (attempt {attempt}/{self.max_retries}): {e}"
                    logger.warning(last_error)
                    if attempt < self.max_retries:
                        time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))
                else:
                    return _failed(self.model, self.PROVIDER, f"API error {status}: {e}", start_time)

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
