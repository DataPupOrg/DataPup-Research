"""
framework/llm/__init__.py — Multi-provider LLM caller package.

Exports both SDK-based and CLI-based callers; the factory picks based on
the model entry's `transport` field in cross_provider_models.yaml.
"""

from __future__ import annotations

from .anthropic_caller import AnthropicCaller
from .base import LLMCallerBase, LLMResponse, extract_sql
from .cli_caller import (
    ClaudeCLICaller,
    CodexCLICaller,
    GeminiCLICaller,
)
from .factory import get_caller, list_models, list_tiers, load_model_config
from .google_caller import GoogleCaller
from .openai_caller import OpenAICaller

__all__ = [
    # Base
    "LLMResponse",
    "LLMCallerBase",
    "extract_sql",
    # SDK callers
    "AnthropicCaller",
    "OpenAICaller",
    "GoogleCaller",
    # CLI callers
    "ClaudeCLICaller",
    "CodexCLICaller",
    "GeminiCLICaller",
    # Factory
    "get_caller",
    "load_model_config",
    "list_models",
    "list_tiers",
]
