"""
framework/llm/__init__.py — Multi-provider LLM caller package.

Exports:
    LLMResponse        — structured response (provider-agnostic)
    LLMCallerBase      — abstract base class
    AnthropicCaller    — Anthropic SDK wrapper
    OpenAICaller       — OpenAI SDK wrapper
    GoogleCaller       — Google generativeai SDK wrapper
    get_caller         — factory: load by model key from cross_provider_models.yaml
    extract_sql        — provider-agnostic SQL extraction utility

Part of the cross-provider evaluation framework for:
    "DataPup: Engineering Reliable Text-to-SQL for Analytical Databases"
"""

from __future__ import annotations

from .base import LLMResponse, LLMCallerBase, extract_sql
from .anthropic_caller import AnthropicCaller
from .openai_caller import OpenAICaller
from .google_caller import GoogleCaller
from .factory import get_caller, load_model_config, list_models, list_tiers

__all__ = [
    "LLMResponse",
    "LLMCallerBase",
    "extract_sql",
    "AnthropicCaller",
    "OpenAICaller",
    "GoogleCaller",
    "get_caller",
    "load_model_config",
    "list_models",
    "list_tiers",
]
