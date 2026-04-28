"""
framework/llm/__init__.py — Multi-provider LLM caller package.

Eagerly imports CLI callers (no SDK dependencies) and lazily loads SDK
callers only when actually needed. This means a CLI-only user can run
the framework with just `pyyaml` installed; SDK users install the
provider SDK only for the providers they actually use via transport=sdk.
"""

from __future__ import annotations

from .base import LLMCallerBase, LLMResponse, extract_sql
from .cli_caller import (
    ClaudeCLICaller,
    CodexCLICaller,
    GeminiCLICaller,
)
from .factory import get_caller, list_models, list_tiers, load_model_config

__all__ = [
    # Base
    "LLMResponse",
    "LLMCallerBase",
    "extract_sql",
    # CLI callers (always available)
    "ClaudeCLICaller",
    "CodexCLICaller",
    "GeminiCLICaller",
    # Factory
    "get_caller",
    "load_model_config",
    "list_models",
    "list_tiers",
]


# SDK callers are imported lazily — only loaded if transport=sdk routes
# require them. Access them via framework.llm.<name> on demand:
#
#     from framework.llm import AnthropicCaller   # triggers SDK import
#
# This keeps "claude + pyyaml" sufficient for the default CLI workflow.

def __getattr__(name):
    if name == "AnthropicCaller":
        from .anthropic_caller import AnthropicCaller
        return AnthropicCaller
    if name == "OpenAICaller":
        from .openai_caller import OpenAICaller
        return OpenAICaller
    if name == "GoogleCaller":
        from .google_caller import GoogleCaller
        return GoogleCaller
    raise AttributeError(f"module 'framework.llm' has no attribute {name!r}")
