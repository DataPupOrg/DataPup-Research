"""
framework/llm/factory.py — Provider dispatch and model-config loading.

Loads `config/cross_provider_models.yaml` once and exposes:
    get_caller(model_key, **overrides)   -> LLMCallerBase concrete instance
    list_models()                        -> sorted list of model keys
    list_tiers()                         -> {tier: [model_key, ...]}
    load_model_config()                  -> raw config dict

Each model entry has a `transport` field:
    cli   — invoke via the public CLI binary (default for the paper)
    sdk   — invoke via the provider's Python SDK with personal API key

CLI callers are imported eagerly. SDK callers are imported lazily — only
loaded when transport=sdk dispatch actually triggers them. This means a
CLI-only user does not need to install anthropic/openai/google-genai.

`model_key` is the top-level key in the yaml `models:` block (e.g.,
"anthropic-opus-4-7"), NOT the provider's canonical model id.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from .base import LLMCallerBase
from .cli_caller import ClaudeCLICaller, CodexCLICaller, GeminiCLICaller

logger = logging.getLogger(__name__)


_THIS_FILE = Path(__file__).resolve()
_CONFIG_PATH = _THIS_FILE.parent.parent.parent / "config" / "cross_provider_models.yaml"


@functools.lru_cache(maxsize=1)
def load_model_config(path: Optional[Path] = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else _CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"cross_provider_models.yaml not found at {cfg_path}. "
            f"Did you clone the DataPup-Research repo?"
        )
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_models() -> list[str]:
    return sorted(load_model_config().get("models", {}).keys())


def list_tiers() -> dict[str, list[str]]:
    return load_model_config().get("tiers", {})


# CLI dispatch — always available, no extra deps.
_CLI_DISPATCH: dict[str, type[LLMCallerBase]] = {
    "anthropic": ClaudeCLICaller,
    "openai":    CodexCLICaller,
    "google":    GeminiCLICaller,
}


def _load_sdk_caller(provider: str) -> type[LLMCallerBase]:
    """Import the SDK caller for `provider` lazily, raising a friendly
    error if the underlying SDK is not installed."""
    try:
        if provider == "anthropic":
            from .anthropic_caller import AnthropicCaller
            return AnthropicCaller
        if provider == "openai":
            from .openai_caller import OpenAICaller
            return OpenAICaller
        if provider == "google":
            from .google_caller import GoogleCaller
            return GoogleCaller
    except ImportError as e:
        sdk_pkg = {
            "anthropic": "anthropic",
            "openai":    "openai",
            "google":    "google-genai",
        }.get(provider, "<unknown>")
        raise ImportError(
            f"transport=sdk for provider '{provider}' requires the {sdk_pkg} "
            f"Python package. Install with: pip install {sdk_pkg}"
        ) from e
    raise ValueError(f"Unknown provider for SDK dispatch: {provider}")


def get_caller(model_key: str, **overrides: Any) -> LLMCallerBase:
    cfg = load_model_config()
    models = cfg.get("models", {})
    if model_key not in models:
        raise KeyError(
            f"Unknown model_key '{model_key}'. Known keys: {sorted(models.keys())}"
        )
    entry = models[model_key]
    provider = entry["provider"]
    model_id = entry["model_id"]
    transport = entry.get("transport") or cfg.get("defaults", {}).get("transport", "cli")

    if transport == "cli":
        if provider not in _CLI_DISPATCH:
            raise ValueError(
                f"No CLI caller registered for provider={provider}. "
                f"Known: {sorted(_CLI_DISPATCH.keys())}"
            )
        cls = _CLI_DISPATCH[provider]
    elif transport == "sdk":
        cls = _load_sdk_caller(provider)
    else:
        raise ValueError(
            f"Unknown transport '{transport}' for model {model_key}. "
            f"Use 'cli' or 'sdk'."
        )

    defaults = cfg.get("defaults", {}) or {}
    kwargs = {
        "max_tokens": defaults.get("max_tokens", 2048),
        "temperature": defaults.get("temperature", 0.0),
        "max_retries": defaults.get("max_retries", 4),
        "retry_base_delay": defaults.get("retry_base_delay", 1.5),
    }
    kwargs.update(overrides)

    logger.info(
        "get_caller: model_key=%s provider=%s transport=%s model_id=%s class=%s",
        model_key, provider, transport, model_id, cls.__name__,
    )
    return cls(model=model_id, **kwargs)
