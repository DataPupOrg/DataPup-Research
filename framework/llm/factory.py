"""
framework/llm/factory.py — Provider dispatch and model-config loading.

Loads `config/cross_provider_models.yaml` once and exposes:
    get_caller(model_key, **overrides)   -> LLMCallerBase concrete instance
    list_models()                        -> sorted list of model keys
    list_tiers()                         -> {tier: [model_key, ...]}
    load_model_config()                  -> raw config dict

`model_key` is the top-level key in the yaml `models:` block (e.g.,
"anthropic-opus-4-7"), NOT the provider's canonical model id.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from .anthropic_caller import AnthropicCaller
from .base import LLMCallerBase
from .google_caller import GoogleCaller
from .openai_caller import OpenAICaller

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

# Locate config relative to this file so the package is portable.
_THIS_FILE = Path(__file__).resolve()
_CONFIG_PATH = _THIS_FILE.parent.parent.parent / "config" / "cross_provider_models.yaml"


@functools.lru_cache(maxsize=1)
def load_model_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Load and cache the cross-provider model config."""
    cfg_path = Path(path) if path else _CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"cross_provider_models.yaml not found at {cfg_path}. "
            f"Did you clone the DataPup-Research repo?"
        )
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_models() -> list[str]:
    """Return all configured model keys, sorted."""
    return sorted(load_model_config().get("models", {}).keys())


def list_tiers() -> dict[str, list[str]]:
    """Return the tier-to-model-keys mapping."""
    return load_model_config().get("tiers", {})


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

_PROVIDER_CLASSES: dict[str, type[LLMCallerBase]] = {
    "anthropic": AnthropicCaller,
    "openai": OpenAICaller,
    "google": GoogleCaller,
}


def get_caller(model_key: str, **overrides: Any) -> LLMCallerBase:
    """Return a concrete LLMCallerBase instance for `model_key`.

    `model_key` must be a top-level key in `cross_provider_models.yaml`
    (e.g., "anthropic-opus-4-7"). Per-call overrides may include
    `max_tokens`, `temperature`, `max_retries`, `retry_base_delay`.
    """
    cfg = load_model_config()
    models = cfg.get("models", {})
    if model_key not in models:
        raise KeyError(
            f"Unknown model_key '{model_key}'. Known keys: {sorted(models.keys())}"
        )
    entry = models[model_key]
    provider = entry["provider"]
    model_id = entry["model_id"]

    if provider not in _PROVIDER_CLASSES:
        raise ValueError(f"Unknown provider '{provider}' for model {model_key}")

    defaults = cfg.get("defaults", {}) or {}
    kwargs = {
        "max_tokens": defaults.get("max_tokens", 2048),
        "temperature": defaults.get("temperature", 0.0),
        "max_retries": defaults.get("max_retries", 4),
        "retry_base_delay": defaults.get("retry_base_delay", 1.5),
    }
    kwargs.update(overrides)

    cls = _PROVIDER_CLASSES[provider]
    logger.info("get_caller: model_key=%s provider=%s model_id=%s", model_key, provider, model_id)
    return cls(model=model_id, **kwargs)
