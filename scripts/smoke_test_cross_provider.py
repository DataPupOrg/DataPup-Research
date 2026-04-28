#!/usr/bin/env python3
"""
scripts/smoke_test_cross_provider.py — Verify all 9 cross-provider models work.

Calls each model in config/cross_provider_models.yaml with a trivial
text-to-SQL prompt, verifies that:
  (a) the call succeeds
  (b) the response contains extractable SQL
  (c) input/output token counts and latency are reported

Independent calls per model so partial failures don't block the whole test.
Outputs a pass/fail table to stdout and JSON details to results/smoke/.

Usage:
    python scripts/smoke_test_cross_provider.py
    python scripts/smoke_test_cross_provider.py --tier flagship
    python scripts/smoke_test_cross_provider.py --models anthropic-opus-4-7,openai-gpt-5-2

Exits 0 if all attempted models pass, 1 otherwise. Skipped models (missing
API key for the provider) do not contribute to failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Make the framework package importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from framework.llm import (  # noqa: E402
    LLMResponse,
    get_caller,
    list_models,
    list_tiers,
    load_model_config,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_test")


SMOKE_PROMPT = (
    "You are evaluating connectivity. Return one ClickHouse query that selects "
    "the integer literal 1 with the column alias `n`. Return ONLY the SQL inside "
    "a ```sql fenced block — no explanation."
)
SMOKE_SYSTEM = "You are a helpful ClickHouse SQL assistant."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke-test the 9-model cross-provider matrix."
    )
    p.add_argument(
        "--tier",
        choices=["flagship", "mid", "small", "all"],
        default="all",
        help="Restrict to one tier (default: all 9 models).",
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated explicit list of model keys (overrides --tier).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "smoke",
        help="Where to write JSON detail records.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return p.parse_args()


def select_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        keys = [k.strip() for k in args.models.split(",") if k.strip()]
        unknown = [k for k in keys if k not in list_models()]
        if unknown:
            raise SystemExit(f"Unknown model keys: {unknown}. Known: {list_models()}")
        return keys
    if args.tier == "all":
        return list_models()
    tiers = list_tiers()
    if args.tier not in tiers:
        raise SystemExit(f"Unknown tier '{args.tier}'. Known: {sorted(tiers.keys())}")
    return tiers[args.tier]


# ---------------------------------------------------------------------------
# Provider key presence check (per-provider, so missing keys skip cleanly)
# ---------------------------------------------------------------------------

def provider_runnable(model_key: str) -> tuple[bool, str]:
    """For CLI transport, try to construct the caller (probes the binary).
    For SDK transport, check the env var. Return (ok, reason)."""
    cfg = load_model_config()
    entry = cfg["models"][model_key]
    transport = entry.get("transport") or cfg.get("defaults", {}).get("transport", "cli")
    provider = entry["provider"]
    if transport == "sdk":
        if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            return True, "ok"
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            return True, "ok"
        if provider == "google" and (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            return True, "ok"
        return False, f"missing {provider} API key"
    # CLI transport
    try:
        get_caller(model_key)
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"CLI unavailable: {str(e)[:200]}"


def provider_key_present_or_cli_ok(provider: str, model_key: str) -> bool:
    """Backwards-compat shim used by smoke_one()."""
    ok, _ = provider_runnable(model_key)
    return ok


# ---------------------------------------------------------------------------
# Per-model smoke run
# ---------------------------------------------------------------------------

def smoke_one(model_key: str, model_entry: dict) -> dict:
    """Smoke-test a single model. Returns a result dict."""
    provider = model_entry["provider"]
    model_id = model_entry["model_id"]
    display = model_entry["display_name"]

    record: dict = {
        "model_key": model_key,
        "provider": provider,
        "model_id": model_id,
        "display_name": display,
        "tier": model_entry.get("tier"),
        "skipped": False,
        "passed": False,
        "checks": {},
        "error": "",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not provider_key_present_or_cli_ok(provider, model_key):
        record["skipped"] = True
        record["error"] = f"Provider {provider} not runnable on this host"
        return record

    try:
        caller = get_caller(model_key)
        start = time.perf_counter()
        response: LLMResponse = caller.call(SMOKE_PROMPT, system=SMOKE_SYSTEM)
        wall_ms = (time.perf_counter() - start) * 1000

        record["wall_ms"] = round(wall_ms, 1)
        record["latency_ms"] = response.latency_ms
        record["input_tokens"] = response.input_tokens
        record["output_tokens"] = response.output_tokens
        record["sql"] = response.sql
        record["raw_response"] = response.raw_response[:500]
        record["api_success"] = response.success
        record["api_error"] = response.error

        # Acceptance checks
        sql_upper = (response.sql or "").upper().strip()
        checks = {
            "api_call_succeeded": response.success,
            "sql_non_empty": bool(response.sql and response.sql.strip()),
            "sql_contains_select": "SELECT" in sql_upper,
            "sql_contains_one": "1" in sql_upper,
            "input_tokens_reported": response.input_tokens > 0,
            "output_tokens_reported": response.output_tokens > 0,
            "latency_reported": response.latency_ms > 0,
        }
        record["checks"] = checks
        record["passed"] = all(checks.values())
        if not record["passed"] and not record["error"]:
            failed = [k for k, v in checks.items() if not v]
            record["error"] = f"Failed checks: {failed}"

    except Exception as e:  # noqa: BLE001 — we want to capture every failure
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()

    return record


# ---------------------------------------------------------------------------
# Pretty table output
# ---------------------------------------------------------------------------

def print_table(records: list[dict]) -> None:
    headers = ["model_key", "provider", "tier", "status", "in_tok", "out_tok", "lat_ms", "notes"]
    widths = {h: len(h) for h in headers}

    rows: list[dict] = []
    for r in records:
        if r["skipped"]:
            status = "SKIP"
            notes = r["error"]
        elif r["passed"]:
            status = "PASS"
            notes = "all checks ok"
        else:
            status = "FAIL"
            notes = r["error"][:60] if r.get("error") else "checks failed"
        row = {
            "model_key": r["model_key"],
            "provider": r["provider"],
            "tier": str(r.get("tier", "")),
            "status": status,
            "in_tok": str(r.get("input_tokens", "-")),
            "out_tok": str(r.get("output_tokens", "-")),
            "lat_ms": str(r.get("latency_ms", "-")),
            "notes": notes,
        }
        rows.append(row)
        for k in headers:
            widths[k] = max(widths[k], len(str(row[k])))

    sep = "  "
    fmt = sep.join(f"{{:{widths[h]}}}" for h in headers)
    print()
    print(fmt.format(*headers))
    print(fmt.format(*("-" * widths[h] for h in headers)))
    for row in rows:
        print(fmt.format(*(row[h] for h in headers)))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    cfg = load_model_config()
    models_cfg = cfg["models"]
    targets = select_models(args)

    print(f"Smoke-testing {len(targets)} model(s):")
    for k in targets:
        print(f"  - {k}")
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"smoke_{timestamp}.jsonl"

    records: list[dict] = []
    with out_path.open("w", encoding="utf-8") as f:
        for model_key in targets:
            entry = models_cfg[model_key]
            print(f"Testing {model_key} ({entry['provider']}/{entry['model_id']}) ... ", end="", flush=True)
            r = smoke_one(model_key, entry)
            records.append(r)
            if r["skipped"]:
                print("SKIP")
            elif r["passed"]:
                print("PASS")
            else:
                print("FAIL")
                print(f"   error: {r['error']}")
            f.write(json.dumps(r) + "\n")

    print_table(records)

    # Summary + exit code
    attempted = [r for r in records if not r["skipped"]]
    passed = [r for r in attempted if r["passed"]]
    failed = [r for r in attempted if not r["passed"]]
    skipped = [r for r in records if r["skipped"]]

    print(f"Summary: {len(passed)}/{len(attempted)} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped (missing keys).")
    print(f"Detail: {out_path}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
