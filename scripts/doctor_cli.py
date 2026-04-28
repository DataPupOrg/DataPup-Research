#!/usr/bin/env python3
"""
scripts/doctor_cli.py — Diagnose Claude / Codex / Gemini CLI installation.

Probes each CLI binary on the host: which path resolves, what `--version`
reports, and (with --probe-call) whether a trivial round-trip succeeds.

Usage:
    python scripts/doctor_cli.py
    python scripts/doctor_cli.py --provider anthropic
    python scripts/doctor_cli.py --probe-call    # actually invoke each CLI

Exits 0 if every probed CLI is reachable; 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from framework.llm import load_model_config  # noqa: E402
from framework.llm.cli_caller import (  # noqa: E402
    ClaudeCLICaller,
    CodexCLICaller,
    GeminiCLICaller,
)


CLI_SPECS = {
    "anthropic": {
        "binary": "claude",
        "allowed_paths": ClaudeCLICaller.ALLOWED_PATHS,
        "install_hint": (
            "  npm install -g @anthropic-ai/claude-code\n"
            "  # or: curl -fsSL https://claude.ai/install.sh | bash"
        ),
        "auth_hint": "  claude login",
    },
    "openai": {
        "binary": "codex",
        "allowed_paths": CodexCLICaller.ALLOWED_PATHS,
        "install_hint": (
            "  brew install --cask codex\n"
            "  # or: npm install -g @openai/codex"
        ),
        "auth_hint": "  codex login",
    },
    "google": {
        "binary": "gemini",
        "allowed_paths": GeminiCLICaller.ALLOWED_PATHS,
        "install_hint": (
            "  npm install -g @google/gemini-cli\n"
            "  # or: brew install gemini-cli"
        ),
        "auth_hint": "  gemini auth login",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose CLI setup for cross-provider eval.")
    p.add_argument(
        "--provider",
        choices=["anthropic", "openai", "google", "all"],
        default="all",
    )
    p.add_argument(
        "--probe-call",
        action="store_true",
        help="Actually invoke each CLI with a trivial prompt (slower, real auth check).",
    )
    return p.parse_args()


def find_binary(binary: str, allowed_paths: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Return (preferred_paths, all_paths) for the binary."""
    preferred: list[str] = []
    for prefix in allowed_paths:
        cand = os.path.join(prefix, binary)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            preferred.append(cand)
    all_found: list[str] = []
    for path_entry in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(path_entry, binary)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            real = os.path.realpath(cand)
            if real not in all_found:
                all_found.append(real)
    return preferred, all_found


def probe_version(binary_path: str) -> str:
    """Run `--version` and return whatever it prints (trimmed)."""
    try:
        result = subprocess.run(
            [binary_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "NO_COLOR": "1"},
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"<probe failed: {type(e).__name__}>"
    return ((result.stdout or "") + (result.stderr or "")).strip()[:300]


def probe_provider(provider: str, do_probe_call: bool) -> dict:
    spec = CLI_SPECS[provider]
    binary = spec["binary"]
    allowed = spec["allowed_paths"]

    preferred, all_found = find_binary(binary, allowed)

    result: dict = {
        "provider": provider,
        "binary": binary,
        "preferred_paths": preferred,
        "all_paths": all_found,
        "selected": None,
        "version_text": "",
        "passed": False,
        "notes": [],
    }

    if not all_found:
        result["notes"].append(f"NOT INSTALLED — install with:\n{spec['install_hint']}")
        return result

    # Pick the first preferred path; fall back to first PATH entry
    if preferred:
        selected = preferred[0]
    else:
        selected = all_found[0]
        result["notes"].append(
            f"Found outside ALLOWED_PATHS hint. Allowed: {list(allowed)}"
        )
    result["selected"] = selected
    result["version_text"] = probe_version(selected)

    if do_probe_call:
        ok, msg = probe_call(provider, selected)
        result["probe_call_ok"] = ok
        result["probe_call_msg"] = msg
        if not ok:
            result["notes"].append(f"Auth probe failed: {msg}\nLog in with:\n{spec['auth_hint']}")
            return result

    result["passed"] = True
    return result


def probe_call(provider: str, binary_path: str) -> tuple[bool, str]:
    """Invoke the CLI with a trivial prompt to verify it's authenticated."""
    prompt = "Reply with the single word OK and nothing else."
    cmd: list[str]
    if provider == "anthropic":
        cmd = [binary_path, "--print", "--output-format", "json",
               "--allowed-tools", "", "--max-turns", "1", prompt]
    elif provider == "openai":
        cmd = [binary_path, "exec", "--skip-git-repo-check",
               "--sandbox", "read-only", prompt]
    elif provider == "google":
        cmd = [binary_path, "--prompt", prompt, "--output-format", "json", "--yolo"]
    else:
        return False, f"unknown provider {provider}"

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env={**os.environ, "NO_COLOR": "1"},
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, f"{type(e).__name__}: {e}"

    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "")[:300].strip()
    output = (proc.stdout or "").strip()
    if "OK" not in output.upper():
        return False, f"Unexpected response: {output[:200]}"
    return True, "OK"


def print_report(results: list[dict]) -> None:
    print()
    for r in results:
        provider = r["provider"]
        binary = r["binary"]
        status = "PASS" if r["passed"] else "FAIL"
        marker = "OK" if r["passed"] else "FAIL"
        print(f"  [{marker}] {provider} ({binary})")
        if r.get("selected"):
            print(f"      selected: {r['selected']}")
        if r.get("version_text"):
            short_version = r["version_text"].split("\n")[0][:120]
            print(f"      --version: {short_version}")
        if r.get("all_paths") and len(r["all_paths"]) > 1:
            others = [p for p in r["all_paths"] if p != r["selected"]]
            if others:
                print(f"      other PATH entries: {others}")
        if r.get("probe_call_ok") is True:
            print("      auth probe: OK")
        elif r.get("probe_call_ok") is False:
            print(f"      auth probe: FAILED — {r.get('probe_call_msg', '')[:200]}")
        for note in r.get("notes", []):
            for ln in note.splitlines():
                print(f"        {ln}")
    print()


def main() -> int:
    args = parse_args()
    cfg = load_model_config()
    print(f"DataPup-Research CLI doctor")
    print(f"Default transport: {cfg.get('defaults', {}).get('transport', 'cli')}")

    providers = (
        list(CLI_SPECS.keys()) if args.provider == "all" else [args.provider]
    )

    results = [probe_provider(p, args.probe_call) for p in providers]
    print_report(results)

    failed = [r for r in results if not r["passed"]]
    print(f"Summary: {len(results) - len(failed)}/{len(results)} CLIs passed.")
    if failed:
        print("Failures:")
        for r in failed:
            for note in r.get("notes", []):
                first_line = note.splitlines()[0]
                print(f"  - [{r['provider']}] {first_line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
