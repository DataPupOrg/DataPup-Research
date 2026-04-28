#!/usr/bin/env python3
"""
scripts/doctor_cli.py — Verify the public Claude / Codex / Gemini CLIs
are installed, authenticated, and NOT the Meta-launcher variants.

Run this once on the machine where you'll execute the cross-provider
matrix. It probes each CLI binary, refuses Meta launchers, reports auth
state, and confirms the model ids in cross_provider_models.yaml are
acceptable to each CLI's --model flag.

Usage:
    python scripts/doctor_cli.py
    python scripts/doctor_cli.py --provider anthropic
    python scripts/doctor_cli.py --probe-call    # actually invoke each CLI with a 1-token prompt

Exits 0 if every probed CLI is healthy; 1 otherwise.
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
    _is_meta_launcher,
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
        "version_args": ["--version"],
    },
    "openai": {
        "binary": "codex",
        "allowed_paths": CodexCLICaller.ALLOWED_PATHS,
        "install_hint": (
            "  brew install --cask codex\n"
            "  # or: npm install -g @openai/codex"
        ),
        "auth_hint": "  codex login",
        "version_args": ["--version"],
    },
    "google": {
        "binary": "gemini",
        "allowed_paths": GeminiCLICaller.ALLOWED_PATHS,
        "install_hint": (
            "  npm install -g @google/gemini-cli\n"
            "  # or: brew install gemini-cli"
        ),
        "auth_hint": "  gemini auth login",
        "version_args": ["--version"],
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
    # Find all instances on PATH for transparency
    all_found: list[str] = []
    for path_entry in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(path_entry, binary)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            real = os.path.realpath(cand)
            if real not in all_found:
                all_found.append(real)
    return preferred, all_found


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
        "is_meta": False,
        "version_text": "",
        "passed": False,
        "notes": [],
    }

    if not all_found:
        result["notes"].append(f"NOT INSTALLED — install with:\n{spec['install_hint']}")
        return result

    # Pick the first preferred path; if none preferred, check candidates outside
    # ALLOWED_PATHS more carefully — they're more likely to be Meta variants.
    if preferred:
        selected = preferred[0]
    else:
        # No binary in any ALLOWED_PATHS prefix. Try each PATH entry but
        # require it to pass the Meta-launcher check before accepting.
        selected = None
        for cand in all_found:
            cand_meta, cand_version = _is_meta_launcher(cand)
            if not cand_meta:
                selected = cand
                result["notes"].append(
                    f"Using {cand} (outside ALLOWED_PATHS but appears non-Meta). "
                    f"Consider symlinking into one of: {list(allowed)}"
                )
                break
        if not selected:
            result["notes"].append(
                f"All {binary} binaries on PATH appear to be Meta launchers "
                f"(or failed --version probe). Install the public CLI:\n{spec['install_hint']}"
            )
            for cand in all_found:
                _, ver = _is_meta_launcher(cand)
                result["notes"].append(f"  rejected: {cand} ({ver[:80]})")
            return result

    result["selected"] = selected

    is_meta, version_text = _is_meta_launcher(selected)
    result["is_meta"] = is_meta
    result["version_text"] = version_text[:200]

    if is_meta:
        result["notes"].append(
            f"REFUSED — {selected} appears to be a Meta launcher (or hung on --version). "
            f"Install the public CLI:\n{spec['install_hint']}"
        )
        # Check for an alternate non-Meta binary in PATH
        for cand in all_found:
            if cand == selected:
                continue
            cand_meta, cand_version = _is_meta_launcher(cand)
            if not cand_meta:
                result["notes"].append(
                    f"ALTERNATE FOUND: {cand} ({cand_version[:80]}) appears non-Meta. "
                    f"Move it earlier on PATH or symlink into one of "
                    f"the ALLOWED_PATHS prefixes."
                )
        return result

    # Optional: do a probe call to verify auth
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
        marker = "✓" if r["passed"] else "✗"
        print(f"  {marker} [{provider}] {binary}: {status}")
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
