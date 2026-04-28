"""
framework/llm/cli_caller.py — CLI-based provider adapter.

Shells out to the Claude Code, Codex (OpenAI), and Gemini CLIs for
non-interactive single-shot inference.

Supported CLIs (binary name -> auth model):
    claude   — Anthropic Claude Code           (auth handled by CLI)
    codex    — OpenAI Codex CLI                (auth handled by CLI)
    gemini   — Google Gemini CLI               (auth handled by CLI)

Each binary is invoked as a subprocess with model + prompt arguments.
Auth is handled entirely inside the CLI itself (`claude login`, `codex
login`, `gemini auth login`), so this adapter never sees an API key.

Token usage is best-effort: we parse the CLI's structured output where
available (`codex exec --json`, `claude --output-format json`) and fall
back to character-based estimation when the CLI emits plain text.

Binary resolution: prefers entries in ALLOWED_PATHS (a hint, not a gate),
falls back to `shutil.which`. Whatever binary is selected is invoked
without further validation — the caller is responsible for ensuring the
right binary is on PATH.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from .base import LLMCallerBase, LLMResponse, extract_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base CLI caller
# ---------------------------------------------------------------------------

@dataclass
class CLIInvocation:
    """Internal: how to invoke a CLI for one request."""
    cmd: list[str]
    stdin: Optional[str] = None  # If provided, prompt goes here; else inline in cmd
    env: Optional[dict] = None   # Extra env overlay
    timeout: int = 600


class CLICallerBase(LLMCallerBase):
    """Abstract base for CLI-driven providers.

    Concrete subclasses set:
        BINARY:        binary name expected on PATH
        ALLOWED_PATHS: tuple of preferred absolute prefixes (resolution
                       order hint; PATH fallback if none match)
        PROVIDER:      "anthropic" | "openai" | "google"

    and implement `_build_invocation(prompt, system) -> CLIInvocation` and
    `_parse_output(stdout, stderr, returncode) -> (sql, raw_text, in_tok, out_tok)`.
    """

    BINARY: str = ""
    ALLOWED_PATHS: tuple[str, ...] = ()

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self._resolved_path = self._resolve_binary()

    # ----- binary resolution -------------------------------------------------

    def _resolve_binary(self) -> str:
        """Locate the binary: prefer ALLOWED_PATHS, fall back to PATH."""
        if not self.BINARY:
            raise NotImplementedError(f"{type(self).__name__} must set BINARY")

        # 1. Try each ALLOWED_PATHS prefix first (deterministic resolution)
        for prefix in self.ALLOWED_PATHS:
            cand = os.path.join(prefix, self.BINARY)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                logger.info("Using %s CLI at %s", self.PROVIDER, cand)
                return cand

        # 2. Fall back to PATH lookup
        which = shutil.which(self.BINARY)
        if which:
            logger.info("Using %s CLI at %s (via PATH)", self.PROVIDER, which)
            return which

        raise EnvironmentError(
            f"CLI binary '{self.BINARY}' not found on PATH. "
            f"Install the {self.PROVIDER} CLI."
        )

    # ----- abstract methods --------------------------------------------------

    def _build_invocation(self, prompt: str, system: Optional[str]) -> CLIInvocation:
        raise NotImplementedError

    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> tuple[str, str, int, int]:
        """Return (sql, raw_response, input_tokens, output_tokens)."""
        raise NotImplementedError

    # ----- main call --------------------------------------------------------

    def call(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            invocation = self._build_invocation(prompt, system)
            start_time = time.perf_counter()
            try:
                proc = subprocess.run(
                    invocation.cmd,
                    input=invocation.stdin,
                    capture_output=True,
                    text=True,
                    timeout=invocation.timeout,
                    env={**os.environ, "NO_COLOR": "1", **(invocation.env or {})},
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if proc.returncode != 0:
                    err_text = (proc.stderr or proc.stdout or "").strip()[:1000]
                    last_error = f"CLI exit {proc.returncode}: {err_text}"
                    if _is_transient(err_text) and attempt < self.max_retries:
                        logger.warning("Transient CLI error (attempt %d/%d): %s",
                                       attempt, self.max_retries, err_text[:200])
                        time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))
                        continue
                    return _failed(self.model, self.PROVIDER, last_error, start_time)

                sql, raw_text, in_tok, out_tok = self._parse_output(
                    proc.stdout, proc.stderr, proc.returncode
                )
                return LLMResponse(
                    sql=sql,
                    raw_response=raw_text,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=round(elapsed_ms, 2),
                    model=self.model,
                    success=True,
                    provider=self.PROVIDER,
                )

            except subprocess.TimeoutExpired as e:
                last_error = f"CLI timeout after {invocation.timeout}s: {e}"
                logger.warning(last_error)
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))

            except Exception as e:  # noqa: BLE001
                return _failed(
                    self.model, self.PROVIDER,
                    f"Unexpected error: {type(e).__name__}: {e}",
                    start_time,
                )

        return _failed(
            self.model, self.PROVIDER,
            f"All {self.max_retries} retry attempts exhausted. {last_error}",
            time.perf_counter(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _failed(model: str, provider: str, error: str, start_time: float) -> LLMResponse:
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.error(error)
    return LLMResponse(
        sql="", raw_response="", input_tokens=0, output_tokens=0,
        latency_ms=round(max(elapsed_ms, 0.0), 2), model=model,
        success=False, error=error, provider=provider,
    )


def _is_transient(err_text: str) -> bool:
    blob = err_text.lower()
    triggers = (
        "rate limit", "rate-limit", "ratelimit", "too many requests",
        "overloaded", "503", "502", "504", "529",
        "timeout", "timed out", "temporarily unavailable",
        "try again", "transient",
    )
    return any(t in blob for t in triggers)


def _estimate_tokens(text: str) -> int:
    """Char-based token estimate (~3.5 chars/token)."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.5))


# ---------------------------------------------------------------------------
# Anthropic Claude Code CLI
# ---------------------------------------------------------------------------

class ClaudeCLICaller(CLICallerBase):
    """Wrapper around the `claude` CLI (Claude Code).

    Invocation:
        claude --print --output-format json --model <id> --append-system-prompt <sys> "<prompt>"

    The `--print` flag tells Claude Code to run non-interactively. The
    `--output-format json` flag emits a single JSON object on stdout
    containing the result and token counts.

    Authentication is handled by `claude login`.
    """
    PROVIDER = "anthropic"
    BINARY = "claude"
    ALLOWED_PATHS = (
        "/opt/homebrew/bin",
        os.path.expanduser("~/.npm-global/bin"),
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/.bun/bin"),
        "/usr/local/share/npm-global/bin",
        "/usr/local/bin",
    )

    def _build_invocation(self, prompt: str, system: Optional[str]) -> CLIInvocation:
        cmd: list[str] = [
            self._resolved_path,
            "--print",
            "--output-format", "json",
            "--model", self.model,
            "--max-turns", "1",
            "--allowed-tools", "",       # disable all tools — pure text response
            "--permission-mode", "default",
        ]
        if system:
            cmd += ["--append-system-prompt", system]
        cmd.append(prompt)
        return CLIInvocation(cmd=cmd, stdin=None, timeout=600)

    def _parse_output(
        self, stdout: str, stderr: str, returncode: int,
    ) -> tuple[str, str, int, int]:
        try:
            payload = json.loads(stdout.strip())
        except json.JSONDecodeError:
            raw_text = stdout.strip()
            return extract_sql(raw_text), raw_text, _estimate_tokens(""), _estimate_tokens(raw_text)

        raw_text = payload.get("result") or payload.get("text") or ""
        usage = payload.get("usage") or {}
        in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        return extract_sql(raw_text), raw_text, in_tok, out_tok


# ---------------------------------------------------------------------------
# OpenAI Codex CLI
# ---------------------------------------------------------------------------

class CodexCLICaller(CLICallerBase):
    """Wrapper around the `codex` CLI (OpenAI Codex).

    Invocation:
        codex exec --skip-git-repo-check --sandbox read-only --model <id> \\
            --output-last-message <tmpfile> --json "<prompt>"

    Authentication is handled by `codex login`.
    """
    PROVIDER = "openai"
    BINARY = "codex"
    ALLOWED_PATHS = (
        "/opt/homebrew/bin",
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/.cargo/bin"),
        "/usr/local/bin",
    )

    def _build_invocation(self, prompt: str, system: Optional[str]) -> CLIInvocation:
        if system:
            full_prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}"
        else:
            full_prompt = prompt

        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
        )
        tmp.close()
        self._last_message_path = tmp.name

        cmd: list[str] = [
            self._resolved_path, "exec",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "--model", self.model,
            "--output-last-message", self._last_message_path,
            "--json",
            full_prompt,
        ]
        return CLIInvocation(cmd=cmd, stdin=None, timeout=900)

    def _parse_output(
        self, stdout: str, stderr: str, returncode: int,
    ) -> tuple[str, str, int, int]:
        raw_text = ""
        try:
            with open(self._last_message_path, "r", encoding="utf-8") as f:
                raw_text = f.read().strip()
        except Exception:  # noqa: BLE001
            raw_text = stdout.strip()
        finally:
            try:
                os.unlink(self._last_message_path)
            except OSError:
                pass

        in_tok = out_tok = 0
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            usage = evt.get("usage") or evt.get("token_usage") or {}
            if usage:
                in_tok = max(in_tok, int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0))
                out_tok = max(out_tok, int(usage.get("output_tokens") or usage.get("completion_tokens") or 0))
            if "input_tokens" in evt:
                in_tok = max(in_tok, int(evt["input_tokens"]))
            if "output_tokens" in evt:
                out_tok = max(out_tok, int(evt["output_tokens"]))

        if in_tok == 0:
            in_tok = _estimate_tokens(raw_text)
        if out_tok == 0:
            out_tok = _estimate_tokens(raw_text)

        return extract_sql(raw_text), raw_text, in_tok, out_tok


# ---------------------------------------------------------------------------
# Google Gemini CLI
# ---------------------------------------------------------------------------

class GeminiCLICaller(CLICallerBase):
    """Wrapper around the `gemini` CLI (Google).

    Invocation:
        gemini --prompt "<text>" --model <id> --output-format json --yolo

    Authentication is handled by `gemini auth login`.
    """
    PROVIDER = "google"
    BINARY = "gemini"
    ALLOWED_PATHS = (
        "/opt/homebrew/bin",
        os.path.expanduser("~/.npm-global/bin"),
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/.bun/bin"),
        "/usr/local/share/npm-global/bin",
        "/usr/local/bin",
    )

    def _build_invocation(self, prompt: str, system: Optional[str]) -> CLIInvocation:
        if system:
            full_prompt = f"[System]\n{system}\n\n[User]\n{prompt}"
        else:
            full_prompt = prompt

        cmd: list[str] = [
            self._resolved_path,
            "--prompt", full_prompt,
            "--model", self.model,
            "--output-format", "json",
            "--yolo",
        ]
        return CLIInvocation(cmd=cmd, stdin=None, timeout=600)

    def _parse_output(
        self, stdout: str, stderr: str, returncode: int,
    ) -> tuple[str, str, int, int]:
        try:
            payload = json.loads(stdout.strip())
        except json.JSONDecodeError:
            raw_text = stdout.strip()
            return extract_sql(raw_text), raw_text, 0, _estimate_tokens(raw_text)

        raw_text = payload.get("response") or payload.get("text") or ""

        in_tok = out_tok = 0
        stats = payload.get("stats") or {}
        models = stats.get("models") or {}
        for model_stats in models.values():
            tokens = (model_stats or {}).get("tokens") or {}
            in_tok = max(in_tok, int(tokens.get("prompt") or tokens.get("input") or 0))
            out_tok = max(out_tok, int(tokens.get("candidates") or tokens.get("output") or 0))

        if in_tok == 0:
            in_tok = _estimate_tokens(raw_text)
        if out_tok == 0:
            out_tok = _estimate_tokens(raw_text)

        return extract_sql(raw_text), raw_text, in_tok, out_tok

