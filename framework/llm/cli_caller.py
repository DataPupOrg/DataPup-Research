"""
framework/llm/cli_caller.py — CLI-based provider adapter.

Shells out to the public Claude Code, Codex (OpenAI), and Gemini CLIs
for non-interactive single-shot inference. Personal-account only — refuses
to run if the binary on PATH is the Meta launcher variant.

Supported CLIs (binary name -> auth model):
    claude   — Anthropic Claude Code           (subscription / API key handled by CLI)
    codex    — OpenAI Codex CLI                (ChatGPT account handled by CLI)
    gemini   — Google Gemini CLI               (Google account handled by CLI)

Each binary is invoked as a subprocess with model + prompt arguments.
Auth is handled entirely inside the CLI itself (`claude login`, `codex
login`, `gemini auth login`), so this adapter never sees an API key.

Token usage is best-effort: we parse the CLI's structured output where
available (`codex exec --json`, `claude --output-format json`) and
fall back to character-based estimation when the CLI emits plain text.

Per memory feedback_no_meta_cli_for_datapup.md:
    Refuses to invoke any binary whose --version output contains
    "at Meta" / "Meta Launcher" / similar strings.
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
# Meta-launcher detection
# ---------------------------------------------------------------------------

_META_MARKERS = (
    "at meta",
    "meta launcher",
    "meta agent",
    "fburl.com",
    "manifold native",
)


def _is_meta_launcher(binary: str) -> tuple[bool, str]:
    """Probe `binary --version` and decide whether to refuse it.

    Returns (refuse, version_text). We refuse if:
      - the version output mentions a Meta marker, OR
      - the version probe failed (timeout/missing) — better safe than sorry,
        since Meta launchers can hang on `--version` while waiting for
        upstream resolution
    """
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "NO_COLOR": "1"},
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        # Refuse on probe failure — could be a hanging Meta launcher
        return True, f"<probe failed: {type(e).__name__}>"
    blob = ((result.stdout or "") + "\n" + (result.stderr or "")).lower()
    for marker in _META_MARKERS:
        if marker in blob:
            return True, blob.strip()[:500]
    return False, blob.strip()[:500]


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
        ALLOWED_PATHS: tuple of acceptable absolute prefixes (so we don't
                       silently invoke the Meta-installed binary even
                       if it's first on PATH)
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
        """Locate the binary, prefer ALLOWED_PATHS, refuse Meta launcher."""
        if not self.BINARY:
            raise NotImplementedError(f"{type(self).__name__} must set BINARY")

        # 1. Try each ALLOWED_PATHS prefix first (deterministic, public-CLI-only)
        candidates: list[str] = []
        for prefix in self.ALLOWED_PATHS:
            cand = os.path.join(prefix, self.BINARY)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                candidates.append(cand)

        # 2. Fall back to PATH lookup if no ALLOWED_PATHS match
        if not candidates:
            which = shutil.which(self.BINARY)
            if which:
                candidates.append(which)

        if not candidates:
            raise EnvironmentError(
                f"CLI binary '{self.BINARY}' not found. Install the public {self.PROVIDER} "
                f"CLI on PATH. Allowed prefixes (preferred): {self.ALLOWED_PATHS}"
            )

        # 3. Reject Meta-launcher variants
        for cand in candidates:
            is_meta, version_text = _is_meta_launcher(cand)
            if is_meta:
                logger.warning(
                    "Rejecting Meta-launcher binary at %s (--version: %s)",
                    cand, version_text[:120],
                )
                continue
            logger.info("Using %s CLI at %s", self.PROVIDER, cand)
            return cand

        raise EnvironmentError(
            f"All discovered '{self.BINARY}' binaries appear to be Meta launchers. "
            f"Install the public {self.PROVIDER} CLI at one of: {self.ALLOWED_PATHS}. "
            f"Per memory feedback_no_meta_cli_for_datapup.md, the Meta-launcher variants "
            f"must not be used for the DataPup paper."
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
    """Wrapper around the public `claude` CLI (Claude Code).

    Invocation:
        claude --print --output-format json --model <id> --append-system-prompt <sys> "<prompt>"

    The `--print` flag tells Claude Code to run non-interactively. The
    `--output-format json` flag emits a single JSON object on stdout containing
    the result and token counts.

    Authentication is handled by `claude login` — never an API key passed
    by this code.
    """
    PROVIDER = "anthropic"
    BINARY = "claude"
    # Public CLI usually installed via `npm install -g @anthropic-ai/claude-code`
    # which lands under one of these prefixes. Meta launcher lives at /usr/local/bin/.
    ALLOWED_PATHS = (
        "/opt/homebrew/bin",
        os.path.expanduser("~/.npm-global/bin"),
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/.bun/bin"),
        "/usr/local/share/npm-global/bin",
    )

    def _build_invocation(self, prompt: str, system: Optional[str]) -> CLIInvocation:
        cmd: list[str] = [
            self._resolved_path,
            "--print",
            "--output-format", "json",
            "--model", self.model,
            "--max-turns", "1",
            "--allowed-tools", "",       # disable all tools — we want pure text response
            "--permission-mode", "default",
        ]
        if system:
            cmd += ["--append-system-prompt", system]
        # Prompt as positional arg so we don't have to escape stdin.
        cmd.append(prompt)
        return CLIInvocation(cmd=cmd, stdin=None, timeout=600)

    def _parse_output(
        self, stdout: str, stderr: str, returncode: int,
    ) -> tuple[str, str, int, int]:
        # Claude Code --output-format json emits a single object like:
        #   {"type":"result","subtype":"success","result":"...text...","usage":{"input_tokens":N,"output_tokens":M},...}
        try:
            payload = json.loads(stdout.strip())
        except json.JSONDecodeError:
            # Fallback: assume stdout is the raw response text
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
    """Wrapper around the public `codex` CLI (OpenAI Codex).

    Invocation:
        codex exec --skip-git-repo-check --sandbox read-only --model <id> \\
            --output-last-message <tmpfile> "<prompt>"

    `codex exec` runs non-interactively. We force read-only sandbox so the
    model cannot run shell commands and `--output-last-message` writes the
    final assistant turn to a tempfile (clean text, no event chatter).

    Token usage is parsed from `--json` event stream (fallback to estimate).

    Authentication is handled by `codex login` — never an API key.
    """
    PROVIDER = "openai"
    BINARY = "codex"
    ALLOWED_PATHS = (
        "/opt/homebrew/bin",
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/.cargo/bin"),
        "/usr/local/bin",  # codex itself is not a Meta launcher; binary check guards it
    )

    def _build_invocation(self, prompt: str, system: Optional[str]) -> CLIInvocation:
        # Combine system + user content — codex exec doesn't have a separate
        # system flag; concat with a clear delimiter.
        if system:
            full_prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}"
        else:
            full_prompt = prompt

        # Use a per-call tempfile to capture clean final message
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
        # 1. Read clean final message from the tempfile
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

        # 2. Parse JSONL event stream from stdout for token usage
        in_tok = out_tok = 0
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Codex emits events with various shapes; look for token_count fields
            usage = evt.get("usage") or evt.get("token_usage") or {}
            if usage:
                in_tok = max(in_tok, int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0))
                out_tok = max(out_tok, int(usage.get("output_tokens") or usage.get("completion_tokens") or 0))
            # Some versions emit token_count at top level
            if "input_tokens" in evt:
                in_tok = max(in_tok, int(evt["input_tokens"]))
            if "output_tokens" in evt:
                out_tok = max(out_tok, int(evt["output_tokens"]))

        # Fallback to estimate if parsing failed
        if in_tok == 0:
            in_tok = _estimate_tokens(raw_text)  # crude lower bound
        if out_tok == 0:
            out_tok = _estimate_tokens(raw_text)

        return extract_sql(raw_text), raw_text, in_tok, out_tok


# ---------------------------------------------------------------------------
# Google Gemini CLI
# ---------------------------------------------------------------------------

class GeminiCLICaller(CLICallerBase):
    """Wrapper around the public `gemini` CLI (Google).

    Invocation:
        gemini --prompt "<text>" --model <id> --output-format json --yolo

    The `--prompt` flag is the non-interactive entry point. `--output-format
    json` emits a structured response with token stats. `--yolo` skips
    confirmation prompts for built-in tools (we don't use tools, but the
    flag is harmless and prevents stdin blocking).

    Authentication is handled by `gemini auth login` — never an API key.
    """
    PROVIDER = "google"
    BINARY = "gemini"
    ALLOWED_PATHS = (
        "/opt/homebrew/bin",
        os.path.expanduser("~/.npm-global/bin"),
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/.bun/bin"),
        "/usr/local/share/npm-global/bin",
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
        # Gemini --output-format json emits a single object like:
        #   {"response": "...", "stats": {"models": {"<model>": {"tokens": {"prompt": N, "candidates": M}}}}}
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
