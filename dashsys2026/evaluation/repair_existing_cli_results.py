#!/usr/bin/env python3
"""Repair and rescore existing CLI text-to-SQL results.

This is a targeted alternative to rerunning the full generation matrix with
EXECUTE=1 and REPAIR_ON_ERROR=1. It reads generated results, executes the SQL,
calls the same model only for non-empty SQL that fails to execute, and compares
the final SQL against the gold result set.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.framework.result_comparator import MatchStrategy, compare_results  # noqa: E402
from evaluation.run_cli_experiments import clean_sql, extract_sql  # noqa: E402


MODEL_COMMANDS = {
    "claude_cli": '/usr/local/bin/claude --dangerously-disable-osx-sandbox --dangerously-skip-permissions -p',
    "codex_cli": '/usr/local/bin/codex --dangerously-disable-osx-sandbox exec --dangerously-bypass-approvals-and-sandbox -',
    "gemini_cli": '/usr/local/bin/gemini --dangerously-disable-osx-sandbox --approval-mode=yolo -p ""',
}


@dataclass
class SqlResult:
    success: bool
    rows: list[list[Any]]
    columns: list[str]
    latency_ms: float
    error: str = ""


@dataclass
class ModelResponse:
    success: bool
    raw_response: str
    sql: str
    latency_ms: float
    error: str = ""


def run_clickhouse_query(
    sql: str,
    clickhouse_client: str,
    host: str,
    port: int,
    user: str,
    password: str,
    timeout_sec: int,
) -> SqlResult:
    if not sql.strip():
        return SqlResult(False, [], [], 0.0, "empty SQL")

    cmd = [
        clickhouse_client,
        "--host",
        host,
        "--port",
        str(port),
        "--user",
        user,
        "--format",
        "JSONCompact",
        "--query",
        sql,
    ]
    if password:
        cmd.extend(["--password", password])

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = (time.perf_counter() - start) * 1000
        return SqlResult(False, [], [], round(elapsed, 2), f"timeout: {e}")

    elapsed = (time.perf_counter() - start) * 1000
    if proc.returncode != 0:
        return SqlResult(
            False,
            [],
            [],
            round(elapsed, 2),
            (proc.stderr or proc.stdout or "").strip()[:2000],
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return SqlResult(False, [], [], round(elapsed, 2), f"bad JSONCompact: {e}")

    return SqlResult(
        True,
        payload.get("data", []),
        [m.get("name", "") for m in payload.get("meta", [])],
        round(elapsed, 2),
    )


def run_model_command(
    model_cmd: str,
    prompt: str,
    cwd: Path,
    timeout_sec: int,
) -> ModelResponse:
    cwd.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".prompt.txt",
        delete=False,
    ) as f:
        f.write(prompt)
        prompt_file = Path(f.name)

    try:
        if "{prompt_file}" in model_cmd or "{prompt}" in model_cmd:
            cmd = model_cmd.replace("{prompt_file}", shlex.quote(str(prompt_file)))
            cmd = cmd.replace("{prompt}", shlex.quote(prompt))
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=timeout_sec,
            )
        else:
            proc = subprocess.run(
                model_cmd,
                shell=True,
                cwd=str(cwd),
                input=prompt,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
            )
    except subprocess.TimeoutExpired as e:
        elapsed = (time.perf_counter() - start) * 1000
        return ModelResponse(False, "", "", round(elapsed, 2), f"timeout: {e}")
    finally:
        try:
            prompt_file.unlink()
        except OSError:
            pass

    elapsed = (time.perf_counter() - start) * 1000
    stdout = proc.stdout or ""
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return ModelResponse(
            False,
            stdout,
            extract_sql(stdout),
            round(elapsed, 2),
            f"exit={proc.returncode}: {stderr[:2000]}",
        )

    return ModelResponse(True, stdout, extract_sql(stdout), round(elapsed, 2))


def build_repair_prompt(question: str, sql: str, error: str) -> str:
    return (
        "The SQL below failed when executed against ClickHouse.\n\n"
        f"Question:\n{question}\n\n"
        f"SQL:\n{sql}\n\n"
        f"Error:\n{error}\n\n"
        "Return ONLY a corrected ClickHouse SQL query. Do not include explanation."
    )


def row_key(row: list[Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)


def exact_multiset_match(pred_rows: list[list[Any]], gold_rows: list[list[Any]]) -> bool:
    return Counter(row_key(r) for r in pred_rows) == Counter(row_key(r) for r in gold_rows)


def compare_result_sets(
    pred: SqlResult,
    gold: SqlResult,
    max_semantic_rows: int,
) -> tuple[bool, float, str]:
    if not pred.success or not gold.success:
        return False, 0.0, "One or both SQL queries failed to execute."
    if len(pred.rows) != len(gold.rows):
        return (
            False,
            0.0,
            f"Row-count mismatch: predicted={len(pred.rows)}, gold={len(gold.rows)}.",
        )
    if max(len(pred.rows), len(gold.rows)) > max_semantic_rows:
        match = exact_multiset_match(pred.rows, gold.rows)
        return (
            match,
            1.0 if match else 0.0,
            f"Large equal-sized result set; exact JSON multiset comparison above {max_semantic_rows} rows.",
        )

    comparison = compare_results(
        predicted_rows=[tuple(r) for r in pred.rows],
        gold_rows=[tuple(r) for r in gold.rows],
        predicted_cols=pred.columns,
        gold_cols=gold.columns,
        strategy=MatchStrategy.SEMANTIC,
    )
    return comparison.match, comparison.partial_score, comparison.details


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def infer_model_config(path: Path) -> tuple[str, str]:
    # Expected path: .../<model>/<config>/results.jsonl
    return path.parent.parent.name, path.parent.name


def score_and_repair_file(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_name, config = infer_model_config(path)
    model_cmd = args.model_cmd or MODEL_COMMANDS.get(model_name)
    if not model_cmd:
        raise ValueError(f"No model command configured for {model_name}")

    rows = read_jsonl(path)
    scored: list[dict[str, Any]] = []
    gold_cache: dict[str, SqlResult] = {}

    for idx, row in enumerate(rows, 1):
        if args.progress and (idx == 1 or idx % args.progress == 0 or idx == len(rows)):
            print(f"{model_name}/{config}: {idx}/{len(rows)} {row.get('query_id', '')}", flush=True)

        qid = row.get("query_id", "")
        original_sql = row.get("predicted_sql", "")
        final_sql = original_sql
        repair_attempts = 0
        repaired = False
        repair_error = ""
        repair_latency_ms = 0.0

        pred = run_clickhouse_query(
            final_sql,
            args.clickhouse_client,
            args.clickhouse_host,
            args.clickhouse_port,
            args.clickhouse_user,
            args.clickhouse_password,
            args.sql_timeout_sec,
        )

        if (
            not pred.success
            and original_sql.strip()
            and row.get("model_success")
            and args.max_repairs > 0
        ):
            for attempt in range(1, args.max_repairs + 1):
                repair_attempts = attempt
                prompt = build_repair_prompt(
                    row.get("natural_language", ""),
                    final_sql,
                    pred.error,
                )
                response = run_model_command(
                    model_cmd,
                    prompt,
                    Path(args.model_cwd or f"/tmp/datapup_cli_repair_{model_name}"),
                    args.model_timeout_sec,
                )
                repair_latency_ms += response.latency_ms
                repair_error = response.error
                if not response.success or not response.sql.strip():
                    break

                candidate_sql = clean_sql(response.sql)
                candidate = run_clickhouse_query(
                    candidate_sql,
                    args.clickhouse_client,
                    args.clickhouse_host,
                    args.clickhouse_port,
                    args.clickhouse_user,
                    args.clickhouse_password,
                    args.sql_timeout_sec,
                )
                final_sql = candidate_sql
                pred = candidate
                if candidate.success:
                    repaired = True
                    break

        if qid not in gold_cache:
            gold_cache[qid] = run_clickhouse_query(
                row.get("gold_sql", ""),
                args.clickhouse_client,
                args.clickhouse_host,
                args.clickhouse_port,
                args.clickhouse_user,
                args.clickhouse_password,
                args.sql_timeout_sec,
            )
        gold = gold_cache[qid]
        match, partial_score, details = compare_result_sets(
            pred,
            gold,
            args.max_semantic_rows,
        )

        scored.append(
            {
                "query_id": qid,
                "category": row.get("category", ""),
                "difficulty": row.get("difficulty", ""),
                "natural_language": row.get("natural_language", ""),
                "model_success": row.get("model_success", False),
                "original_executed": bool(original_sql.strip()) and row.get("model_success", False),
                "final_executed": pred.success,
                "gold_executed": gold.success,
                "result_match": match,
                "partial_score": partial_score,
                "repair_attempts": repair_attempts,
                "repaired": repaired,
                "repair_error": repair_error,
                "repair_latency_ms": round(repair_latency_ms, 2),
                "pred_error": pred.error,
                "gold_error": gold.error,
                "pred_rows": len(pred.rows),
                "gold_rows": len(gold.rows),
                "comparison_details": details,
                "original_sql": original_sql,
                "final_sql": final_sql,
                "gold_sql": row.get("gold_sql", ""),
            }
        )

    total = len(scored)
    summary = {
        "model_name": model_name,
        "config": config,
        "path": str(path),
        "total": total,
        "final_execution_rate": sum(r["final_executed"] for r in scored) / total if total else 0.0,
        "gold_execution_rate": sum(r["gold_executed"] for r in scored) / total if total else 0.0,
        "result_correctness": sum(r["result_match"] for r in scored) / total if total else 0.0,
        "avg_partial_score": sum(float(r["partial_score"]) for r in scored) / total if total else 0.0,
        "repair_attempts": sum(int(r["repair_attempts"]) for r in scored),
        "repair_successes": sum(r["repaired"] for r in scored),
        "avg_repair_latency_ms": (
            sum(float(r["repair_latency_ms"]) for r in scored) / total if total else 0.0
        ),
        "per_category": {},
    }

    for row in scored:
        bucket = summary["per_category"].setdefault(
            row["category"],
            {"total": 0, "executed": 0, "correct": 0, "repair_successes": 0},
        )
        bucket["total"] += 1
        bucket["executed"] += int(row["final_executed"])
        bucket["correct"] += int(row["result_match"])
        bucket["repair_successes"] += int(row["repaired"])

    return summary, scored


def write_outputs(output_dir: Path, label: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = label.replace("/", "__")
    (output_dir / f"{safe}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / f"{safe}.scored.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cmd", default="")
    parser.add_argument("--model-cwd", default="")
    parser.add_argument("--model-timeout-sec", type=int, default=300)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--clickhouse-client", default=str(PROJECT_ROOT / "evaluation/bin/clickhouse-client"))
    parser.add_argument("--clickhouse-host", default="localhost")
    parser.add_argument("--clickhouse-port", type=int, default=19000)
    parser.add_argument("--clickhouse-user", default="default")
    parser.add_argument("--clickhouse-password", default="")
    parser.add_argument("--sql-timeout-sec", type=int, default=60)
    parser.add_argument("--max-semantic-rows", type=int, default=5000)
    parser.add_argument("--progress", type=int, default=25)
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for path in args.paths:
        summary, rows = score_and_repair_file(path, args)
        label = f"{summary['model_name']}/{summary['config']}"
        write_outputs(args.output_dir, label, summary, rows)
        summaries.append(summary)
        print(
            f"{label}: N={summary['total']} "
            f"exec={summary['final_execution_rate']:.1%} "
            f"correct={summary['result_correctness']:.1%} "
            f"repairs={summary['repair_successes']}/{summary['repair_attempts']}",
            flush=True,
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
