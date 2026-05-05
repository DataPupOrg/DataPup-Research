#!/usr/bin/env python3
"""
Run DataPup text-to-SQL experiments through local/hosted model CLIs.

This runner is intentionally independent of Anthropic/OpenAI/Gemini Python SDKs.
It builds prompts from the existing DataPup benchmark files, calls an arbitrary
model command, extracts SQL from stdout, and optionally executes predicted/gold
SQL with clickhouse-client for correctness metrics.

Typical use:

    python3 evaluation/run_cli_experiments.py \
      --model-name claude_cli \
      --model-cmd "claude -p" \
      --model-cwd /tmp/datapup_cli_model_cwd \
      --config best \
      --limit 5

If --execute is provided, clickhouse-client must be installed and the benchmark
data must be loaded into the local ClickHouse instance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.framework.prompt_builder import (  # noqa: E402
    ExampleStrategy,
    MetadataLevel,
    PromptBuilder,
    PromptVersion,
    SchemaFormat,
    SchemaScope,
)
from evaluation.framework.result_comparator import (  # noqa: E402
    MatchStrategy,
    compare_results,
)


CONFIGS = {
    "baseline": {
        "schema_format": SchemaFormat.MARKDOWN,
        "scope": SchemaScope.FULL,
        "metadata": MetadataLevel.NONE,
        "examples": ExampleStrategy.ZERO_SHOT,
        "prompt_version": PromptVersion.MINIMAL,
    },
    "full_context": {
        "schema_format": SchemaFormat.MARKDOWN,
        "scope": SchemaScope.FULL,
        "metadata": MetadataLevel.NONE,
        "examples": ExampleStrategy.ZERO_SHOT,
        "prompt_version": PromptVersion.FULL,
    },
    "best": {
        "schema_format": SchemaFormat.MARKDOWN,
        "scope": SchemaScope.RELEVANT_SUBSET,
        "metadata": MetadataLevel.DESCRIPTIONS,
        "examples": ExampleStrategy.DYNAMIC_FEW_SHOT,
        "prompt_version": PromptVersion.WINDOW,
    },
    "best_full_prompt": {
        "schema_format": SchemaFormat.MARKDOWN,
        "scope": SchemaScope.RELEVANT_SUBSET,
        "metadata": MetadataLevel.DESCRIPTIONS,
        "examples": ExampleStrategy.DYNAMIC_FEW_SHOT,
        "prompt_version": PromptVersion.FULL,
    },
}


@dataclass
class CliResponse:
    raw_response: str
    sql: str
    success: bool
    latency_ms: float
    error: str = ""


@dataclass
class SqlResult:
    success: bool
    rows: list[list[Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class QueryRun:
    query_id: str
    category: str
    difficulty: str
    natural_language: str
    gold_sql: str
    predicted_sql: str
    raw_response: str
    model_success: bool
    model_error: str
    pred_executed: bool
    gold_executed: bool
    pred_error: str
    result_match: bool
    partial_score: float
    pred_row_count: int
    gold_row_count: int
    token_estimate: int
    prompt_chars: int
    output_chars: int
    model_latency_ms: float
    sql_latency_ms: float
    repair_attempts: int
    repaired: bool
    trace: list[dict[str, Any]]


def load_queries(benchmark_dir: Path, dataset: str) -> list[dict[str, Any]]:
    queries_dir = benchmark_dir / "queries"
    queries: list[dict[str, Any]] = []
    for path in sorted(queries_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("queries", [])
        for item in items:
            if item.get("dataset", "").lower() == dataset.lower():
                queries.append(item)
    return queries


def filter_queries(
    queries: list[dict[str, Any]],
    query_ids: set[str] | None,
    categories: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = queries
    if query_ids:
        selected = [q for q in selected if q.get("id") in query_ids]
    if categories:
        selected = [q for q in selected if q.get("category") in categories]
    if limit is not None:
        selected = selected[:limit]
    return selected


def extract_sql(response: str) -> str:
    if not response or not response.strip():
        return ""
    text = response.strip()

    fence_pattern = re.compile(
        r"```(?:sql|clickhouse|SQL)?\s*\n?(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    matches = fence_pattern.findall(text)
    if matches:
        return clean_sql(matches[0])

    starts = re.search(
        r"\b(WITH|SELECT|INSERT|CREATE|ALTER|DELETE|UPDATE)\b",
        text,
        re.IGNORECASE,
    )
    if starts:
        return clean_sql(text[starts.start() :])
    return clean_sql(text)


def clean_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql|clickhouse)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip()
    # Drop trailing explanation after the first semicolon if present.
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip()
    return sql


def estimate_output_tokens(text: str) -> int:
    return max(1, int((len(text) + 3) / 4)) if text else 0


def run_model_command(
    model_cmd: str,
    prompt: str,
    cwd: Path | None,
    timeout_sec: int,
) -> CliResponse:
    cwd_path = cwd or PROJECT_ROOT
    cwd_path.mkdir(parents=True, exist_ok=True)
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
            cmd = model_cmd
            cmd = cmd.replace("{prompt_file}", shlex.quote(str(prompt_file)))
            cmd = cmd.replace("{prompt}", shlex.quote(prompt))
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(cwd_path),
                text=True,
                capture_output=True,
                timeout=timeout_sec,
            )
        else:
            proc = subprocess.run(
                model_cmd,
                shell=True,
                cwd=str(cwd_path),
                input=prompt,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
            )
    except subprocess.TimeoutExpired as e:
        elapsed = (time.perf_counter() - start) * 1000
        return CliResponse("", "", False, round(elapsed, 2), f"timeout: {e}")
    finally:
        try:
            prompt_file.unlink()
        except OSError:
            pass

    elapsed = (time.perf_counter() - start) * 1000
    stdout = proc.stdout or ""
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return CliResponse(
            raw_response=stdout,
            sql=extract_sql(stdout),
            success=False,
            latency_ms=round(elapsed, 2),
            error=f"exit={proc.returncode}: {stderr[:2000]}",
        )

    return CliResponse(
        raw_response=stdout,
        sql=extract_sql(stdout),
        success=True,
        latency_ms=round(elapsed, 2),
    )


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
        return SqlResult(False, error="empty SQL")

    if shutil.which(clickhouse_client) is None and not Path(clickhouse_client).exists():
        return SqlResult(False, error=f"{clickhouse_client} not found")

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
        return SqlResult(False, latency_ms=round(elapsed, 2), error=f"timeout: {e}")

    elapsed = (time.perf_counter() - start) * 1000
    if proc.returncode != 0:
        return SqlResult(
            False,
            latency_ms=round(elapsed, 2),
            error=(proc.stderr or proc.stdout or "").strip()[:2000],
        )

    try:
        payload = json.loads(proc.stdout)
        meta = payload.get("meta", [])
        columns = [m.get("name", "") for m in meta]
        rows = payload.get("data", [])
        return SqlResult(
            True,
            rows=rows,
            columns=columns,
            row_count=len(rows),
            latency_ms=round(elapsed, 2),
        )
    except Exception as e:
        return SqlResult(
            False,
            latency_ms=round(elapsed, 2),
            error=f"failed to parse JSONCompact output: {e}",
        )


def heuristic_relevant_tables(prompt_builder: PromptBuilder, dataset: str, question: str) -> list[str]:
    tables = prompt_builder._parse_schema_metadata(dataset)  # noqa: SLF001
    q = question.lower()
    q_tokens = set(re.findall(r"[a-z0-9_]+", q))
    scored: list[tuple[int, str]] = []

    for table in tables:
        name = table.table_name.lower()
        score = 0
        variants = {name, name.rstrip("s"), name + "s"}
        if variants & q_tokens:
            score += 5
        if name in q:
            score += 5
        for col in table.columns:
            col_name = col["name"].lower()
            parts = set(col_name.split("_"))
            if col_name in q:
                score += 3
            score += len(parts & q_tokens)
        scored.append((score, table.table_name))

    picked = [name for score, name in scored if score > 0]
    if picked:
        return picked

    # Conservative fallback: include all tables rather than silently dropping
    # a needed join target.
    return [t.table_name for t in tables]


def build_prompt_for_query(
    prompt_builder: PromptBuilder,
    query: dict[str, Any],
    dataset: str,
    config: dict[str, Any],
    relevant_source: str,
) -> tuple[Any, list[str]]:
    kwargs: dict[str, Any] = {
        "question": query.get("natural_language", ""),
        "dataset": dataset,
        "format": config["schema_format"],
        "scope": config["scope"],
        "metadata": config["metadata"],
        "examples": config["examples"],
        "prompt_version": config["prompt_version"],
    }
    selected_tables: list[str] = []

    if config["scope"] == SchemaScope.RELEVANT_SUBSET:
        if relevant_source == "gold":
            selected_tables = [
                t.split(".")[-1] for t in query.get("tables_used", [])
            ]
            kwargs["relevant_tables"] = selected_tables
            kwargs["relevant_columns"] = [
                c.split(".")[-1] for c in query.get("columns_used", [])
            ]
        elif relevant_source == "heuristic":
            selected_tables = heuristic_relevant_tables(
                prompt_builder,
                dataset,
                query.get("natural_language", ""),
            )
            kwargs["relevant_tables"] = selected_tables
        else:
            raise ValueError(f"unknown relevant source: {relevant_source}")

    prompt_result = prompt_builder.build_prompt(**kwargs)
    return prompt_result, selected_tables


def build_repair_prompt(question: str, sql: str, error: str) -> str:
    return (
        "The SQL below failed when executed against ClickHouse.\n\n"
        f"Question:\n{question}\n\n"
        f"SQL:\n{sql}\n\n"
        f"Error:\n{error}\n\n"
        "Return ONLY a corrected ClickHouse SQL query. Do not include explanation."
    )


def evaluate_one(
    query: dict[str, Any],
    args: argparse.Namespace,
    prompt_builder: PromptBuilder,
    config: dict[str, Any],
) -> QueryRun:
    prompt_result, selected_tables = build_prompt_for_query(
        prompt_builder,
        query,
        args.dataset,
        config,
        args.relevant_source,
    )
    prompt_text = prompt_result.system_message + "\n\n" + prompt_result.user_message

    trace: list[dict[str, Any]] = [
        {
            "step": 1,
            "action": "select_context",
            "strategy": args.config,
            "schema_scope": config["scope"].value,
            "relevant_source": args.relevant_source,
            "selected_tables": selected_tables,
            "prompt_tokens_estimate": prompt_result.token_estimate,
        }
    ]

    model_response = run_model_command(
        args.model_cmd,
        prompt_text,
        Path(args.model_cwd) if args.model_cwd else None,
        args.model_timeout_sec,
    )
    trace.append(
        {
            "step": len(trace) + 1,
            "action": "generate_sql",
            "status": "success" if model_response.success else "error",
            "sql": model_response.sql,
            "latency_ms": model_response.latency_ms,
            "error": model_response.error,
        }
    )

    predicted_sql = model_response.sql
    pred_result = SqlResult(False)
    gold_result = SqlResult(False)
    result_match = False
    partial_score = 0.0
    repaired = False
    repair_attempts = 0

    if args.execute:
        pred_result = run_clickhouse_query(
            predicted_sql,
            args.clickhouse_client,
            args.clickhouse_host,
            args.clickhouse_port,
            args.clickhouse_user,
            args.clickhouse_password,
            args.sql_timeout_sec,
        )
        trace.append(
            {
                "step": len(trace) + 1,
                "action": "execute_sql",
                "status": "success" if pred_result.success else "error",
                "row_count": pred_result.row_count,
                "latency_ms": pred_result.latency_ms,
                "error": pred_result.error,
            }
        )

        if (
            args.repair_on_error
            and not pred_result.success
            and predicted_sql.strip()
            and model_response.success
        ):
            for attempt in range(1, args.max_repairs + 1):
                repair_attempts = attempt
                repair_prompt = build_repair_prompt(
                    query.get("natural_language", ""),
                    predicted_sql,
                    pred_result.error,
                )
                repair_response = run_model_command(
                    args.model_cmd,
                    repair_prompt,
                    Path(args.model_cwd) if args.model_cwd else None,
                    args.model_timeout_sec,
                )
                trace.append(
                    {
                        "step": len(trace) + 1,
                        "action": "repair_sql",
                        "attempt": attempt,
                        "status": "success" if repair_response.success else "error",
                        "sql": repair_response.sql,
                        "latency_ms": repair_response.latency_ms,
                        "error": repair_response.error,
                    }
                )
                if not repair_response.success or not repair_response.sql.strip():
                    break
                repaired_sql = repair_response.sql
                repaired_result = run_clickhouse_query(
                    repaired_sql,
                    args.clickhouse_client,
                    args.clickhouse_host,
                    args.clickhouse_port,
                    args.clickhouse_user,
                    args.clickhouse_password,
                    args.sql_timeout_sec,
                )
                trace.append(
                    {
                        "step": len(trace) + 1,
                        "action": "execute_repaired_sql",
                        "attempt": attempt,
                        "status": "success" if repaired_result.success else "error",
                        "row_count": repaired_result.row_count,
                        "latency_ms": repaired_result.latency_ms,
                        "error": repaired_result.error,
                    }
                )
                predicted_sql = repaired_sql
                pred_result = repaired_result
                if repaired_result.success:
                    repaired = True
                    break

        gold_result = run_clickhouse_query(
            query.get("sql", ""),
            args.clickhouse_client,
            args.clickhouse_host,
            args.clickhouse_port,
            args.clickhouse_user,
            args.clickhouse_password,
            args.sql_timeout_sec,
        )
        if pred_result.success and gold_result.success:
            comparison = compare_results(
                predicted_rows=[tuple(r) for r in pred_result.rows],
                gold_rows=[tuple(r) for r in gold_result.rows],
                predicted_cols=pred_result.columns,
                gold_cols=gold_result.columns,
                strategy=MatchStrategy.SEMANTIC,
            )
            result_match = comparison.match
            partial_score = comparison.partial_score

    return QueryRun(
        query_id=query.get("id", ""),
        category=query.get("category", ""),
        difficulty=query.get("difficulty", ""),
        natural_language=query.get("natural_language", ""),
        gold_sql=query.get("sql", ""),
        predicted_sql=predicted_sql,
        raw_response=model_response.raw_response,
        model_success=model_response.success,
        model_error=model_response.error,
        pred_executed=pred_result.success,
        gold_executed=gold_result.success,
        pred_error=pred_result.error,
        result_match=result_match,
        partial_score=partial_score,
        pred_row_count=pred_result.row_count,
        gold_row_count=gold_result.row_count,
        token_estimate=prompt_result.token_estimate,
        prompt_chars=len(prompt_text),
        output_chars=len(model_response.raw_response),
        model_latency_ms=model_response.latency_ms,
        sql_latency_ms=pred_result.latency_ms + gold_result.latency_ms,
        repair_attempts=repair_attempts,
        repaired=repaired,
        trace=trace,
    )


def write_summary(results: list[QueryRun], out_path: Path, args: argparse.Namespace) -> None:
    total = len(results)
    if total == 0:
        return
    model_success = sum(r.model_success for r in results)
    pred_executed = sum(r.pred_executed for r in results)
    gold_executed = sum(r.gold_executed for r in results)
    correct = sum(r.result_match for r in results)
    repaired = sum(r.repaired for r in results)
    repair_attempts = sum(r.repair_attempts for r in results)
    avg_prompt_tokens = sum(r.token_estimate for r in results) / total
    avg_model_latency = sum(r.model_latency_ms for r in results) / total
    avg_sql_latency = sum(r.sql_latency_ms for r in results) / total
    avg_steps = sum(len(r.trace) for r in results) / total

    per_category: dict[str, dict[str, Any]] = {}
    for r in results:
        item = per_category.setdefault(
            r.category,
            {"total": 0, "model_success": 0, "pred_executed": 0, "correct": 0},
        )
        item["total"] += 1
        item["model_success"] += int(r.model_success)
        item["pred_executed"] += int(r.pred_executed)
        item["correct"] += int(r.result_match)

    summary = {
        "model_name": args.model_name,
        "config": args.config,
        "dataset": args.dataset,
        "execute": args.execute,
        "repair_on_error": args.repair_on_error,
        "total_queries": total,
        "model_success_rate": model_success / total,
        "execution_accuracy": pred_executed / total,
        "gold_execution_accuracy": gold_executed / total,
        "result_correctness": correct / total,
        "repair_successes": repaired,
        "repair_attempts": repair_attempts,
        "avg_prompt_tokens_estimate": avg_prompt_tokens,
        "avg_model_latency_ms": avg_model_latency,
        "avg_sql_latency_ms": avg_sql_latency,
        "avg_trace_steps": avg_steps,
        "per_category": per_category,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_csv(results: list[QueryRun], out_path: Path) -> None:
    fields = [
        "query_id",
        "category",
        "difficulty",
        "model_success",
        "pred_executed",
        "result_match",
        "repaired",
        "repair_attempts",
        "token_estimate",
        "model_latency_ms",
        "sql_latency_ms",
        "pred_row_count",
        "gold_row_count",
        "pred_error",
        "natural_language",
        "predicted_sql",
        "gold_sql",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            d = asdict(r)
            writer.writerow({k: d.get(k) for k in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--model-cmd",
        required=True,
        help=(
            "Shell command used to call the model. If it contains {prompt_file}, "
            "the placeholder is replaced with a temp prompt path. Otherwise the "
            "prompt is sent on stdin."
        ),
    )
    parser.add_argument(
        "--model-cwd",
        default="",
        help="Directory used as cwd for model CLI calls. Use /tmp/... to avoid repo-specific CLI prompts.",
    )
    parser.add_argument("--model-timeout-sec", type=int, default=180)
    parser.add_argument("--dataset", default="custom_analytics")
    parser.add_argument("--config", choices=sorted(CONFIGS), default="best")
    parser.add_argument(
        "--relevant-source",
        choices=["gold", "heuristic"],
        default="heuristic",
        help="How relevant-subset configs choose tables. heuristic is paper-safe; gold is an oracle.",
    )
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repair-on-error", action="store_true")
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--clickhouse-client", default="clickhouse-client")
    parser.add_argument("--clickhouse-host", default="localhost")
    parser.add_argument("--clickhouse-port", type=int, default=9000)
    parser.add_argument("--clickhouse-user", default="default")
    parser.add_argument("--clickhouse-password", default=os.environ.get("CLICKHOUSE_PASSWORD", ""))
    parser.add_argument("--sql-timeout-sec", type=int, default=60)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "evaluation" / "results" / "cli_runs"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing results.jsonl and skip query IDs already present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_dir = PROJECT_ROOT / "evaluation" / "benchmark"
    prompt_builder = PromptBuilder(str(benchmark_dir))
    config = CONFIGS[args.config]

    queries = load_queries(benchmark_dir, args.dataset)
    queries = filter_queries(
        queries,
        set(args.query_id) if args.query_id else None,
        set(args.category) if args.category else None,
        args.limit,
    )
    if not queries:
        print("No queries selected.", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir) / args.model_name / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "results.csv"

    if args.resume and args.overwrite:
        print("--resume and --overwrite are mutually exclusive.", file=sys.stderr)
        return 2

    if jsonl_path.exists() and not args.overwrite and not args.resume:
        print(f"Refusing to overwrite existing {jsonl_path}; pass --overwrite.", file=sys.stderr)
        return 2

    results: list[QueryRun] = []
    seen_query_ids: set[str] = set()
    if args.resume and jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    result = QueryRun(**row)
                except Exception as e:
                    print(
                        f"Could not parse existing row in {jsonl_path}: {e}",
                        file=sys.stderr,
                    )
                    return 2
                if result.query_id in seen_query_ids:
                    continue
                seen_query_ids.add(result.query_id)
                results.append(result)

    remaining_queries = [
        q for q in queries
        if q.get("id", "") not in seen_query_ids
    ]
    if args.resume:
        print(
            f"Resume mode: loaded {len(results)} existing row(s); "
            f"{len(remaining_queries)} query(ies) remaining.",
            flush=True,
        )

    mode = "a" if args.resume else "w"
    with jsonl_path.open(mode, encoding="utf-8") as f:
        for offset, query in enumerate(remaining_queries, 1):
            idx = len(results) + 1
            qid = query.get("id", f"q{idx}")
            print(f"[{idx}/{len(queries)}] {args.model_name}/{args.config}: {qid}", flush=True)
            result = evaluate_one(query, args, prompt_builder, config)
            results.append(result)
            f.write(json.dumps(asdict(result), default=str) + "\n")
            f.flush()

            status = "CORRECT" if result.result_match else (
                "EXEC" if result.pred_executed else (
                    "SQL" if result.predicted_sql else "FAIL"
                )
            )
            print(
                f"  {status} | tok~{result.token_estimate} | "
                f"model {result.model_latency_ms:.0f}ms | repairs={result.repair_attempts}",
                flush=True,
            )

    write_summary(results, summary_path, args)
    write_csv(results, csv_path)

    print(f"\nWrote:\n  {jsonl_path}\n  {summary_path}\n  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
