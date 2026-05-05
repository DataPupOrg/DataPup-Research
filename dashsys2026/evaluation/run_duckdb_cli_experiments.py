#!/usr/bin/env python3
"""Run a focused DuckDB cross-dialect CLI validation."""

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

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.duckdb_dialect import (  # noqa: E402
    DUCKDB_JSON_SCHEMA,
    schema_for_tables,
    translate_clickhouse_to_duckdb,
)
from evaluation.framework.result_comparator import MatchStrategy, compare_results  # noqa: E402
from evaluation.run_cli_experiments import extract_sql  # noqa: E402


@dataclass
class ModelResponse:
    success: bool
    raw_response: str
    sql: str
    latency_ms: float
    error: str = ""


@dataclass
class SqlResult:
    success: bool
    rows: list[tuple[Any, ...]]
    columns: list[str]
    latency_ms: float
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-cmd", required=True)
    parser.add_argument("--model-cwd", type=Path, default=Path("/tmp/datapup_duckdb_cli"))
    parser.add_argument("--model-timeout-sec", type=int, default=300)
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "evaluation/duckdb/datapup.duckdb")
    parser.add_argument("--config", choices=["baseline", "best"], required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluation/results/duckdb_cli_validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--progress", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_queries() -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for path in sorted((PROJECT_ROOT / "evaluation/benchmark/queries").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("queries", [])
        for item in items:
            if item.get("dataset") == "custom_analytics":
                queries.append(item)
    return queries


def run_sql(con: duckdb.DuckDBPyConnection, sql: str) -> SqlResult:
    if not sql.strip():
        return SqlResult(False, [], [], 0.0, "empty SQL")
    start = time.perf_counter()
    try:
        cur = con.execute(sql)
        rows = [tuple(row) for row in cur.fetchall()]
        cols = [d[0] for d in cur.description] if cur.description else []
        return SqlResult(True, rows, cols, round((time.perf_counter() - start) * 1000, 2))
    except Exception as exc:
        return SqlResult(False, [], [], round((time.perf_counter() - start) * 1000, 2), str(exc)[:2000])


def portable_queries(con: duckdb.DuckDBPyConnection, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for query in queries:
        duckdb_sql = translate_clickhouse_to_duckdb(query["sql"])
        result = run_sql(con, duckdb_sql)
        if result.success:
            copied = dict(query)
            copied["duckdb_gold_sql"] = duckdb_sql
            copied["duckdb_gold_rows"] = len(result.rows)
            selected.append(copied)
    return selected


def load_examples() -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "evaluation/benchmark/examples/examples.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("examples", data if isinstance(data, list) else [])


def tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def choose_examples(question: str, tables: list[str], limit: int = 3) -> list[dict[str, Any]]:
    examples = load_examples()
    q_tokens = tokens(question)
    wanted_tables = {t.split(".")[-1] for t in tables}
    scored: list[tuple[int, dict[str, Any]]] = []
    for example in examples:
        score = len(q_tokens & tokens(example.get("question", "")))
        example_tables = {t.split(".")[-1] for t in example.get("tables_used", [])}
        score += 4 * len(wanted_tables & example_tables)
        try:
            translated_sql = translate_clickhouse_to_duckdb(example["sql"])
        except Exception:
            continue
        scored.append((score, {**example, "duckdb_sql": translated_sql}))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:limit]]


def build_prompt(query: dict[str, Any], config: str) -> str:
    question = query["natural_language"]
    tables = [t.split(".")[-1] for t in query.get("tables_used", [])]
    if config == "baseline":
        schema = DUCKDB_JSON_SCHEMA
        return (
            "You are a SQL generator. Return only one executable DuckDB SQL query, with no explanation. "
            "Use fully qualified table names such as analytics.events.\n\n"
            f"Schema:\n{schema}\n\n"
            f"Question: {question}\n\nSQL:"
        )

    schema = schema_for_tables(tables, "markdown")
    examples = choose_examples(question, tables, 3)
    examples_text = "\n\n".join(
        f"Question: {ex['question']}\nSQL: {ex['duckdb_sql']}"
        for ex in examples
    )
    return (
        "You generate DuckDB SQL for an analyst-facing database client. "
        "Return only one executable SQL query, with no markdown fence and no explanation. "
        "Use fully qualified tables such as analytics.events. "
        "Use count(*) rather than count(); use count(*) FILTER (WHERE ...) for conditional counts; "
        "use date_trunc('month', ts) for month buckets; use json_extract_string(json_col, '$.key') for JSON fields. "
        "Preserve the requested result shape: do not add extra columns, do not recode numeric flags into labels, "
        "and do not change ranking or LIMIT choices unless the question asks for it. Prefer explicit aliases and deterministic ORDER BY clauses when the question asks for top results.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Examples:\n{examples_text}\n\n"
        f"Question: {question}\n\nSQL:"
    )


def run_model_command(model_cmd: str, prompt: str, cwd: Path, timeout_sec: int) -> ModelResponse:
    cwd.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".prompt.txt", delete=False) as handle:
        handle.write(prompt)
        prompt_file = Path(handle.name)
    try:
        if "{prompt_file}" in model_cmd or "{prompt}" in model_cmd:
            cmd = model_cmd.replace("{prompt_file}", shlex.quote(str(prompt_file)))
            cmd = cmd.replace("{prompt}", shlex.quote(prompt))
            proc = subprocess.run(cmd, shell=True, cwd=str(cwd), text=True, capture_output=True, timeout=timeout_sec)
        else:
            proc = subprocess.run(model_cmd, shell=True, cwd=str(cwd), input=prompt, text=True, capture_output=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        return ModelResponse(False, "", "", round((time.perf_counter() - start) * 1000, 2), f"timeout: {exc}")
    finally:
        try:
            prompt_file.unlink()
        except OSError:
            pass
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    stdout = proc.stdout or ""
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return ModelResponse(False, stdout, extract_sql(stdout), elapsed, f"exit={proc.returncode}: {stderr[:2000]}")
    return ModelResponse(True, stdout, extract_sql(stdout), elapsed)


def score_result(pred: SqlResult, gold: SqlResult) -> tuple[bool, float, str]:
    if not pred.success or not gold.success:
        return False, 0.0, "One or both SQL queries failed to execute."
    if len(pred.rows) != len(gold.rows):
        return False, 0.0, f"Row-count mismatch: predicted={len(pred.rows)}, gold={len(gold.rows)}."
    if max(len(pred.rows), len(gold.rows)) > 5000:
        pred_counts = Counter(json.dumps(row, sort_keys=True, default=str) for row in pred.rows)
        gold_counts = Counter(json.dumps(row, sort_keys=True, default=str) for row in gold.rows)
        match = pred_counts == gold_counts
        return (
            match,
            1.0 if match else 0.0,
            "Large equal-sized result set; used exact JSON multiset comparison above 5000 rows.",
        )
    comparison = compare_results(
        predicted_rows=pred.rows,
        gold_rows=gold.rows,
        predicted_cols=pred.columns,
        gold_cols=gold.columns,
        strategy=MatchStrategy.SEMANTIC,
    )
    return comparison.match, comparison.partial_score, comparison.details


def main() -> int:
    args = parse_args()
    con = duckdb.connect(str(args.db_path), read_only=True)
    queries = portable_queries(con, load_queries())
    if args.query_id:
        wanted = set(args.query_id)
        queries = [q for q in queries if q["id"] in wanted]
    if args.category:
        wanted_categories = set(args.category)
        queries = [q for q in queries if q["category"] in wanted_categories]
    if args.limit:
        queries = queries[: args.limit]

    out_dir = args.output_dir / args.model_name / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    csv_path = out_dir / "results.csv"
    summary_path = out_dir / "summary.json"
    if jsonl_path.exists() and not args.overwrite:
        raise SystemExit(f"{jsonl_path} exists; pass --overwrite")

    rows: list[dict[str, Any]] = []
    gold_cache: dict[str, SqlResult] = {}
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for idx, query in enumerate(queries, 1):
            if args.progress and (idx == 1 or idx % args.progress == 0 or idx == len(queries)):
                print(f"{args.model_name}/{args.config}: {idx}/{len(queries)} {query['id']}", flush=True)
            prompt = build_prompt(query, args.config)
            model = run_model_command(args.model_cmd, prompt, args.model_cwd, args.model_timeout_sec)
            pred = run_sql(con, model.sql)
            gold = gold_cache.setdefault(query["id"], run_sql(con, query["duckdb_gold_sql"]))
            match, partial, details = score_result(pred, gold)
            row = {
                "query_id": query["id"],
                "category": query["category"],
                "difficulty": query.get("difficulty", ""),
                "natural_language": query["natural_language"],
                "model_success": model.success,
                "pred_executed": pred.success,
                "gold_executed": gold.success,
                "result_match": match,
                "partial_score": partial,
                "pred_rows": len(pred.rows),
                "gold_rows": len(gold.rows),
                "model_latency_ms": model.latency_ms,
                "sql_latency_ms": pred.latency_ms,
                "prompt_chars": len(prompt),
                "output_chars": len(model.raw_response),
                "pred_error": pred.error,
                "model_error": model.error,
                "comparison_details": details,
                "predicted_sql": model.sql,
                "gold_sql": query["duckdb_gold_sql"],
                "raw_response": model.raw_response,
            }
            rows.append(row)
            jsonl.write(json.dumps(row, default=str) + "\n")
            jsonl.flush()

    total = len(rows)
    summary = {
        "model_name": args.model_name,
        "config": args.config,
        "db_path": str(args.db_path),
        "total": total,
        "portable_gold_queries": len(portable_queries(con, load_queries())),
        "model_success_rate": sum(r["model_success"] for r in rows) / total if total else 0.0,
        "execution_rate": sum(r["pred_executed"] for r in rows) / total if total else 0.0,
        "result_correctness": sum(r["result_match"] for r in rows) / total if total else 0.0,
        "avg_partial_score": sum(float(r["partial_score"]) for r in rows) / total if total else 0.0,
        "avg_prompt_chars": sum(r["prompt_chars"] for r in rows) / total if total else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
