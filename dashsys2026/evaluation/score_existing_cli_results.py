#!/usr/bin/env python3
"""Score already-generated CLI text-to-SQL results.

This script does not call any model. It reads result JSONL files produced by
run_cli_experiments.py, executes predicted and gold SQL against ClickHouse, and
compares result sets with the same semantic comparator used by the main runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.framework.result_comparator import MatchStrategy, compare_results  # noqa: E402


@dataclass
class SqlResult:
    success: bool
    rows: list[list[Any]]
    columns: list[str]
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


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_key(row: list[Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)


def exact_multiset_match(pred_rows: list[list[Any]], gold_rows: list[list[Any]]) -> bool:
    return Counter(row_key(r) for r in pred_rows) == Counter(row_key(r) for r in gold_rows)


def score_file(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = iter_jsonl(path)
    gold_cache: dict[str, SqlResult] = {}
    scored: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, 1):
        qid = row.get("query_id", "")
        if args.progress and (idx == 1 or idx % args.progress == 0 or idx == len(rows)):
            print(f"{path.parent.parent.name}/{path.parent.name}: {idx}/{len(rows)} {qid}", flush=True)

        pred = run_clickhouse_query(
            row.get("predicted_sql", ""),
            args.clickhouse_client,
            args.clickhouse_host,
            args.clickhouse_port,
            args.clickhouse_user,
            args.clickhouse_password,
            args.sql_timeout_sec,
        )
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

        match = False
        partial_score = 0.0
        details = ""
        if pred.success and gold.success:
            if len(pred.rows) != len(gold.rows):
                details = (
                    f"Row-count mismatch: predicted={len(pred.rows)}, "
                    f"gold={len(gold.rows)}."
                )
            elif max(len(pred.rows), len(gold.rows)) > args.max_semantic_rows:
                match = exact_multiset_match(pred.rows, gold.rows)
                partial_score = 1.0 if match else 0.0
                details = (
                    "Large equal-sized result set; used exact JSON multiset "
                    f"comparison above {args.max_semantic_rows} rows."
                )
            else:
                comparison = compare_results(
                    predicted_rows=[tuple(r) for r in pred.rows],
                    gold_rows=[tuple(r) for r in gold.rows],
                    predicted_cols=pred.columns,
                    gold_cols=gold.columns,
                    strategy=MatchStrategy.SEMANTIC,
                )
                match = comparison.match
                partial_score = comparison.partial_score
                details = comparison.details

        scored.append(
            {
                "query_id": qid,
                "category": row.get("category", ""),
                "difficulty": row.get("difficulty", ""),
                "natural_language": row.get("natural_language", ""),
                "model_success": row.get("model_success", False),
                "pred_executed": pred.success,
                "gold_executed": gold.success,
                "result_match": match,
                "partial_score": partial_score,
                "pred_rows": len(pred.rows),
                "gold_rows": len(gold.rows),
                "pred_latency_ms": pred.latency_ms,
                "gold_latency_ms": gold.latency_ms,
                "pred_error": pred.error,
                "gold_error": gold.error,
                "comparison_details": details,
                "predicted_sql": row.get("predicted_sql", ""),
                "gold_sql": row.get("gold_sql", ""),
            }
        )

    total = len(scored)
    pred_executed = sum(r["pred_executed"] for r in scored)
    gold_executed = sum(r["gold_executed"] for r in scored)
    correct = sum(r["result_match"] for r in scored)
    partial = sum(float(r["partial_score"]) for r in scored) / total if total else 0.0

    per_category: dict[str, dict[str, Any]] = {}
    for row in scored:
        cat = row["category"]
        bucket = per_category.setdefault(cat, {"total": 0, "executed": 0, "correct": 0})
        bucket["total"] += 1
        bucket["executed"] += int(row["pred_executed"])
        bucket["correct"] += int(row["result_match"])

    summary = {
        "path": str(path),
        "total": total,
        "pred_execution_rate": pred_executed / total if total else 0.0,
        "gold_execution_rate": gold_executed / total if total else 0.0,
        "result_correctness": correct / total if total else 0.0,
        "avg_partial_score": partial,
        "per_category": per_category,
    }
    return summary, scored


def write_outputs(
    output_dir: Path,
    label: str,
    summary: dict[str, Any],
    scored: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "__")
    (output_dir / f"{safe_label}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / f"{safe_label}.scored.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()) if scored else [])
        if scored:
            writer.writeheader()
            writer.writerows(scored)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
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
        summary, scored = score_file(path, args)
        summaries.append(summary)
        label = "/".join(path.parts[-3:-1])
        if args.output_dir:
            write_outputs(args.output_dir, label, summary, scored)

        print(
            f"{label}: N={summary['total']} "
            f"exec={summary['pred_execution_rate']:.1%} "
            f"gold={summary['gold_execution_rate']:.1%} "
            f"correct={summary['result_correctness']:.1%} "
            f"partial={summary['avg_partial_score']:.3f}"
        )

    if args.output_dir:
        (args.output_dir / "summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
