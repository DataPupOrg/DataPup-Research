#!/usr/bin/env python3
"""
Summarize DataPup CLI experiment runs.

Reads summary.json files produced by run_cli_experiments.py and emits:
  - cli_runs_summary.csv
  - cli_runs_summary.md
  - cli_runs_failures.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "model_name",
    "config",
    "dataset",
    "total_queries",
    "model_success_rate",
    "execution_accuracy",
    "gold_execution_accuracy",
    "result_correctness",
    "repair_successes",
    "repair_attempts",
    "avg_prompt_tokens_estimate",
    "avg_model_latency_ms",
    "avg_sql_latency_ms",
    "avg_trace_steps",
    "summary_path",
]


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return ""


def num(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def load_summaries(input_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*/*/summary.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        item["summary_path"] = str(path)
        summaries.append(item)
    return summaries


def write_csv(summaries: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for item in summaries:
            writer.writerow({field: item.get(field, "") for field in FIELDS})


def write_markdown(summaries: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        "| Model | Config | N | Model OK | Exec | Correct | Repairs | Prompt tok | Model ms | SQL ms | Steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {model} | {config} | {n} | {model_ok} | {exec_ok} | {rc} | {repairs} | {tok} | {lat} | {sql} | {steps} |".format(
                model=item.get("model_name", ""),
                config=item.get("config", ""),
                n=item.get("total_queries", ""),
                model_ok=pct(item.get("model_success_rate", "")),
                exec_ok=pct(item.get("execution_accuracy", "")),
                rc=pct(item.get("result_correctness", "")),
                repairs=item.get("repair_successes", ""),
                tok=num(item.get("avg_prompt_tokens_estimate", ""), 0),
                lat=num(item.get("avg_model_latency_ms", ""), 0),
                sql=num(item.get("avg_sql_latency_ms", ""), 0),
                steps=num(item.get("avg_trace_steps", ""), 1),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_failures(input_dir: Path, max_per_run: int) -> str:
    sections: list[str] = []
    for results_path in sorted(input_dir.glob("*/*/results.jsonl")):
        rel = results_path.relative_to(input_dir)
        failures: list[dict[str, Any]] = []
        try:
            with results_path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not row.get("result_match"):
                        failures.append(row)
        except Exception:
            continue
        if not failures:
            continue
        sections.append(f"## {rel}")
        for row in failures[:max_per_run]:
            sections.append(
                "- `{qid}` {status}: {question}\n"
                "  - Error: `{error}`\n"
                "  - Pred SQL: `{sql}`".format(
                    qid=row.get("query_id", ""),
                    status="exec-failed" if not row.get("pred_executed") else "wrong-result",
                    question=row.get("natural_language", ""),
                    error=(row.get("pred_error") or row.get("model_error") or "")[:300],
                    sql=(row.get("predicted_sql") or "").replace("\n", " ")[:500],
                )
            )
        if len(failures) > max_per_run:
            sections.append(f"- ... {len(failures) - max_per_run} more failures omitted")
        sections.append("")
    return "\n".join(sections).strip() + "\n" if sections else "No failures found.\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="evaluation/results/cli_runs",
        help="Directory containing model/config run subdirectories.",
    )
    parser.add_argument("--max-failures-per-run", type=int, default=20)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    summaries = load_summaries(input_dir)
    if not summaries:
        print(f"No summary.json files found under {input_dir}")
        return 1

    csv_path = input_dir / "cli_runs_summary.csv"
    md_path = input_dir / "cli_runs_summary.md"
    failures_path = input_dir / "cli_runs_failures.md"

    write_csv(summaries, csv_path)
    write_markdown(summaries, md_path)
    failures_path.write_text(
        collect_failures(input_dir, args.max_failures_per_run),
        encoding="utf-8",
    )

    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
