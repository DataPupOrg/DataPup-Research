#!/usr/bin/env python3
"""Generate extra evidence artifacts for the DASHSys strong-accept revision."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def boolish(value: Any) -> bool:
    return str(value).lower() == "true"


def classify_failure(row: dict[str, str]) -> str:
    if boolish(row["result_match"]):
        return "Correct"
    if not boolish(row["final_executed"]):
        if not row.get("final_sql", "").strip() or not boolish(row.get("model_success", "")):
            return "Timeout / empty output"
        return "Residual execution error"

    details = row.get("comparison_details", "")
    gold_sql = row.get("gold_sql", "").lower()
    pred_sql = row.get("final_sql", "").lower()

    if "column count mismatch" in details.lower() or "projected gold" in details.lower():
        return "Projection mismatch"
    if "row-count mismatch" in details.lower():
        return "Wrong granularity or filter"
    if any(tok in gold_sql for tok in ["countif", " filter ", " where "]) and " where " not in pred_sql and " filter " not in pred_sql:
        return "Missing filter assumption"
    if "group by" in gold_sql and "group by" in pred_sql and gold_sql.split("group by", 1)[1].split("order by", 1)[0] != pred_sql.split("group by", 1)[1].split("order by", 1)[0]:
        return "Grouping-key mismatch"
    if "join" in gold_sql and "join" not in pred_sql:
        return "Join-path mismatch"
    if "properties[" in gold_sql and "properties[" not in pred_sql:
        return "JSON/property mismatch"
    if "large equal-sized result set" in details.lower():
        return "Wrong values or ranking"
    return "Other wrong result"


def assumption_tags(row: dict[str, str]) -> set[str]:
    if boolish(row["result_match"]) or not boolish(row["final_executed"]):
        return set()
    nl = row.get("natural_language", "").lower()
    gold = row.get("gold_sql", "").lower()
    pred = row.get("final_sql", "").lower()
    tags: set[str] = set()
    if any(tok in nl + gold for tok in ["month", "week", "year", "day", "last", "past", "date", "time", "cohort"]):
        if any(tok in gold for tok in ["tostartof", "toyear", "tomonth", "date", "timestamp", "start_time"]):
            tags.add("time window / bucket")
    if "group by" in gold:
        tags.add("grouping key")
    if any(tok in gold for tok in ["countif", "sumif", "where", "having"]):
        tags.add("filter / predicate")
    if "order by" in gold or "top" in nl or "rank" in nl:
        tags.add("ranking / tie-break")
    if "join" in gold:
        tags.add("join path")
    if "properties[" in gold or "json_extract" in gold or "mapcontains" in gold:
        tags.add("JSON property")
    if "over (" in gold:
        tags.add("window frame")
    if "count" in gold or "sum" in gold or "avg" in gold or "quantile" in gold:
        tags.add("metric definition")
    if pred and gold and pred != gold and not tags:
        tags.add("result shape")
    return tags


def analyze_system_upgrade() -> dict[str, Any]:
    current = read_json(ROOT / "evaluation/results/phase1/json_full_none_zero_shot__claude-3-5-sonnet-20241022.json")
    revised = read_json(ROOT / "evaluation/results/phase2/markdown_relevant_subset_descriptions_dynamic_few_shot__claude-3-5-sonnet-20241022.json")
    current_rows = current["query_results"]
    revised_rows = revised["query_results"]
    current_by_id = {row["query_id"]: bool(row["result_match"]) for row in current_rows}
    revised_by_id = {row["query_id"]: bool(row["result_match"]) for row in revised_rows}
    ids = [qid for qid in current_by_id if qid in revised_by_id]
    current_only = sum(current_by_id[qid] and not revised_by_id[qid] for qid in ids)
    revised_only = sum((not current_by_id[qid]) and revised_by_id[qid] for qid in ids)
    discordant = current_only + revised_only
    tail = min(current_only, revised_only)
    mcnemar_exact_p = (
        min(1.0, 2 * sum(math.comb(discordant, i) for i in range(tail + 1)) / (2 ** discordant))
        if discordant
        else 1.0
    )
    return {
        "current_datapup_style": {
            "label": "Full-schema JSON zero-shot",
            "execution_accuracy": current["execution_accuracy"],
            "result_correctness": current["result_correctness"],
            "successful_queries": current["successful_queries"],
            "correct_queries": current["correct_queries"],
            "total_queries": current["total_queries"],
        },
        "revised_datapup_style": {
            "label": "Markdown relevant-subset descriptions dynamic few-shot",
            "execution_accuracy": revised["execution_accuracy"],
            "result_correctness": revised["result_correctness"],
            "successful_queries": revised["successful_queries"],
            "correct_queries": revised["correct_queries"],
            "total_queries": revised["total_queries"],
        },
        "delta_execution_accuracy_pp": round((revised["execution_accuracy"] - current["execution_accuracy"]) * 100, 1),
        "delta_result_correctness_pp": round((revised["result_correctness"] - current["result_correctness"]) * 100, 1),
        "mcnemar_current_only": current_only,
        "mcnemar_revised_only": revised_only,
        "mcnemar_exact_p": mcnemar_exact_p,
    }


def analyze_semantic_assumptions() -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for path in sorted((ROOT / "evaluation/results/cli_runs_repair_existing").glob("*__best.scored.csv")):
        model = path.name.split("__", 1)[0]
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["model"] = model
                rows.append(row)

    failure_counts = Counter(classify_failure(row) for row in rows)
    tag_counts: Counter[str] = Counter()
    tag_by_model: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for tag in assumption_tags(row):
            tag_counts[tag] += 1
            tag_by_model[row["model"]][tag] += 1

    wrong_executed = [
        row for row in rows
        if boolish(row["final_executed"]) and not boolish(row["result_match"])
    ]
    examples = []
    for row in wrong_executed[:8]:
        examples.append(
            {
                "model": row["model"],
                "query_id": row["query_id"],
                "category": row["category"],
                "natural_language": row["natural_language"],
                "assumption_tags": sorted(assumption_tags(row)),
                "comparison_details": row["comparison_details"],
            }
        )

    return {
        "total_best_cli_queries": len(rows),
        "correct": failure_counts["Correct"],
        "wrong_executed": len(wrong_executed),
        "failure_counts": dict(failure_counts),
        "assumption_tag_counts": dict(tag_counts),
        "assumption_tag_counts_by_model": {k: dict(v) for k, v in tag_by_model.items()},
        "examples": examples,
    }


def analyze_duckdb_if_present() -> dict[str, Any] | None:
    base = ROOT / "evaluation/results/duckdb_cli_validation/claude_cli"
    baseline = base / "baseline/summary.json"
    best = base / "best/summary.json"
    if not baseline.exists() or not best.exists():
        return None
    b = read_json(baseline)
    r = read_json(best)
    return {
        "portable_gold_queries": b["portable_gold_queries"],
        "baseline": b,
        "best": r,
        "delta_result_correctness_pp": round((r["result_correctness"] - b["result_correctness"]) * 100, 1),
        "delta_execution_rate_pp": round((r["execution_rate"] - b["execution_rate"]) * 100, 1),
    }


def write_markdown(data: dict[str, Any], path: Path) -> None:
    upgrade = data["system_upgrade"]
    sem = data["semantic_assumptions"]
    duckdb = data.get("duckdb_validation")
    lines = [
        "# Strong-Accept Evidence Artifacts",
        "",
        "## Current DataPup vs Revised Prompt",
        "",
        "| Configuration | EX | RC | Correct |",
        "|---|---:|---:|---:|",
        f"| {upgrade['current_datapup_style']['label']} | {pct(upgrade['current_datapup_style']['execution_accuracy'])} | {pct(upgrade['current_datapup_style']['result_correctness'])} | {upgrade['current_datapup_style']['correct_queries']}/{upgrade['current_datapup_style']['total_queries']} |",
        f"| {upgrade['revised_datapup_style']['label']} | {pct(upgrade['revised_datapup_style']['execution_accuracy'])} | {pct(upgrade['revised_datapup_style']['result_correctness'])} | {upgrade['revised_datapup_style']['correct_queries']}/{upgrade['revised_datapup_style']['total_queries']} |",
        "",
        f"Delta: +{upgrade['delta_execution_accuracy_pp']:.1f}pp EX, +{upgrade['delta_result_correctness_pp']:.1f}pp RC; McNemar exact p={upgrade['mcnemar_exact_p']:.2e} ({upgrade['mcnemar_revised_only']} revised-only correct vs. {upgrade['mcnemar_current_only']} current-only correct).",
        "",
        "## Semantic Assumption Audit",
        "",
        "| Failure/Outcome | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(sem["failure_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "| Assumption to Surface | Wrong-executable Count |", "|---|---:|"])
    for key, value in sorted(sem["assumption_tag_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {value} |")
    if duckdb:
        lines.extend(
            [
                "",
                "## DuckDB CLI Validation",
                "",
                "| Config | EX | RC | Correct set |",
                "|---|---:|---:|---:|",
                f"| Baseline | {pct(duckdb['baseline']['execution_rate'])} | {pct(duckdb['baseline']['result_correctness'])} | {duckdb['baseline']['total']} portable queries |",
                f"| Best | {pct(duckdb['best']['execution_rate'])} | {pct(duckdb['best']['result_correctness'])} | {duckdb['best']['total']} portable queries |",
                "",
                f"Delta: {duckdb['delta_execution_rate_pp']:+.1f}pp EX, {duckdb['delta_result_correctness_pp']:+.1f}pp RC.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = {
        "system_upgrade": analyze_system_upgrade(),
        "semantic_assumptions": analyze_semantic_assumptions(),
        "duckdb_validation": analyze_duckdb_if_present(),
    }
    out_json = ROOT / "evaluation/results/strong_accept_evidence.json"
    out_md = ROOT / "evaluation/results/strong_accept_evidence.md"
    out_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    write_markdown(data, out_md)
    print(out_json)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
