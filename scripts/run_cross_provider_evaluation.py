#!/usr/bin/env python3
"""
scripts/run_cross_provider_evaluation.py
---------------------------------------------------------------------------

Run the OPTIMAL prompt configuration across all 9 cross-provider models on
every benchmark dataset, capture per-query results to JSONL, compute
aggregate metrics per (model, dataset), and write a results matrix.

Optimal config (per VLDB 2026 paper, Table 8):
    schema_format    = MARKDOWN
    schema_scope     = RELEVANT_SUBSET
    metadata_level   = DESCRIPTIONS
    example_strategy = DYNAMIC_FEW_SHOT
    prompt_version   = FULL

Resumable from checkpoint: if a JSONL output file already contains a row
for (model_key, dataset, query_id), the call is skipped on rerun.

Usage:
    python scripts/run_cross_provider_evaluation.py
    python scripts/run_cross_provider_evaluation.py --tier flagship
    python scripts/run_cross_provider_evaluation.py --models anthropic-opus-4-7
    python scripts/run_cross_provider_evaluation.py --datasets custom_analytics
    python scripts/run_cross_provider_evaluation.py --max-queries 5    # smoke run
    python scripts/run_cross_provider_evaluation.py --concurrency 8
    python scripts/run_cross_provider_evaluation.py --dry-run          # build prompts only

Per-query JSONL goes to:
    results/cross_provider/{model_key}/{dataset}.jsonl

Aggregate matrix goes to:
    results/cross_provider/aggregate.jsonl
    results/cross_provider/aggregate_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from framework.llm import (  # noqa: E402
    LLMResponse,
    get_caller,
    list_models,
    list_tiers,
    load_model_config,
)
from framework.prompt_builder import (  # noqa: E402
    ExampleStrategy,
    MetadataLevel,
    PromptBuilder,
    PromptVersion,
    SchemaFormat,
    SchemaScope,
)
# `framework.sql_executor` and `framework.result_comparator` are imported
# lazily inside main() so that --no-execute / --dry-run do not require
# clickhouse-driver to be installed.
if TYPE_CHECKING:
    from framework.sql_executor import SQLExecutor

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cross_provider")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTIMAL_CONFIG = dict(
    format=SchemaFormat.MARKDOWN,
    scope=SchemaScope.RELEVANT_SUBSET,
    metadata=MetadataLevel.DESCRIPTIONS,
    examples=ExampleStrategy.DYNAMIC_FEW_SHOT,
    prompt_version=PromptVersion.FULL,
)

DATASETS: dict[str, list[str]] = {
    "custom_analytics": [
        "simple_select", "aggregation", "window_functions",
        "time_series", "complex_joins", "clickhouse_specific",
    ],
    "clickbench": ["clickbench"],
    "ssb": ["ssb"],
}

DATASET_DATABASE: dict[str, str] = {
    "custom_analytics": "analytics",
    "clickbench": "clickbench",
    "ssb": "ssb",
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run optimal config across the 9-model cross-provider matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tier",
        choices=["flagship", "mid", "small", "all"],
        default="all",
        help="Which tier of models to evaluate.",
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys (overrides --tier).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS.keys()),
        help="Comma-separated dataset keys.",
    )
    p.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap queries per dataset (smoke runs).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent inference calls per model. Higher = faster, more RL pressure.",
    )
    p.add_argument(
        "--benchmark-dir",
        type=Path,
        default=REPO_ROOT / "benchmark",
        help="Path to benchmark directory.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "cross_provider",
        help="Output directory.",
    )
    p.add_argument(
        "--clickhouse-host",
        default=os.environ.get("CLICKHOUSE_HOST", "localhost"),
    )
    p.add_argument(
        "--clickhouse-port",
        type=int,
        default=int(os.environ.get("CLICKHOUSE_PORT", "9000")),
    )
    p.add_argument(
        "--no-execute",
        action="store_true",
        help="Skip SQL execution and result comparison; only generate SQL.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and report counts; do not call any model.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even queries already present in checkpoint JSONL.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers: loading
# ---------------------------------------------------------------------------

def load_queries(benchmark_dir: Path, dataset: str, max_queries: Optional[int]) -> list[dict]:
    """Load all queries for a dataset, deduplicate by id, optionally cap count.

    Accepts both shapes commonly used in this repo:
      - top-level JSON list of query dicts (custom_analytics files)
      - top-level JSON object with a "queries" key (clickbench, ssb)
    """
    files = DATASETS.get(dataset)
    if files is None:
        raise SystemExit(f"Unknown dataset '{dataset}'. Known: {list(DATASETS.keys())}")
    queries: list[dict] = []
    seen_ids: set[str] = set()
    for stem in files:
        path = benchmark_dir / "queries" / f"{stem}.json"
        if not path.exists():
            logger.warning("Query file missing: %s", path)
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("queries") or payload.get("data") or []
        if not isinstance(payload, list):
            logger.warning("Unexpected query file shape: %s", path)
            continue
        for q in payload:
            if not isinstance(q, dict):
                continue
            qid = q.get("id")
            if not qid or qid in seen_ids:
                continue
            seen_ids.add(qid)
            queries.append(q)
    if max_queries is not None:
        queries = queries[:max_queries]
    return queries


def select_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        keys = [k.strip() for k in args.models.split(",") if k.strip()]
        unknown = [k for k in keys if k not in list_models()]
        if unknown:
            raise SystemExit(f"Unknown model keys: {unknown}")
        return keys
    if args.tier == "all":
        return list_models()
    tiers = list_tiers()
    return tiers[args.tier]


def provider_key_present(provider: str) -> bool:
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "google":
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return False


def model_runnable(cfg: dict, model_key: str) -> tuple[bool, str]:
    """Return (is_runnable, reason). For CLI transport, instantiating
    the caller probes the binary; we surface the resulting EnvironmentError
    as the reason. For SDK transport, we check the env var presence.
    """
    entry = cfg["models"][model_key]
    transport = entry.get("transport") or cfg.get("defaults", {}).get("transport", "cli")
    provider = entry["provider"]
    if transport == "sdk":
        return (True, "ok") if provider_key_present(provider) \
            else (False, f"missing {provider} API key in environment")
    # CLI transport: try to construct the caller (it probes the binary)
    try:
        from framework.llm import get_caller
        get_caller(model_key)
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"CLI unavailable: {type(e).__name__}: {str(e)[:200]}"


def load_checkpoint(out_path: Path) -> set[str]:
    """Return the set of query_ids already completed in `out_path` JSONL."""
    if not out_path.exists():
        return set()
    completed: set[str] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("query_id")
            if qid:
                completed.add(qid)
    return completed


# ---------------------------------------------------------------------------
# Per-query worker
# ---------------------------------------------------------------------------

def evaluate_one(
    *,
    model_key: str,
    query: dict,
    dataset: str,
    prompt_builder: PromptBuilder,
    executor: "Optional[SQLExecutor]",
    no_execute: bool,
    dry_run: bool,
) -> dict:
    """Execute one query end-to-end and return a JSONL-serializable record."""
    qid = query["id"]
    nl = query["natural_language"]
    gold_sql = query["sql"]
    relevant_tables = query.get("tables_used")
    relevant_columns = query.get("columns_used")
    database = DATASET_DATABASE.get(dataset, "default")

    record: dict = {
        "query_id": qid,
        "dataset": dataset,
        "model_key": model_key,
        "natural_language": nl,
        "gold_sql": gold_sql,
        "category": query.get("category"),
        "difficulty": query.get("difficulty"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Build the prompt with the optimal config.
    try:
        prompt = prompt_builder.build_prompt(
            question=nl,
            dataset=dataset,
            format=OPTIMAL_CONFIG["format"],
            scope=OPTIMAL_CONFIG["scope"],
            metadata=OPTIMAL_CONFIG["metadata"],
            examples=OPTIMAL_CONFIG["examples"],
            relevant_tables=relevant_tables,
            relevant_columns=relevant_columns,
            prompt_version=OPTIMAL_CONFIG["prompt_version"],
        )
    except Exception as e:  # noqa: BLE001
        record.update({"prompt_error": f"{type(e).__name__}: {e}"})
        return record

    record["token_estimate"] = prompt.token_estimate
    record["num_tables"] = prompt.num_tables
    record["num_columns"] = prompt.num_columns
    record["num_examples"] = prompt.num_examples

    if dry_run:
        record["dry_run"] = True
        return record

    # Inference call
    caller = get_caller(model_key)
    response: LLMResponse = caller.call(prompt.user_message, system=prompt.system_message)
    record.update({
        "predicted_sql": response.sql,
        "raw_response_excerpt": (response.raw_response or "")[:1500],
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_ms": response.latency_ms,
        "model_id": response.model,
        "provider": response.provider,
        "api_success": response.success,
        "api_error": response.error,
    })

    if not response.success:
        return record

    if no_execute or executor is None:
        return record

    # Execute predicted vs gold against ClickHouse
    try:
        from framework.result_comparator import compare_results  # lazy
        pred_result = executor.execute(response.sql, database=database)
        gold_result = executor.execute(gold_sql, database=database)
        record["predicted_execution_success"] = pred_result.success
        record["predicted_execution_error"] = pred_result.error
        record["predicted_row_count"] = pred_result.row_count
        record["gold_execution_success"] = gold_result.success
        record["gold_execution_error"] = gold_result.error
        record["gold_row_count"] = gold_result.row_count

        if pred_result.success and gold_result.success:
            cmp = compare_results(
                predicted_rows=pred_result.results,
                predicted_columns=pred_result.columns,
                gold_rows=gold_result.results,
                gold_columns=gold_result.columns,
            )
            record["result_correct"] = bool(cmp.get("match", False))
            record["result_comparison"] = {
                k: cmp[k] for k in cmp if k not in ("predicted_rows_sample", "gold_rows_sample")
            }
        else:
            record["result_correct"] = False
    except Exception as e:  # noqa: BLE001
        record["execution_error"] = f"{type(e).__name__}: {e}"
        record["execution_traceback"] = traceback.format_exc()

    return record


# ---------------------------------------------------------------------------
# Per-(model, dataset) driver
# ---------------------------------------------------------------------------

def run_model_dataset(
    *,
    model_key: str,
    dataset: str,
    queries: list[dict],
    prompt_builder: PromptBuilder,
    executor: "Optional[SQLExecutor]",
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """Run all queries for one (model, dataset) pair."""
    out_path = out_dir / model_key / f"{dataset}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.force else load_checkpoint(out_path)

    pending = [q for q in queries if q["id"] not in completed]
    print(f"  [{model_key} / {dataset}] {len(pending)} pending "
          f"({len(completed)} already complete) -> {out_path}")

    if not pending:
        return _summarize(model_key, dataset, out_path)

    results: list[dict] = []
    if args.dry_run or executor is None or args.concurrency <= 1:
        for q in pending:
            r = evaluate_one(
                model_key=model_key, query=q, dataset=dataset,
                prompt_builder=prompt_builder, executor=executor,
                no_execute=args.no_execute, dry_run=args.dry_run,
            )
            results.append(r)
    else:
        # Concurrent inference; SQL execution runs after each future completes
        # (executor is single-connection; we serialize execute() via a lock).
        from threading import Lock
        exec_lock = Lock()

        def _do(q):
            r = evaluate_one(
                model_key=model_key, query=q, dataset=dataset,
                prompt_builder=prompt_builder, executor=None,  # defer execute
                no_execute=True, dry_run=False,
            )
            return r, q

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_do, q) for q in pending]
            for fut in as_completed(futures):
                try:
                    r, q = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.error("Worker exception: %s", e)
                    continue
                # Now execute predicted/gold serially (single ClickHouse client)
                if r.get("api_success") and not args.no_execute and executor is not None:
                    with exec_lock:
                        _execute_into(r, q, dataset, executor)
                results.append(r)

    # Append-only write
    with out_path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=_json_default) + "\n")

    return _summarize(model_key, dataset, out_path)


def _execute_into(record: dict, query: dict, dataset: str, executor: "SQLExecutor") -> None:
    """Run predicted + gold SQL and merge results into `record` in-place."""
    from framework.result_comparator import compare_results  # lazy
    database = DATASET_DATABASE.get(dataset, "default")
    pred_sql = record.get("predicted_sql") or ""
    gold_sql = query["sql"]
    try:
        pred_result = executor.execute(pred_sql, database=database)
        gold_result = executor.execute(gold_sql, database=database)
        record["predicted_execution_success"] = pred_result.success
        record["predicted_execution_error"] = pred_result.error
        record["predicted_row_count"] = pred_result.row_count
        record["gold_execution_success"] = gold_result.success
        record["gold_execution_error"] = gold_result.error
        record["gold_row_count"] = gold_result.row_count
        if pred_result.success and gold_result.success:
            cmp = compare_results(
                predicted_rows=pred_result.results,
                predicted_columns=pred_result.columns,
                gold_rows=gold_result.results,
                gold_columns=gold_result.columns,
            )
            record["result_correct"] = bool(cmp.get("match", False))
            record["result_comparison"] = {
                k: cmp[k] for k in cmp if k not in ("predicted_rows_sample", "gold_rows_sample")
            }
        else:
            record["result_correct"] = False
    except Exception as e:  # noqa: BLE001
        record["execution_error"] = f"{type(e).__name__}: {e}"


def _summarize(model_key: str, dataset: str, out_path: Path) -> dict:
    """Compute aggregate metrics for a single JSONL file."""
    if not out_path.exists():
        return {"model_key": model_key, "dataset": dataset, "n": 0}
    n = ex_pass = rc_pass = api_fail = 0
    in_tok = out_tok = lat = 0.0
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            if not r.get("api_success", False):
                api_fail += 1
                continue
            if r.get("predicted_execution_success"):
                ex_pass += 1
            if r.get("result_correct"):
                rc_pass += 1
            in_tok += float(r.get("input_tokens", 0) or 0)
            out_tok += float(r.get("output_tokens", 0) or 0)
            lat += float(r.get("latency_ms", 0) or 0)
    return {
        "model_key": model_key,
        "dataset": dataset,
        "n": n,
        "api_failures": api_fail,
        "ex_count": ex_pass,
        "rc_count": rc_pass,
        "ex": (ex_pass / n) if n else 0.0,
        "rc": (rc_pass / n) if n else 0.0,
        "avg_input_tokens": (in_tok / n) if n else 0.0,
        "avg_output_tokens": (out_tok / n) if n else 0.0,
        "avg_latency_ms": (lat / n) if n else 0.0,
    }


def _json_default(o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if hasattr(o, "value") and hasattr(o, "name"):  # Enum
        return o.value
    return str(o)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    cfg = load_model_config()
    model_keys = select_models(args)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # Filter to runnable models (CLI binary present, or SDK key set).
    # In --dry-run we skip the runnability check.
    runnable: list[str] = []
    for mk in model_keys:
        if args.dry_run:
            runnable.append(mk)
            continue
        ok, reason = model_runnable(cfg, mk)
        if ok:
            runnable.append(mk)
        else:
            print(f"  SKIP {mk}: {reason}")
    if not runnable and not args.dry_run:
        print("No models runnable. Run scripts/doctor_cli.py to diagnose.")
        return 1

    print(f"Running {len(runnable)} model(s) on {len(datasets)} dataset(s)")
    print(f"  Models:   {runnable}")
    print(f"  Datasets: {datasets}")
    print(f"  Optimal config: format=MARKDOWN, scope=RELEVANT_SUBSET, "
          f"metadata=DESCRIPTIONS, examples=DYNAMIC_FEW_SHOT, prompt=FULL")
    print(f"  Output:   {args.out_dir}")
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    prompt_builder = PromptBuilder(benchmark_dir=str(args.benchmark_dir))

    executor: "Optional[SQLExecutor]" = None
    if not args.no_execute and not args.dry_run:
        try:
            from framework.sql_executor import SQLExecutor  # lazy
        except ImportError as e:
            print(f"  WARNING: clickhouse-driver not installed ({e}). "
                  f"Install with `pip install clickhouse-driver` or pass --no-execute.")
            executor = None
        else:
            executor = SQLExecutor(host=args.clickhouse_host, port=args.clickhouse_port)
            if not executor.test_connection():
                print(f"  WARNING: cannot connect to ClickHouse at "
                      f"{args.clickhouse_host}:{args.clickhouse_port}. "
                      f"Continuing in --no-execute mode.")
                executor = None

    summaries: list[dict] = []
    t0 = time.perf_counter()

    for model_key in runnable:
        for dataset in datasets:
            queries = load_queries(args.benchmark_dir, dataset, args.max_queries)
            print(f"Model={model_key} dataset={dataset} queries={len(queries)}")
            try:
                summary = run_model_dataset(
                    model_key=model_key,
                    dataset=dataset,
                    queries=queries,
                    prompt_builder=prompt_builder,
                    executor=executor,
                    out_dir=args.out_dir,
                    args=args,
                )
                summaries.append(summary)
                print(f"    -> n={summary['n']} ex={summary['ex']:.3f} rc={summary['rc']:.3f}")
            except Exception as e:  # noqa: BLE001
                logger.error("Model %s dataset %s failed: %s", model_key, dataset, e)
                logger.error(traceback.format_exc())

    elapsed = time.perf_counter() - t0
    print(f"\nWall time: {elapsed:.1f}s")

    # Write aggregate
    agg_path = args.out_dir / "aggregate.jsonl"
    with agg_path.open("w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s) + "\n")

    matrix_path = args.out_dir / "aggregate_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_key", "dataset", "n", "api_failures",
                        "ex_count", "rc_count", "ex", "rc",
                        "avg_input_tokens", "avg_output_tokens", "avg_latency_ms"],
        )
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    print(f"\nAggregate JSONL: {agg_path}")
    print(f"Aggregate CSV:   {matrix_path}")

    # Pretty pivot table
    print("\nResult Correctness matrix (RC %, n):")
    by_model: dict[str, dict[str, str]] = {}
    for s in summaries:
        by_model.setdefault(s["model_key"], {})[s["dataset"]] = (
            f"{100*s['rc']:5.1f} (n={s['n']})"
        )
    if by_model:
        ds_cols = sorted({s["dataset"] for s in summaries})
        col_w = max([len(c) for c in ds_cols] + [14])
        mk_w = max(len(k) for k in by_model)
        header = f"{'model':{mk_w}}  " + "  ".join(f"{d:>{col_w}}" for d in ds_cols)
        print(header)
        print("-" * len(header))
        for mk in sorted(by_model.keys()):
            cells = "  ".join(f"{by_model[mk].get(d, '-'):>{col_w}}" for d in ds_cols)
            print(f"{mk:{mk_w}}  {cells}")

    if executor is not None:
        executor.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
