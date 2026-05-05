#!/usr/bin/env python3
"""Create a local DuckDB copy of the custom analytics benchmark."""

from __future__ import annotations

import argparse
import json
import random
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
from faker import Faker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "evaluation/benchmark/schemas/custom_analytics"))

import generate_data as gen  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "evaluation/duckdb/datapup.duckdb")
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_value(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def numeric(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    con.execute("DROP TABLE IF EXISTS analytics.events")
    con.execute("DROP TABLE IF EXISTS analytics.sessions")
    con.execute("DROP TABLE IF EXISTS analytics.users")
    con.execute("DROP TABLE IF EXISTS analytics.products")

    con.execute(
        """
        CREATE TABLE analytics.products (
            product_id BIGINT,
            name VARCHAR,
            category VARCHAR,
            subcategory VARCHAR,
            price DOUBLE,
            tags JSON,
            created_at TIMESTAMP,
            is_active INTEGER,
            rating DOUBLE,
            review_count INTEGER
        )
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.users (
            user_id BIGINT,
            email VARCHAR,
            name VARCHAR,
            signup_date DATE,
            plan VARCHAR,
            country VARCHAR,
            tags JSON,
            lifetime_value DOUBLE,
            last_active TIMESTAMP,
            preferences JSON
        )
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.sessions (
            session_id VARCHAR,
            user_id BIGINT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            duration_seconds INTEGER,
            page_count INTEGER,
            device_type VARCHAR,
            browser VARCHAR,
            os VARCHAR,
            country VARCHAR,
            entry_page VARCHAR,
            exit_page VARCHAR,
            utm_source VARCHAR,
            utm_medium VARCHAR,
            utm_campaign VARCHAR,
            is_converted INTEGER
        )
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.events (
            event_id VARCHAR,
            session_id VARCHAR,
            user_id BIGINT,
            event_type VARCHAR,
            page_url VARCHAR,
            referrer VARCHAR,
            device_type VARCHAR,
            browser VARCHAR,
            os VARCHAR,
            country VARCHAR,
            city VARCHAR,
            properties JSON,
            timestamp TIMESTAMP,
            duration_ms INTEGER,
            is_bounce INTEGER
        )
        """
    )


def insert_rows(
    con: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    json_columns: set[str],
) -> None:
    placeholders = ", ".join(["?"] * len(columns))
    col_sql = ", ".join(columns)
    values = []
    for row in rows:
        values.append(
            [
                json_value(row[col]) if col in json_columns else numeric(row[col])
                for col in columns
            ]
        )
    con.executemany(f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})", values)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    Faker.seed(args.seed)

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.db_path.exists():
        args.db_path.unlink()

    n_products = max(1, int(gen.BASE_PRODUCTS * args.scale))
    n_users = max(1, int(gen.BASE_USERS * args.scale))
    n_sessions = max(1, int(gen.BASE_SESSIONS * args.scale))
    n_events = max(1, int(gen.BASE_EVENTS * args.scale))

    products = gen.generate_products(n_products)
    users = gen.generate_users(n_users)
    sessions = gen.generate_sessions(n_sessions, [u["user_id"] for u in users])
    events = gen.generate_events(n_events, sessions, [p["product_id"] for p in products])

    con = duckdb.connect(str(args.db_path))
    create_schema(con)
    insert_rows(
        con,
        "analytics.products",
        ["product_id", "name", "category", "subcategory", "price", "tags", "created_at", "is_active", "rating", "review_count"],
        products,
        {"tags"},
    )
    insert_rows(
        con,
        "analytics.users",
        ["user_id", "email", "name", "signup_date", "plan", "country", "tags", "lifetime_value", "last_active", "preferences"],
        users,
        {"tags", "preferences"},
    )
    insert_rows(
        con,
        "analytics.sessions",
        ["session_id", "user_id", "start_time", "end_time", "duration_seconds", "page_count", "device_type", "browser", "os", "country", "entry_page", "exit_page", "utm_source", "utm_medium", "utm_campaign", "is_converted"],
        sessions,
        set(),
    )
    insert_rows(
        con,
        "analytics.events",
        ["event_id", "session_id", "user_id", "event_type", "page_url", "referrer", "device_type", "browser", "os", "country", "city", "properties", "timestamp", "duration_ms", "is_bounce"],
        events,
        {"properties"},
    )

    counts = {
        "analytics.products": con.execute("SELECT count(*) FROM analytics.products").fetchone()[0],
        "analytics.users": con.execute("SELECT count(*) FROM analytics.users").fetchone()[0],
        "analytics.sessions": con.execute("SELECT count(*) FROM analytics.sessions").fetchone()[0],
        "analytics.events": con.execute("SELECT count(*) FROM analytics.events").fetchone()[0],
    }
    summary = {
        "db_path": str(args.db_path),
        "scale": args.scale,
        "seed": args.seed,
        "row_counts": counts,
    }
    out = PROJECT_ROOT / "evaluation/results/duckdb_setup_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
