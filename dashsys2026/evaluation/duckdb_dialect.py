#!/usr/bin/env python3
"""DuckDB helpers for focused cross-dialect DataPup validation."""

from __future__ import annotations

import re


def split_args(text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(text[start:i].strip())
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        args.append(tail)
    return args


def find_matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    quote: str | None = None
    i = open_idx
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def replace_function_calls(sql: str, name: str, converter) -> str:
    pattern = re.compile(rf"\b{re.escape(name)}\s*\(", re.IGNORECASE)
    pos = 0
    out: list[str] = []
    while True:
        match = pattern.search(sql, pos)
        if not match:
            out.append(sql[pos:])
            break
        open_idx = match.end() - 1
        close_idx = find_matching_paren(sql, open_idx)
        if close_idx < 0:
            out.append(sql[pos:])
            break
        out.append(sql[pos : match.start()])
        inner = sql[open_idx + 1 : close_idx]
        args = [translate_clickhouse_to_duckdb(a) for a in split_args(inner)]
        out.append(converter(args))
        pos = close_idx + 1
    return "".join(out)


def replace_json_access(sql: str) -> str:
    def repl(match: re.Match[str]) -> str:
        prefix = match.group("prefix") or ""
        col = match.group("col")
        key = match.group("key")
        return f"json_extract_string({prefix}{col}, '$.{key}')"

    return re.sub(
        r"(?P<prefix>\b\w+\.)?(?P<col>properties|preferences)\s*\[\s*'(?P<key>[^']+)'\s*\]",
        repl,
        sql,
        flags=re.IGNORECASE,
    )


def translate_clickhouse_to_duckdb(sql: str) -> str:
    """Best-effort translation for the portable custom_analytics query subset."""
    s = sql.strip().rstrip(";")
    s = replace_json_access(s)

    # ClickHouse aggregate/function syntax with a second parenthesized argument.
    s = re.sub(
        r"\bquantile(?:Exact)?\s*\(\s*([0-9.]+)\s*\)\s*\(\s*([^)]+)\s*\)",
        r"quantile_cont(\2, \1)",
        s,
        flags=re.IGNORECASE,
    )

    converters = {
        "uniqExactIf": lambda a: f"count(DISTINCT {a[0]}) FILTER (WHERE {a[1]})" if len(a) >= 2 else f"count(DISTINCT {', '.join(a)})",
        "uniqExact": lambda a: f"count(DISTINCT {a[0]})",
        "countIf": lambda a: f"count(*) FILTER (WHERE {a[0]})",
        "sumIf": lambda a: f"sum({a[0]}) FILTER (WHERE {a[1]})" if len(a) >= 2 else f"sum({', '.join(a)})",
        "avgIf": lambda a: f"avg({a[0]}) FILTER (WHERE {a[1]})" if len(a) >= 2 else f"avg({', '.join(a)})",
        "toFloat64OrZero": lambda a: f"COALESCE(TRY_CAST({a[0]} AS DOUBLE), 0)",
        "toUInt64OrZero": lambda a: f"COALESCE(TRY_CAST({a[0]} AS UBIGINT), 0)",
        "toFloat64": lambda a: f"CAST({a[0]} AS DOUBLE)",
        "toUInt64": lambda a: f"CAST({a[0]} AS UBIGINT)",
        "toDecimal64": lambda a: f"CAST({a[0]} AS DECIMAL(18, {a[1] if len(a) > 1 else '2'}))",
        "toStartOfMonth": lambda a: f"date_trunc('month', {a[0]})",
        "toStartOfWeek": lambda a: f"date_trunc('week', {a[0]})",
        "toStartOfDay": lambda a: f"date_trunc('day', {a[0]})",
        "toDate": lambda a: f"CAST({a[0]} AS DATE)",
        "toDateTime": lambda a: f"CAST({a[0]} AS TIMESTAMP)",
        "toYear": lambda a: f"EXTRACT(year FROM {a[0]})",
        "toMonth": lambda a: f"EXTRACT(month FROM {a[0]})",
        "toHour": lambda a: f"EXTRACT(hour FROM {a[0]})",
        "toDayOfWeek": lambda a: f"EXTRACT(isodow FROM {a[0]})",
        "dateDiff": lambda a: f"date_diff({a[0]}, {a[1]}, {a[2]})" if len(a) >= 3 else f"date_diff({', '.join(a)})",
        "argMax": lambda a: f"arg_max({a[0]}, {a[1]})" if len(a) >= 2 else f"arg_max({', '.join(a)})",
        "argMin": lambda a: f"arg_min({a[0]}, {a[1]})" if len(a) >= 2 else f"arg_min({', '.join(a)})",
        "lagInFrame": lambda a: f"lag({', '.join(a)})",
        "leadInFrame": lambda a: f"lead({', '.join(a)})",
        "stddevPop": lambda a: f"stddev_pop({', '.join(a)})",
        "mapContains": lambda a: f"json_extract_string({a[0]}, '$.{a[1].strip(chr(39)).strip(chr(34))}') IS NOT NULL" if len(a) >= 2 else "FALSE",
        "has": lambda a: f"json_contains({a[0]}, json_quote({a[1]}))" if len(a) >= 2 else "FALSE",
        "if": lambda a: f"CASE WHEN {a[0]} THEN {a[1]} ELSE {a[2]} END" if len(a) >= 3 else f"if({', '.join(a)})",
        "multiIf": lambda a: multiif_to_case(a),
    }
    for name in sorted(converters, key=len, reverse=True):
        s = replace_function_calls(s, name, converters[name])

    s = re.sub(r"\bcount\s*\(\s*\)", "count(*)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bnow\s*\(\s*\)", "CURRENT_TIMESTAMP", s, flags=re.IGNORECASE)
    s = re.sub(r"\btoday\s*\(\s*\)", "CURRENT_DATE", s, flags=re.IGNORECASE)
    return s


def multiif_to_case(args: list[str]) -> str:
    if len(args) < 3:
        return f"multiIf({', '.join(args)})"
    parts = ["CASE"]
    for i in range(0, len(args) - 1, 2):
        if i + 1 >= len(args) - 1:
            break
        parts.append(f"WHEN {args[i]} THEN {args[i + 1]}")
    parts.append(f"ELSE {args[-1]} END")
    return " ".join(parts)


DUCKDB_MARKDOWN_SCHEMA = """# DuckDB analytics schema

Schema name: `analytics`

## analytics.events
| Column | Type | Description |
|---|---|---|
| event_id | VARCHAR | Event UUID as text |
| session_id | VARCHAR | Session identifier |
| user_id | BIGINT NULL | Registered user id when available |
| event_type | VARCHAR | One of page_view, click, purchase, signup, logout |
| page_url | VARCHAR | URL associated with the event |
| referrer | VARCHAR | Referrer URL, empty for direct traffic |
| device_type | VARCHAR | desktop, mobile, or tablet |
| browser | VARCHAR | Browser family |
| os | VARCHAR | Operating system |
| country | VARCHAR | Two-letter country code |
| city | VARCHAR | City when available |
| properties | JSON | Event attributes; use json_extract_string(properties, '$.key') |
| timestamp | TIMESTAMP | Event timestamp |
| duration_ms | INTEGER | Event duration in milliseconds |
| is_bounce | INTEGER | 1 for bounce events, else 0 |

## analytics.users
| Column | Type | Description |
|---|---|---|
| user_id | BIGINT | User identifier |
| email | VARCHAR | Email address |
| name | VARCHAR | User name |
| signup_date | DATE | Signup date |
| plan | VARCHAR | free, starter, pro, enterprise |
| country | VARCHAR | Two-letter country code |
| tags | JSON | User tags array |
| lifetime_value | DOUBLE | Monetary lifetime value |
| last_active | TIMESTAMP | Last activity timestamp |
| preferences | JSON | User preferences; use json_extract_string(preferences, '$.key') |

## analytics.sessions
| Column | Type | Description |
|---|---|---|
| session_id | VARCHAR | Session identifier |
| user_id | BIGINT NULL | Registered user id when available |
| start_time | TIMESTAMP | Session start |
| end_time | TIMESTAMP NULL | Session end |
| duration_seconds | INTEGER | Session duration |
| page_count | INTEGER | Number of pages in session |
| device_type | VARCHAR | desktop, mobile, or tablet |
| browser | VARCHAR | Browser family |
| os | VARCHAR | Operating system |
| country | VARCHAR | Two-letter country code |
| entry_page | VARCHAR | First page |
| exit_page | VARCHAR | Last page |
| utm_source | VARCHAR NULL | UTM source |
| utm_medium | VARCHAR NULL | UTM medium |
| utm_campaign | VARCHAR NULL | UTM campaign |
| is_converted | INTEGER | 1 for converted sessions, else 0 |

## analytics.products
| Column | Type | Description |
|---|---|---|
| product_id | BIGINT | Product identifier |
| name | VARCHAR | Product name |
| category | VARCHAR | Product category |
| subcategory | VARCHAR | Product subcategory |
| price | DOUBLE | Product price |
| tags | JSON | Product tags array |
| created_at | TIMESTAMP | Creation timestamp |
| is_active | INTEGER | 1 if active |
| rating | DOUBLE | Average rating |
| review_count | INTEGER | Number of reviews |
"""


DUCKDB_JSON_SCHEMA = """{
  "database": "analytics",
  "dialect": "DuckDB",
  "tables": {
    "analytics.events": ["event_id", "session_id", "user_id", "event_type", "page_url", "referrer", "device_type", "browser", "os", "country", "city", "properties", "timestamp", "duration_ms", "is_bounce"],
    "analytics.users": ["user_id", "email", "name", "signup_date", "plan", "country", "tags", "lifetime_value", "last_active", "preferences"],
    "analytics.sessions": ["session_id", "user_id", "start_time", "end_time", "duration_seconds", "page_count", "device_type", "browser", "os", "country", "entry_page", "exit_page", "utm_source", "utm_medium", "utm_campaign", "is_converted"],
    "analytics.products": ["product_id", "name", "category", "subcategory", "price", "tags", "created_at", "is_active", "rating", "review_count"]
  }
}"""


def schema_for_tables(tables: list[str] | None, schema_format: str = "markdown") -> str:
    if not tables:
        return DUCKDB_MARKDOWN_SCHEMA if schema_format == "markdown" else DUCKDB_JSON_SCHEMA
    full = DUCKDB_MARKDOWN_SCHEMA
    chunks = full.split("\n## ")
    selected = [chunks[0].rstrip()]
    wanted = {t.split(".")[-1] for t in tables}
    for chunk in chunks[1:]:
        first = chunk.splitlines()[0].strip()
        table = first.split(".")[-1]
        if table in wanted:
            selected.append("## " + chunk.rstrip())
    return "\n\n".join(selected)
