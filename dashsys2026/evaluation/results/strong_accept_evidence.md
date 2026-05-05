# Strong-Accept Evidence Artifacts

## Current DataPup vs Revised Prompt

| Configuration | EX | RC | Correct |
|---|---:|---:|---:|
| Full-schema JSON zero-shot | 48.7% | 17.3% | 26/150 |
| Markdown relevant-subset descriptions dynamic few-shot | 97.3% | 66.0% | 99/150 |

Delta: +48.7pp EX, +48.7pp RC; McNemar exact p=2.72e-19 (76 revised-only correct vs. 3 current-only correct).

## Semantic Assumption Audit

| Failure/Outcome | Count |
|---|---:|
| Correct | 240 |
| Wrong granularity or filter | 66 |
| Projection mismatch | 48 |
| Wrong values or ranking | 32 |
| Grouping-key mismatch | 26 |
| Missing filter assumption | 23 |
| Timeout / empty output | 10 |
| Other wrong result | 3 |
| Residual execution error | 2 |

| Assumption to Surface | Wrong-executable Count |
|---|---:|
| ranking / tie-break | 192 |
| metric definition | 127 |
| filter / predicate | 120 |
| time window / bucket | 97 |
| grouping key | 94 |
| window frame | 56 |
| join path | 46 |
| JSON property | 11 |

## DuckDB CLI Validation

| Config | EX | RC | Correct set |
|---|---:|---:|---:|
| Baseline | 97.7% | 60.8% | 130 portable queries |
| Best | 100.0% | 60.8% | 130 portable queries |

Delta: +2.3pp EX, +0.0pp RC.
