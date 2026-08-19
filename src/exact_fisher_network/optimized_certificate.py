"""JSON serialization for theorem-powered exact-network certificates."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .optimized import EngineResult


def certificate_data(result: EngineResult) -> dict[str, object]:
    if not result.batching:
        raise ValueError("An optimized certificate requires batching=True.")
    if not result.reject_pruning:
        raise ValueError(
            "The full-audit certificate requires reject_pruning=True."
        )
    if (
        result.metrics.prefix_state_pairs_checked > 0
        and not result.state_records
    ):
        raise ValueError(
            "The engine was run with collect_certificate=False; "
            "rerun with certificate collection enabled."
        )
    return {
        "schema": "exact-fisher-three-row-batching-v1",
        "observed_table": result.observed_table,
        "row_margins": result.row_margins,
        "column_margins": result.column_margins,
        "observed_weight": result.observed_weight,
        "denominator": result.denominator,
        "accepted_weight": result.accepted_weight,
        "p_value": {
            "numerator": result.p_value.numerator,
            "denominator": result.p_value.denominator,
        },
        "batching": result.batching,
        "reject_pruning": result.reject_pruning,
        "state_records": [asdict(record) for record in result.state_records],
        "batch_records": [asdict(record) for record in result.batch_records],
        "metrics": asdict(result.metrics),
    }


def write_certificate(result: EngineResult, path: str | Path) -> None:
    target = Path(path)
    target.write_text(
        json.dumps(certificate_data(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
