"""JSON certificate creation for exact network runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .network import exact_network


def certificate_data(observed: Sequence[Sequence[int]]) -> dict[str, object]:
    result = exact_network(observed)
    return {
        "schema": "exact-fisher-network-certificate-v1",
        "observed": [[int(value) for value in row] for row in observed],
        "result": {
            "row_margins": list(result.row_margins),
            "column_margins": list(result.column_margins),
            "observed_weight": result.observed_weight,
            "denominator": result.denominator,
            "accepted_weight": result.accepted_weight,
            "p_value": {
                "numerator": result.p_value.numerator,
                "denominator": result.p_value.denominator,
            },
            "metrics": asdict(result.metrics),
        },
        "states": [
            {
                **asdict(record),
                "residual_rows": list(record.residual_rows),
                "remaining_columns": list(record.remaining_columns),
                "continuation_multiset": [
                    [weight, count]
                    for weight, count in record.continuation_multiset
                ],
            }
            for record in result.state_records
        ],
        "pruning": [
            {
                **asdict(record),
                "residual_rows": list(record.residual_rows),
                "remaining_columns": list(record.remaining_columns),
            }
            for record in result.pruning_records
        ],
    }


def write_certificate(
    observed: Sequence[Sequence[int]],
    destination: str | Path,
) -> Path:
    path = Path(destination)
    path.write_text(
        json.dumps(certificate_data(observed), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path

