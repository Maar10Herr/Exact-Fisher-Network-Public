"""Compact production certificates for accept-only exact batching."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .flow_extrema import exact_flow_potential
from .optimized import BoundQuery, EngineResult


def _fraction_data(value) -> tuple[int, int]:
    return (value.numerator, value.denominator)


def _extremum_certificate(
    columns: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    if not columns:
        return {
            "columns": (),
            "row_scalars": (),
            "column_scalars": (),
        }
    row_count = len(columns[0])
    table = tuple(
        tuple(column[row] for column in columns)
        for row in range(row_count)
    )
    potential = exact_flow_potential(table)
    return {
        "columns": columns,
        "row_scalars": tuple(
            _fraction_data(value) for value in potential.row_scalars
        ),
        "column_scalars": tuple(
            _fraction_data(value) for value in potential.column_scalars
        ),
    }


def _query_data(
    query: BoundQuery,
    register_extremum,
) -> dict[str, object]:
    suffix = query.maximizing_columns[1:]
    return {
        "index": query.index,
        "column": query.column,
        "branch_longest": query.branch_longest,
        "suffix_extremum_id": register_extremum(suffix),
    }


def compact_certificate_data(result: EngineResult) -> dict[str, object]:
    if not result.batching:
        raise ValueError("Compact certificates require batching=True.")
    if result.reject_pruning:
        raise ValueError(
            "Compact certificates require reject_pruning=False so that "
            "every nonterminal extremum has a convex-flow dual."
        )
    if (
        result.metrics.prefix_state_pairs_checked > 0
        and not result.state_records
    ):
        raise ValueError(
            "The engine was run with collect_certificate=False."
        )

    grouped_states: dict[
        tuple[int, tuple[int, ...], tuple[int, ...]],
        list,
    ] = defaultdict(list)
    for record in result.state_records:
        grouped_states[
            (
                record.layer,
                record.residual_rows,
                record.remaining_columns,
            )
        ].append(record)

    extrema: list[dict[str, object]] = []
    extremum_ids: dict[tuple[tuple[int, ...], ...], int] = {}

    def register_extremum(
        columns: tuple[tuple[int, ...], ...],
    ) -> int:
        if columns not in extremum_ids:
            extremum_ids[columns] = len(extrema)
            extrema.append(_extremum_certificate(columns))
        return extremum_ids[columns]

    states = []
    for key in sorted(grouped_states):
        records = grouped_states[key]
        first = records[0]
        if any(
            record.longest_completion != first.longest_completion
            or record.longest_columns != first.longest_columns
            for record in records[1:]
        ):
            raise AssertionError("A state has inconsistent longest witnesses.")
        states.append(
            {
                "layer": first.layer,
                "residual_rows": first.residual_rows,
                "remaining_columns": first.remaining_columns,
                "longest_completion": first.longest_completion,
                "longest_extremum_id": register_extremum(
                    first.longest_columns
                ),
                "accepted_weight": sum(
                    record.accepted_weight
                    for record in records
                    if record.decision == "accept_all"
                ),
            }
        )

    batches = [
        {
            "layer": record.layer,
            "residual_rows": record.residual_rows,
            "remaining_columns": record.remaining_columns,
            "prefix_weight": record.prefix_weight,
            "prefix_multiplicity": record.prefix_multiplicity,
            "fixed_tail": record.fixed_tail,
            "line_length": len(record.line_columns),
            "peak_index": record.peak_index,
            "accepted_left_end": record.accepted_left_end,
            "accepted_right_start": record.accepted_right_start,
            "accepted_weight": record.accepted_weight,
            "queries": tuple(
                _query_data(query, register_extremum)
                for query in record.queries
            ),
        }
        for record in result.batch_records
    ]

    return {
        "schema": "exact-fisher-compact-v1",
        "theorems": (
            "network-induced M-concavity",
            "strict daughter-line log-concavity",
            "strict cross-slice multiplicative Monge",
            "exact daughter-mass recurrence",
        ),
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
        "extrema": extrema,
        "states": states,
        "batches": batches,
        "metrics": asdict(result.metrics),
    }


def write_compact_certificate(
    result: EngineResult,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(
            compact_certificate_data(result),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
