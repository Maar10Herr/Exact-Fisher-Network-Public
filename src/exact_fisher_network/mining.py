"""Deterministic conjecture mining for exact three-row network relations."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterator

from .extrema import (
    ExactExtrema,
    branch_longest_weights,
    multinomial,
    three_row_slices,
)


def positive_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - parts + 2):
        for tail in positive_compositions(total - first, parts - 1):
            yield (first, *tail)


def positive_partitions(
    total: int,
    parts: int,
    minimum: int = 1,
) -> Iterator[tuple[int, ...]]:
    """Yield nondecreasing positive tuples with a fixed sum."""

    if parts == 1:
        if total >= minimum:
            yield (total,)
        return
    high = total // parts
    for first in range(minimum, high + 1):
        for tail in positive_partitions(total - first, parts - 1, first):
            yield (first, *tail)


def permutation_count(values: tuple[int, ...]) -> int:
    result = math.factorial(len(values))
    for multiplicity in Counter(values).values():
        result //= math.factorial(multiplicity)
    return result


@dataclass(frozen=True)
class Counterexample:
    conjecture: str
    total: int
    row_margins: tuple[int, int, int]
    column_margins: tuple[int, ...]
    witness: dict[str, object]


@dataclass
class MiningSummary:
    domain: dict[str, int]
    margin_problems: int = 0
    daughter_triangles: int = 0
    transfer_lines: int = 0
    adjacent_triples: int = 0
    exchange_pairs: int = 0
    ordered_margin_problems_represented: int = 0
    counterexamples: dict[str, Counterexample] | None = None

    def to_jsonable(self) -> dict[str, object]:
        result = asdict(self)
        return result


def _first(
    counterexamples: dict[str, Counterexample],
    candidate: Counterexample,
) -> None:
    counterexamples.setdefault(candidate.conjecture, candidate)


def _check_triangle(
    *,
    total: int,
    rows: tuple[int, int, int],
    columns: tuple[int, ...],
    weights: dict[tuple[int, ...], int],
    summary: MiningSummary,
    counterexamples: dict[str, Counterexample],
) -> None:
    summary.daughter_triangles += 1
    slices = three_row_slices(weights)
    for line in slices:
        summary.transfer_lines += 1
        values = tuple(weight for _, weight in line)
        columns_on_line = tuple(column for column, _ in line)

        if any(values[index] > values[index + 1] for index in range(len(values) - 1)):
            _first(
                counterexamples,
                Counterexample(
                    "line_nondecreasing_in_first_coordinate",
                    total,
                    rows,
                    columns,
                    {
                        "line": columns_on_line,
                        "weights": values,
                    },
                ),
            )

        for index in range(1, len(values) - 1):
            summary.adjacent_triples += 1
            left, middle, right = values[index - 1 : index + 2]
            if middle * middle < left * right:
                _first(
                    counterexamples,
                    Counterexample(
                        "multiplicative_line_log_concavity",
                        total,
                        rows,
                        columns,
                        {
                            "columns": columns_on_line[index - 1 : index + 2],
                            "weights": (left, middle, right),
                            "comparison": {
                                "middle_squared": middle * middle,
                                "neighbor_product": left * right,
                            },
                        },
                    ),
                )
            if middle * middle == left * right:
                _first(
                    counterexamples,
                    Counterexample(
                        "strict_multiplicative_line_log_concavity",
                        total,
                        rows,
                        columns,
                        {
                            "columns": columns_on_line[index - 1 : index + 2],
                            "weights": (left, middle, right),
                        },
                    ),
                )

    daughter_columns = tuple(sorted(weights))
    for left_column, right_column in combinations(daughter_columns, 2):
        left_arc = multinomial(left_column)
        right_arc = multinomial(right_column)
        if (
            left_arc > right_arc
            and weights[left_column] < weights[right_column]
        ):
            _first(
                counterexamples,
                Counterexample(
                    "arc_weight_orders_branch_longest",
                    total,
                    rows,
                    columns,
                    {
                        "left": left_column,
                        "right": right_column,
                        "left_arc_weight": left_arc,
                        "right_arc_weight": right_arc,
                        "left_branch_longest": weights[left_column],
                        "right_branch_longest": weights[right_column],
                    },
                ),
            )
        if (
            right_arc > left_arc
            and weights[right_column] < weights[left_column]
        ):
            _first(
                counterexamples,
                Counterexample(
                    "arc_weight_orders_branch_longest",
                    total,
                    rows,
                    columns,
                    {
                        "left": right_column,
                        "right": left_column,
                        "left_arc_weight": right_arc,
                        "right_arc_weight": left_arc,
                        "left_branch_longest": weights[right_column],
                        "right_branch_longest": weights[left_column],
                    },
                ),
            )
        for row_out in range(3):
            if left_column[row_out] <= right_column[row_out]:
                continue
            exchange_rows = [
                row_in
                for row_in in range(3)
                if left_column[row_in] < right_column[row_in]
            ]
            if not exchange_rows:
                continue
            summary.exchange_pairs += 1
            exchange_holds = False
            alternatives: list[dict[str, object]] = []
            for row_in in exchange_rows:
                left_exchanged = list(left_column)
                right_exchanged = list(right_column)
                left_exchanged[row_out] -= 1
                left_exchanged[row_in] += 1
                right_exchanged[row_out] += 1
                right_exchanged[row_in] -= 1
                left_key = tuple(left_exchanged)
                right_key = tuple(right_exchanged)
                lhs = weights[left_column] * weights[right_column]
                rhs = weights[left_key] * weights[right_key]
                alternatives.append(
                    {
                        "row_in": row_in,
                        "left_exchanged": left_key,
                        "right_exchanged": right_key,
                        "lhs": lhs,
                        "rhs": rhs,
                    }
                )
                if lhs <= rhs:
                    exchange_holds = True
            if not exchange_holds:
                _first(
                    counterexamples,
                    Counterexample(
                        "multiplicative_M_concave_exchange",
                        total,
                        rows,
                        columns,
                        {
                            "left": left_column,
                            "right": right_column,
                            "row_out": row_out,
                            "alternatives": alternatives,
                        },
                    ),
                )


def mine(
    *,
    max_total: int,
    max_columns: int,
    min_columns: int = 2,
    symmetry_reduced: bool = False,
) -> MiningSummary:
    summary = MiningSummary(
        domain={
            "rows": 3,
            "min_columns": min_columns,
            "max_columns": max_columns,
            "max_total": max_total,
            "positive_margins": 1,
            "symmetry_reduced": int(symmetry_reduced),
        },
        counterexamples={},
    )
    assert summary.counterexamples is not None

    for total in range(3, max_total + 1):
        for column_count in range(
            min_columns,
            min(max_columns, total) + 1,
        ):
            if symmetry_reduced:
                column_iterable = (
                    (first, *tail)
                    for first in range(1, total - column_count + 2)
                    for tail in positive_partitions(
                        total - first,
                        column_count - 1,
                    )
                )
            else:
                column_iterable = positive_compositions(total, column_count)
            for columns in column_iterable:
                solver = ExactExtrema()
                row_iterable = (
                    positive_partitions(total, 3)
                    if symmetry_reduced
                    else positive_compositions(total, 3)
                )
                for rows_untyped in row_iterable:
                    rows = (rows_untyped[0], rows_untyped[1], rows_untyped[2])
                    summary.margin_problems += 1
                    if symmetry_reduced:
                        summary.ordered_margin_problems_represented += (
                            permutation_count(rows)
                            * permutation_count(columns[1:])
                        )
                    else:
                        summary.ordered_margin_problems_represented += 1
                    weights = branch_longest_weights(rows, columns, solver)
                    _check_triangle(
                        total=total,
                        rows=rows,
                        columns=columns,
                        weights=weights,
                        summary=summary,
                        counterexamples=summary.counterexamples,
                    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-total", type=int, default=10)
    parser.add_argument("--max-columns", type=int, default=5)
    parser.add_argument("--symmetry-reduced", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    summary = mine(
        max_total=arguments.max_total,
        max_columns=arguments.max_columns,
        symmetry_reduced=arguments.symmetry_reduced,
    )
    encoded = json.dumps(summary.to_jsonable(), indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
