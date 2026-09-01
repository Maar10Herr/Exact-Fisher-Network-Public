"""Readable direct-enumeration oracle with exact integer/rational arithmetic.

This module deliberately does not import the network implementation.  It is
the independent reference route for small fixed-margin problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from typing import Iterator, Sequence

from ._validation import integer as _integer

Table = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class EnumeratedTable:
    table: Table
    path_weight: int
    probability: Fraction
    is_extreme: bool


@dataclass(frozen=True)
class EnumerationResult:
    row_margins: tuple[int, ...]
    column_margins: tuple[int, ...]
    observed_weight: int
    denominator: int
    p_value: Fraction
    table_count: int
    extreme_table_count: int
    records: tuple[EnumeratedTable, ...]


def _validated_table(table: Sequence[Sequence[int]]) -> Table:
    rows = tuple(
        tuple(_integer(value, "cell count") for value in row) for row in table
    )
    if len(rows) < 2 or not rows or len(rows[0]) < 2:
        raise ValueError("A contingency table must have at least two rows and columns.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("All rows must have the same length.")
    if any(value < 0 for row in rows for value in row):
        raise ValueError("Cell counts must be nonnegative.")
    if any(sum(row) == 0 for row in rows):
        raise ValueError("Row margins must be positive.")
    if any(sum(rows[i][j] for i in range(len(rows))) == 0 for j in range(width)):
        raise ValueError("Column margins must be positive.")
    return rows


def _bounded_column_vectors(
    total: int,
    upper: tuple[int, ...],
) -> Iterator[tuple[int, ...]]:
    """Yield lexicographically ordered vectors summing to ``total``."""

    if len(upper) == 1:
        if total <= upper[0]:
            yield (total,)
        return
    lower = max(0, total - sum(upper[1:]))
    high = min(total, upper[0])
    for first in range(lower, high + 1):
        for tail in _bounded_column_vectors(total - first, upper[1:]):
            yield (first, *tail)


def enumerate_tables(
    row_margins: Sequence[int],
    column_margins: Sequence[int],
) -> Iterator[Table]:
    """Enumerate every nonnegative integer table with the fixed margins."""

    rows = tuple(
        _integer(value, "row margin") for value in row_margins
    )
    columns = tuple(
        _integer(value, "column margin") for value in column_margins
    )
    if len(rows) < 2 or len(columns) < 2:
        raise ValueError("At least two positive row and column margins are required.")
    if any(value <= 0 for value in (*rows, *columns)):
        raise ValueError("All margins must be positive.")
    if sum(rows) != sum(columns):
        raise ValueError("Row and column margins must have the same total.")

    built: list[tuple[int, ...]] = []

    def visit(index: int, residual: tuple[int, ...]) -> Iterator[Table]:
        if index == len(columns) - 1:
            if sum(residual) != columns[index]:
                return
            all_columns = (*built, residual)
            yield tuple(
                tuple(column[row] for column in all_columns)
                for row in range(len(rows))
            )
            return
        for column in _bounded_column_vectors(columns[index], residual):
            built.append(column)
            next_residual = tuple(
                residual[row] - column[row] for row in range(len(rows))
            )
            yield from visit(index + 1, next_residual)
            built.pop()

    yield from visit(0, rows)


def path_weight(table: Sequence[Sequence[int]]) -> int:
    """Return ``prod_j multinomial(c_j; x_1j, ..., x_rj)``."""

    rows = _validated_table(table)
    result = 1
    for column in range(len(rows[0])):
        column_total = sum(rows[row][column] for row in range(len(rows)))
        factor = factorial(column_total)
        for row in range(len(rows)):
            factor //= factorial(rows[row][column])
        result *= factor
    return result


def conditional_denominator(row_margins: Sequence[int]) -> int:
    """Return ``multinomial(N; r_1, ..., r_r)``."""

    rows = tuple(_integer(value, "row margin") for value in row_margins)
    result = factorial(sum(rows))
    for value in rows:
        result //= factorial(value)
    return result


def exact_enumeration(
    observed: Sequence[Sequence[int]],
    *,
    keep_records: bool = True,
) -> EnumerationResult:
    """Compute the probability-ordered exact p-value by full enumeration."""

    observed_table = _validated_table(observed)
    row_margins = tuple(sum(row) for row in observed_table)
    column_margins = tuple(
        sum(observed_table[row][column] for row in range(len(observed_table)))
        for column in range(len(observed_table[0]))
    )
    threshold = path_weight(observed_table)
    denominator = conditional_denominator(row_margins)
    extreme_weight = 0
    table_count = 0
    extreme_count = 0
    records: list[EnumeratedTable] = []
    total_weight = 0

    for table in enumerate_tables(row_margins, column_margins):
        weight = path_weight(table)
        probability = Fraction(weight, denominator)
        is_extreme = weight <= threshold
        table_count += 1
        total_weight += weight
        if is_extreme:
            extreme_count += 1
            extreme_weight += weight
        if keep_records:
            records.append(
                EnumeratedTable(
                    table=table,
                    path_weight=weight,
                    probability=probability,
                    is_extreme=is_extreme,
                )
            )

    if total_weight != denominator:
        raise AssertionError(
            "The enumerated conditional mass does not equal the multinomial denominator."
        )

    return EnumerationResult(
        row_margins=row_margins,
        column_margins=column_margins,
        observed_weight=threshold,
        denominator=denominator,
        p_value=Fraction(extreme_weight, denominator),
        table_count=table_count,
        extreme_table_count=extreme_count,
        records=tuple(records),
    )
