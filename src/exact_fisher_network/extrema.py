"""Exact extremal continuation weights and witnesses.

This module is deliberately small.  It uses the Mehta--Patel residual-margin
recurrence directly; it does not call a generic optimization package and it
does not use floating-point logarithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial
from typing import Iterator, Sequence


def multinomial(parts: Sequence[int]) -> int:
    return _multinomial_tuple(tuple(parts))


@lru_cache(maxsize=None)
def _multinomial_tuple(parts: tuple[int, ...]) -> int:
    result = factorial(sum(parts))
    for value in parts:
        result //= factorial(value)
    return result


def column_vectors(total: int, upper: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Yield bounded compositions in lexicographic order."""

    if len(upper) == 1:
        if total <= upper[0]:
            yield (total,)
        return
    lower = max(0, total - sum(upper[1:]))
    high = min(total, upper[0])
    for first in range(lower, high + 1):
        for tail in column_vectors(total - first, upper[1:]):
            yield (first, *tail)


@dataclass(frozen=True)
class Extremum:
    weight: int
    columns: tuple[tuple[int, ...], ...]


class ExactExtrema:
    """Memoized exact longest/shortest path recurrence."""

    def __init__(self) -> None:
        self.longest_problems_solved = 0
        self.shortest_problems_solved = 0

    @lru_cache(maxsize=None)
    def longest(
        self,
        residual_rows: tuple[int, ...],
        remaining_columns: tuple[int, ...],
    ) -> Extremum:
        self.longest_problems_solved += 1
        self._validate_state(residual_rows, remaining_columns)
        if not remaining_columns:
            return Extremum(1, ())

        best: Extremum | None = None
        for column in column_vectors(remaining_columns[0], residual_rows):
            child_rows = tuple(
                residual_rows[index] - column[index]
                for index in range(len(residual_rows))
            )
            suffix = self.longest(child_rows, remaining_columns[1:])
            candidate = Extremum(
                multinomial(column) * suffix.weight,
                (column, *suffix.columns),
            )
            if best is None or (candidate.weight, candidate.columns) > (
                best.weight,
                best.columns,
            ):
                best = candidate
        if best is None:
            raise AssertionError("A feasible state has no continuation.")
        return best

    @lru_cache(maxsize=None)
    def shortest(
        self,
        residual_rows: tuple[int, ...],
        remaining_columns: tuple[int, ...],
    ) -> Extremum:
        self.shortest_problems_solved += 1
        self._validate_state(residual_rows, remaining_columns)
        if not remaining_columns:
            return Extremum(1, ())

        best: Extremum | None = None
        for column in column_vectors(remaining_columns[0], residual_rows):
            child_rows = tuple(
                residual_rows[index] - column[index]
                for index in range(len(residual_rows))
            )
            suffix = self.shortest(child_rows, remaining_columns[1:])
            candidate = Extremum(
                multinomial(column) * suffix.weight,
                (column, *suffix.columns),
            )
            if best is None or (candidate.weight, candidate.columns) < (
                best.weight,
                best.columns,
            ):
                best = candidate
        if best is None:
            raise AssertionError("A feasible state has no continuation.")
        return best

    @staticmethod
    def _validate_state(
        residual_rows: tuple[int, ...],
        remaining_columns: tuple[int, ...],
    ) -> None:
        if any(value < 0 for value in (*residual_rows, *remaining_columns)):
            raise ValueError("Residual margins must be nonnegative.")
        if sum(residual_rows) != sum(remaining_columns):
            raise ValueError("Residual row and column totals disagree.")
        if len(residual_rows) < 2:
            raise ValueError("At least two residual rows are required.")


def branch_longest_weights(
    residual_rows: tuple[int, ...],
    remaining_columns: tuple[int, ...],
    extrema: ExactExtrema | None = None,
) -> dict[tuple[int, ...], int]:
    """Return the best full continuation weight for each outgoing column."""

    if not remaining_columns:
        return {}
    solver = extrema if extrema is not None else ExactExtrema()
    result: dict[tuple[int, ...], int] = {}
    for column in column_vectors(remaining_columns[0], residual_rows):
        child_rows = tuple(
            residual_rows[index] - column[index]
            for index in range(len(residual_rows))
        )
        result[column] = (
            multinomial(column)
            * solver.longest(child_rows, remaining_columns[1:]).weight
        )
    return result


def three_row_slices(
    weights: dict[tuple[int, ...], int],
) -> tuple[tuple[tuple[tuple[int, int, int], int], ...], ...]:
    """Slice a three-row daughter triangle by the third allocation."""

    if any(len(column) != 3 for column in weights):
        raise ValueError("The slice theorem is specialized to three rows.")
    grouped: dict[int, list[tuple[tuple[int, int, int], int]]] = {}
    for column, weight in weights.items():
        grouped.setdefault(column[2], []).append((column, weight))
    return tuple(
        tuple(sorted(grouped[third], key=lambda item: item[0][0]))
        for third in sorted(grouped)
    )
