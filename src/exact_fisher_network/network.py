"""Exact baseline residual-margin network with replayable instrumentation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import factorial
from typing import Iterator, Mapping, Sequence

WeightMultiset = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StateRecord:
    layer: int
    residual_rows: tuple[int, ...]
    remaining_columns: tuple[int, ...]
    continuation_multiset: WeightMultiset
    shortest_completion: int
    longest_completion: int
    total_completion_weight: int


@dataclass(frozen=True)
class PruningRecord:
    layer: int
    residual_rows: tuple[int, ...]
    remaining_columns: tuple[int, ...]
    prefix_weight: int
    prefix_multiplicity: int
    shortest_completion: int
    longest_completion: int
    observed_weight: int
    decision: str
    accepted_weight: int


@dataclass(frozen=True)
class NetworkMetrics:
    residual_states_created: int
    arcs_traversed: int
    extremal_completion_problems_solved: int
    pruning_comparisons: int
    prefix_state_pairs_checked: int


@dataclass(frozen=True)
class NetworkResult:
    row_margins: tuple[int, ...]
    column_margins: tuple[int, ...]
    observed_weight: int
    denominator: int
    p_value: Fraction
    accepted_weight: int
    state_records: tuple[StateRecord, ...]
    pruning_records: tuple[PruningRecord, ...]
    metrics: NetworkMetrics


def _validate_observed(table: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(int(value) for value in row) for row in table)
    if len(rows) < 2 or not rows or len(rows[0]) < 2:
        raise ValueError("A contingency table must have at least two rows and columns.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("All rows must have the same length.")
    if any(value < 0 for row in rows for value in row):
        raise ValueError("Cell counts must be nonnegative.")
    if any(sum(row) <= 0 for row in rows):
        raise ValueError("Row margins must be positive.")
    if any(sum(rows[i][j] for i in range(len(rows))) <= 0 for j in range(width)):
        raise ValueError("Column margins must be positive.")
    return rows


def _multinomial(parts: Sequence[int]) -> int:
    result = factorial(sum(parts))
    for value in parts:
        result //= factorial(value)
    return result


def _column_vectors(total: int, upper: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    if len(upper) == 1:
        if total <= upper[0]:
            yield (total,)
        return
    lower = max(0, total - sum(upper[1:]))
    high = min(total, upper[0])
    for first in range(lower, high + 1):
        for tail in _column_vectors(total - first, upper[1:]):
            yield (first, *tail)


def _observed_path_weight(table: tuple[tuple[int, ...], ...]) -> int:
    weight = 1
    for column_index in range(len(table[0])):
        weight *= _multinomial(
            tuple(table[row][column_index] for row in range(len(table)))
        )
    return weight


def _as_multiset(counter: Mapping[int, int]) -> WeightMultiset:
    return tuple(sorted((int(weight), int(count)) for weight, count in counter.items()))


class _ContinuationOracle:
    """Memoized exact suffix distributions used by the baseline network."""

    def __init__(self) -> None:
        self.states_solved = 0

    @lru_cache(maxsize=None)
    def counter(
        self,
        residual_rows: tuple[int, ...],
        remaining_columns: tuple[int, ...],
    ) -> WeightMultiset:
        self.states_solved += 1
        if not remaining_columns:
            if any(residual_rows):
                raise AssertionError("A terminal state has nonzero residual row margins.")
            return ((1, 1),)
        if sum(residual_rows) != sum(remaining_columns):
            raise AssertionError("Residual row and column totals disagree.")

        first_total = remaining_columns[0]
        collected: Counter[int] = Counter()
        for column in _column_vectors(first_total, residual_rows):
            child_rows = tuple(
                residual_rows[index] - column[index]
                for index in range(len(residual_rows))
            )
            arc_weight = _multinomial(column)
            for suffix_weight, multiplicity in self.counter(
                child_rows,
                remaining_columns[1:],
            ):
                collected[arc_weight * suffix_weight] += multiplicity
        if not collected:
            raise AssertionError("A feasible residual state has no completions.")
        expected_mass = _multinomial(residual_rows)
        actual_mass = sum(weight * count for weight, count in collected.items())
        if actual_mass != expected_mass:
            raise AssertionError("The continuation multiset has incorrect total mass.")
        return _as_multiset(collected)


def exact_network(observed: Sequence[Sequence[int]]) -> NetworkResult:
    """Run exact path trimming on the residual-margin network.

    The frontier maps each residual state to a multiset of prefix path
    weights.  For every prefix-state pair, exact suffix extrema certify one
    of three actions:

    - ``accept_all``: every completion is in the probability tail;
    - ``reject_all``: no completion is in the tail;
    - ``expand``: traverse the next-column arcs.
    """

    table = _validate_observed(observed)
    row_margins = tuple(sum(row) for row in table)
    column_margins = tuple(
        sum(table[row][column] for row in range(len(table)))
        for column in range(len(table[0]))
    )
    threshold = _observed_path_weight(table)
    denominator = _multinomial(row_margins)
    continuation = _ContinuationOracle()

    frontier: dict[tuple[int, ...], Counter[int]] = {
        row_margins: Counter({1: 1})
    }
    state_records: dict[
        tuple[int, tuple[int, ...], tuple[int, ...]],
        StateRecord,
    ] = {}
    pruning_records: list[PruningRecord] = []
    accepted_weight = 0
    arcs_traversed = 0
    pruning_comparisons = 0
    prefix_state_pairs_checked = 0

    for layer in range(len(column_margins) + 1):
        remaining = column_margins[layer:]
        next_frontier: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)

        for residual_rows in sorted(frontier):
            suffix = continuation.counter(residual_rows, remaining)
            shortest = suffix[0][0]
            longest = suffix[-1][0]
            total_suffix_weight = sum(
                weight * multiplicity for weight, multiplicity in suffix
            )
            state_key = (layer, residual_rows, remaining)
            state_records[state_key] = StateRecord(
                layer=layer,
                residual_rows=residual_rows,
                remaining_columns=remaining,
                continuation_multiset=suffix,
                shortest_completion=shortest,
                longest_completion=longest,
                total_completion_weight=total_suffix_weight,
            )

            for prefix_weight, prefix_multiplicity in sorted(
                frontier[residual_rows].items()
            ):
                prefix_state_pairs_checked += 1
                pruning_comparisons += 1
                if prefix_weight * longest <= threshold:
                    contribution = (
                        prefix_multiplicity
                        * prefix_weight
                        * total_suffix_weight
                    )
                    accepted_weight += contribution
                    pruning_records.append(
                        PruningRecord(
                            layer=layer,
                            residual_rows=residual_rows,
                            remaining_columns=remaining,
                            prefix_weight=prefix_weight,
                            prefix_multiplicity=prefix_multiplicity,
                            shortest_completion=shortest,
                            longest_completion=longest,
                            observed_weight=threshold,
                            decision="accept_all",
                            accepted_weight=contribution,
                        )
                    )
                    continue

                pruning_comparisons += 1
                if prefix_weight * shortest > threshold:
                    pruning_records.append(
                        PruningRecord(
                            layer=layer,
                            residual_rows=residual_rows,
                            remaining_columns=remaining,
                            prefix_weight=prefix_weight,
                            prefix_multiplicity=prefix_multiplicity,
                            shortest_completion=shortest,
                            longest_completion=longest,
                            observed_weight=threshold,
                            decision="reject_all",
                            accepted_weight=0,
                        )
                    )
                    continue

                if not remaining:
                    raise AssertionError(
                        "A terminal prefix must be accepted or rejected exactly."
                    )
                pruning_records.append(
                    PruningRecord(
                        layer=layer,
                        residual_rows=residual_rows,
                        remaining_columns=remaining,
                        prefix_weight=prefix_weight,
                        prefix_multiplicity=prefix_multiplicity,
                        shortest_completion=shortest,
                        longest_completion=longest,
                        observed_weight=threshold,
                        decision="expand",
                        accepted_weight=0,
                    )
                )
                for column in _column_vectors(remaining[0], residual_rows):
                    arcs_traversed += 1
                    child_rows = tuple(
                        residual_rows[index] - column[index]
                        for index in range(len(residual_rows))
                    )
                    next_frontier[child_rows][
                        prefix_weight * _multinomial(column)
                    ] += prefix_multiplicity

        frontier = dict(next_frontier)

    if frontier:
        raise AssertionError("The final frontier was not exhausted.")
    if accepted_weight > denominator:
        raise AssertionError("The accepted conditional mass exceeds one.")

    return NetworkResult(
        row_margins=row_margins,
        column_margins=column_margins,
        observed_weight=threshold,
        denominator=denominator,
        p_value=Fraction(accepted_weight, denominator),
        accepted_weight=accepted_weight,
        state_records=tuple(
            state_records[key]
            for key in sorted(state_records)
        ),
        pruning_records=tuple(pruning_records),
        metrics=NetworkMetrics(
            residual_states_created=len(state_records),
            arcs_traversed=arcs_traversed,
            extremal_completion_problems_solved=continuation.states_solved,
            pruning_comparisons=pruning_comparisons,
            prefix_state_pairs_checked=prefix_state_pairs_checked,
        ),
    )

