"""Independent replay of an exact network certificate.

This module imports neither ``network`` nor ``certificate``.  It reconstructs
all suffix multisets and the entire trimmed frontier from the observed table.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from math import factorial
from typing import Iterator, Mapping


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}.")
    return value


def _vector(value: object, name: str, *, positive: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list.")
    minimum = 1 if positive else 0
    return tuple(_integer(item, f"{name}[]", minimum=minimum) for item in value)


def _multinomial(parts: tuple[int, ...]) -> int:
    result = factorial(sum(parts))
    for part in parts:
        result //= factorial(part)
    return result


def _columns(total: int, upper: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    if len(upper) == 1:
        if total <= upper[0]:
            yield (total,)
        return
    lower = max(0, total - sum(upper[1:]))
    high = min(total, upper[0])
    for first in range(lower, high + 1):
        for tail in _columns(total - first, upper[1:]):
            yield (first, *tail)


@lru_cache(maxsize=None)
def _continuations(
    rows: tuple[int, ...],
    columns: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    if not columns:
        if any(rows):
            raise ValueError("Terminal residual rows are nonzero.")
        return ((1, 1),)
    if sum(rows) != sum(columns):
        raise ValueError("Residual row and column totals disagree.")
    values: Counter[int] = Counter()
    for column in _columns(columns[0], rows):
        child = tuple(rows[index] - column[index] for index in range(len(rows)))
        arc = _multinomial(column)
        for suffix, count in _continuations(child, columns[1:]):
            values[arc * suffix] += count
    return tuple(sorted(values.items()))


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return value


def validate_certificate_data(data: Mapping[str, object]) -> Fraction:
    if data.get("schema") != "exact-fisher-network-certificate-v1":
        raise ValueError("Unknown certificate schema.")

    observed_value = data.get("observed")
    if not isinstance(observed_value, list) or len(observed_value) < 2:
        raise ValueError("observed must contain at least two rows.")
    observed = tuple(
        _vector(row, f"observed[{index}]")
        for index, row in enumerate(observed_value)
    )
    width = len(observed[0])
    if width < 2 or any(len(row) != width for row in observed):
        raise ValueError("observed must be a rectangular table with at least two columns.")

    rows = tuple(sum(row) for row in observed)
    columns = tuple(
        sum(observed[row][column] for row in range(len(observed)))
        for column in range(width)
    )
    if any(value <= 0 for value in (*rows, *columns)):
        raise ValueError("All observed margins must be positive.")
    threshold = 1
    for column in range(width):
        threshold *= _multinomial(
            tuple(observed[row][column] for row in range(len(observed)))
        )
    denominator = _multinomial(rows)

    result = _mapping(data.get("result"), "result")
    if _vector(result.get("row_margins"), "result.row_margins", positive=True) != rows:
        raise ValueError("Certified row margins disagree with the observed table.")
    if (
        _vector(
            result.get("column_margins"),
            "result.column_margins",
            positive=True,
        )
        != columns
    ):
        raise ValueError("Certified column margins disagree with the observed table.")
    if _integer(result.get("observed_weight"), "result.observed_weight", minimum=1) != threshold:
        raise ValueError("Certified observed weight is incorrect.")
    if _integer(result.get("denominator"), "result.denominator", minimum=1) != denominator:
        raise ValueError("Certified denominator is incorrect.")

    state_values = data.get("states")
    pruning_values = data.get("pruning")
    if not isinstance(state_values, list) or not isinstance(pruning_values, list):
        raise ValueError("states and pruning must be lists.")

    states: dict[
        tuple[int, tuple[int, ...], tuple[int, ...]],
        Mapping[str, object],
    ] = {}
    for index, raw in enumerate(state_values):
        state = _mapping(raw, f"states[{index}]")
        layer = _integer(state.get("layer"), f"states[{index}].layer")
        residual = _vector(
            state.get("residual_rows"),
            f"states[{index}].residual_rows",
        )
        remaining = _vector(
            state.get("remaining_columns"),
            f"states[{index}].remaining_columns",
        ) if state.get("remaining_columns") != [] else ()
        if len(residual) != len(rows) or remaining != columns[layer:]:
            raise ValueError("A state has inconsistent dimensions or layer.")
        expected = _continuations(residual, remaining)
        raw_multiset = state.get("continuation_multiset")
        if not isinstance(raw_multiset, list):
            raise ValueError("A state continuation multiset must be a list.")
        certified = tuple(
            (
                _integer(pair[0], "continuation weight", minimum=1),
                _integer(pair[1], "continuation multiplicity", minimum=1),
            )
            for pair in raw_multiset
            if isinstance(pair, list) and len(pair) == 2
        )
        if len(certified) != len(raw_multiset) or certified != expected:
            raise ValueError("A continuation multiset is incomplete or incorrect.")
        shortest = expected[0][0]
        longest = expected[-1][0]
        total = sum(weight * count for weight, count in expected)
        if _integer(state.get("shortest_completion"), "shortest_completion", minimum=1) != shortest:
            raise ValueError("A shortest completion is incorrect.")
        if _integer(state.get("longest_completion"), "longest_completion", minimum=1) != longest:
            raise ValueError("A longest completion is incorrect.")
        if _integer(state.get("total_completion_weight"), "total_completion_weight", minimum=1) != total:
            raise ValueError("A total completion weight is incorrect.")
        key = (layer, residual, remaining)
        if key in states:
            raise ValueError("A state is duplicated.")
        states[key] = state

    pruning: dict[
        tuple[int, tuple[int, ...], int],
        Mapping[str, object],
    ] = {}
    for index, raw in enumerate(pruning_values):
        record = _mapping(raw, f"pruning[{index}]")
        layer = _integer(record.get("layer"), f"pruning[{index}].layer")
        residual = _vector(
            record.get("residual_rows"),
            f"pruning[{index}].residual_rows",
        )
        remaining = (
            _vector(
                record.get("remaining_columns"),
                f"pruning[{index}].remaining_columns",
            )
            if record.get("remaining_columns") != []
            else ()
        )
        if remaining != columns[layer:]:
            raise ValueError("A pruning record has an inconsistent layer.")
        prefix = _integer(
            record.get("prefix_weight"),
            f"pruning[{index}].prefix_weight",
            minimum=1,
        )
        key = (layer, residual, prefix)
        if key in pruning:
            raise ValueError("A prefix-state pruning record is duplicated.")
        pruning[key] = record

    frontier: dict[tuple[int, ...], Counter[int]] = {rows: Counter({1: 1})}
    accepted_weight = 0
    visited_states: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
    visited_pruning: set[tuple[int, tuple[int, ...], int]] = set()

    for layer in range(len(columns) + 1):
        remaining = columns[layer:]
        next_frontier: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for residual, prefixes in sorted(frontier.items()):
            state_key = (layer, residual, remaining)
            if state_key not in states:
                raise ValueError("A reachable residual state is missing.")
            visited_states.add(state_key)
            suffix = _continuations(residual, remaining)
            shortest = suffix[0][0]
            longest = suffix[-1][0]
            total = sum(weight * count for weight, count in suffix)

            for prefix, multiplicity in sorted(prefixes.items()):
                record_key = (layer, residual, prefix)
                record = pruning.get(record_key)
                if record is None:
                    raise ValueError("A reachable prefix-state pair is missing.")
                visited_pruning.add(record_key)
                certified_multiplicity = _integer(
                    record.get("prefix_multiplicity"),
                    "prefix_multiplicity",
                    minimum=1,
                )
                if certified_multiplicity != multiplicity:
                    raise ValueError("A prefix multiplicity is incorrect.")
                if record.get("remaining_columns") != list(remaining):
                    raise ValueError("A pruning record has incorrect remaining columns.")
                for name, certified, expected in (
                    ("shortest_completion", record.get("shortest_completion"), shortest),
                    ("longest_completion", record.get("longest_completion"), longest),
                    ("observed_weight", record.get("observed_weight"), threshold),
                ):
                    if _integer(certified, name, minimum=1) != expected:
                        raise ValueError(f"A pruning record has incorrect {name}.")

                decision = record.get("decision")
                certified_accepted = _integer(
                    record.get("accepted_weight"),
                    "accepted_weight",
                )
                if prefix * longest <= threshold:
                    expected_decision = "accept_all"
                    contribution = multiplicity * prefix * total
                    if certified_accepted != contribution:
                        raise ValueError("An accept-all contribution is incorrect.")
                    accepted_weight += contribution
                elif prefix * shortest > threshold:
                    expected_decision = "reject_all"
                    if certified_accepted != 0:
                        raise ValueError("A reject-all record has nonzero contribution.")
                else:
                    expected_decision = "expand"
                    if certified_accepted != 0 or not remaining:
                        raise ValueError("An expand record is invalid.")
                    for column in _columns(remaining[0], residual):
                        child = tuple(
                            residual[index] - column[index]
                            for index in range(len(residual))
                        )
                        next_frontier[child][prefix * _multinomial(column)] += multiplicity
                if decision != expected_decision:
                    raise ValueError("A pruning decision is not certified by its inequalities.")
        frontier = dict(next_frontier)

    if frontier:
        raise ValueError("Certificate replay left an unprocessed frontier.")
    if visited_states != set(states):
        raise ValueError("Certificate contains unreachable or unused states.")
    if visited_pruning != set(pruning):
        raise ValueError("Certificate contains unreachable or unused pruning records.")

    certified_weight = _integer(
        result.get("accepted_weight"),
        "result.accepted_weight",
    )
    if certified_weight != accepted_weight:
        raise ValueError("Certified accepted weight is incorrect.")
    p_value = Fraction(accepted_weight, denominator)
    p_data = _mapping(result.get("p_value"), "result.p_value")
    certified_p = Fraction(
        _integer(p_data.get("numerator"), "p_value.numerator"),
        _integer(p_data.get("denominator"), "p_value.denominator", minimum=1),
    )
    if certified_p != p_value:
        raise ValueError("Certified p-value is incorrect.")
    return p_value

