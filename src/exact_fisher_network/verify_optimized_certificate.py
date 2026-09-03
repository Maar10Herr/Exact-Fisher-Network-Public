"""Independent replay of theorem-powered exact-network certificates.

This module imports neither the optimized engine nor its extremal solver.
It reconstructs every extremum by its own exact residual-margin recurrence.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from math import factorial
from pathlib import Path
from typing import Iterator, Sequence


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


def _observed_weight(table: tuple[tuple[int, ...], ...]) -> int:
    result = 1
    for column in range(len(table[0])):
        result *= _multinomial(
            tuple(table[row][column] for row in range(len(table)))
        )
    return result


def _path_weight(columns: tuple[tuple[int, ...], ...]) -> int:
    result = 1
    for column in columns:
        result *= _multinomial(column)
    return result


def _integer(value: object, name: str) -> int:
    """Accept JSON integers only; never silently truncate strings/floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssertionError(f"{name} must be an integer.")
    return value


class _IndependentExtrema:
    @lru_cache(maxsize=None)
    def longest(
        self,
        rows: tuple[int, ...],
        columns: tuple[int, ...],
    ) -> int:
        if not columns:
            if any(rows):
                raise AssertionError("Invalid terminal extremum state.")
            return 1
        best = 0
        for column in _column_vectors(columns[0], rows):
            child = tuple(rows[index] - column[index] for index in range(len(rows)))
            best = max(
                best,
                _multinomial(column) * self.longest(child, columns[1:]),
            )
        if best == 0:
            raise AssertionError("Feasible state has no longest continuation.")
        return best

    @lru_cache(maxsize=None)
    def shortest(
        self,
        rows: tuple[int, ...],
        columns: tuple[int, ...],
    ) -> int:
        if not columns:
            if any(rows):
                raise AssertionError("Invalid terminal extremum state.")
            return 1
        best: int | None = None
        for column in _column_vectors(columns[0], rows):
            child = tuple(rows[index] - column[index] for index in range(len(rows)))
            value = _multinomial(column) * self.shortest(child, columns[1:])
            if best is None or value < best:
                best = value
        if best is None:
            raise AssertionError("Feasible state has no shortest continuation.")
        return best


def _tuple_vector(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise AssertionError("Expected an integer vector.")
    return tuple(_integer(item, "vector item") for item in value)


def _tuple_matrix(value: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise AssertionError("Expected an integer matrix.")
    return tuple(_tuple_vector(row) for row in value)


def _verify_witness(
    *,
    witness: tuple[tuple[int, ...], ...],
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    expected_weight: int,
) -> None:
    if len(witness) != len(columns):
        raise AssertionError("Extremal witness has the wrong number of columns.")
    if any(len(column) != len(rows) for column in witness):
        raise AssertionError("Extremal witness has the wrong number of rows.")
    if tuple(sum(column) for column in witness) != columns:
        raise AssertionError("Extremal witness has incorrect column margins.")
    if tuple(
        sum(column[row] for column in witness)
        for row in range(len(rows))
    ) != rows:
        raise AssertionError("Extremal witness has incorrect row margins.")
    if _path_weight(witness) != expected_weight:
        raise AssertionError("Extremal witness has the wrong exact weight.")


def _record_key(record: dict[str, object]) -> tuple[object, ...]:
    return (
        _integer(record["layer"], "layer"),
        _tuple_vector(record["residual_rows"]),
        _tuple_vector(record["remaining_columns"]),
        _integer(record["prefix_weight"], "prefix_weight"),
        _integer(record["prefix_multiplicity"], "prefix_multiplicity"),
    )


def verify_certificate_data(data: dict[str, object]) -> Fraction:
    if data.get("schema") != "exact-fisher-three-row-batching-v1":
        raise AssertionError("Unknown optimized-certificate schema.")
    if data.get("batching") is not True:
        raise AssertionError("This verifier expects theorem batching to be active.")

    table = _tuple_matrix(data["observed_table"])
    if len(table) != 3 or not table or len(table[0]) < 2:
        raise AssertionError("The theorem certificate must contain a 3 x c table.")
    if any(len(row) != len(table[0]) for row in table):
        raise AssertionError("Observed table is ragged.")
    if any(value < 0 for row in table for value in row):
        raise AssertionError("Observed table has a negative cell.")

    rows = tuple(sum(row) for row in table)
    columns = tuple(
        sum(table[row][column] for row in range(3))
        for column in range(len(table[0]))
    )
    threshold = _observed_weight(table)
    denominator = _multinomial(rows)
    if _tuple_vector(data["row_margins"]) != rows:
        raise AssertionError("Certificate row margins do not match the table.")
    if _tuple_vector(data["column_margins"]) != columns:
        raise AssertionError("Certificate column margins do not match the table.")
    if _integer(data["observed_weight"], "observed_weight") != threshold:
        raise AssertionError("Certificate observed weight is corrupted.")
    if _integer(data["denominator"], "denominator") != denominator:
        raise AssertionError("Certificate denominator is corrupted.")

    raw_states = data["state_records"]
    raw_batches = data["batch_records"]
    if not isinstance(raw_states, list) or not isinstance(raw_batches, list):
        raise AssertionError("Certificate records must be lists.")

    state_map: dict[tuple[object, ...], dict[str, object]] = {}
    for raw in raw_states:
        if not isinstance(raw, dict):
            raise AssertionError("Malformed state record.")
        key = _record_key(raw)
        if key in state_map:
            raise AssertionError("Duplicate state-decision record.")
        state_map[key] = raw

    batch_map: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for raw in raw_batches:
        if not isinstance(raw, dict):
            raise AssertionError("Malformed batch record.")
        batch_map[_record_key(raw)].append(raw)

    extrema = _IndependentExtrema()
    used_states: set[tuple[object, ...]] = set()
    used_batches: set[int] = set()
    frontier: dict[tuple[int, ...], Counter[int]] = {rows: Counter({1: 1})}
    accepted_total = 0

    for layer in range(len(columns) + 1):
        remaining = columns[layer:]
        next_frontier: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for state_rows in sorted(frontier):
            actual_longest = extrema.longest(state_rows, remaining)
            for prefix, multiplicity in sorted(frontier[state_rows].items()):
                key = (layer, state_rows, remaining, prefix, multiplicity)
                if key not in state_map:
                    raise AssertionError("Missing state-decision record.")
                record = state_map[key]
                used_states.add(key)

                if _integer(record["longest_completion"], "longest_completion") != actual_longest:
                    raise AssertionError("State longest completion is corrupted.")
                longest_witness = _tuple_matrix(record["longest_columns"])
                _verify_witness(
                    witness=longest_witness,
                    rows=state_rows,
                    columns=remaining,
                    expected_weight=actual_longest,
                )

                if prefix * actual_longest <= threshold:
                    expected_decision = "accept_all"
                    actual_shortest: int | None = None
                else:
                    actual_shortest = extrema.shortest(state_rows, remaining)
                    expected_decision = (
                        "reject_all"
                        if prefix * actual_shortest > threshold
                        else "expand"
                    )
                if record["decision"] != expected_decision:
                    raise AssertionError("State pruning decision is corrupted.")

                if actual_shortest is None:
                    if record["shortest_completion"] is not None:
                        raise AssertionError("Unexpected shortest completion.")
                else:
                    if (
                        _integer(record["shortest_completion"], "shortest_completion")
                        != actual_shortest
                    ):
                        raise AssertionError("State shortest completion is corrupted.")
                    shortest_witness = _tuple_matrix(record["shortest_columns"])
                    _verify_witness(
                        witness=shortest_witness,
                        rows=state_rows,
                        columns=remaining,
                        expected_weight=actual_shortest,
                    )

                if expected_decision == "accept_all":
                    contribution = multiplicity * prefix * _multinomial(state_rows)
                    if _integer(record["accepted_weight"], "accepted_weight") != contribution:
                        raise AssertionError("State accepted contribution is corrupted.")
                    accepted_total += contribution
                    continue
                if _integer(record["accepted_weight"], "accepted_weight") != 0:
                    raise AssertionError("Non-accepting state has accepted mass.")
                if expected_decision == "reject_all":
                    continue
                if not remaining:
                    raise AssertionError("Terminal state cannot expand.")

                accepted_columns: set[tuple[int, ...]] = set()
                for batch in batch_map.get(key, []):
                    used_batches.add(id(batch))
                    line = _tuple_matrix(batch["line_columns"])
                    fixed_third = _integer(batch["fixed_third"], "fixed_third")
                    expected_line = tuple(
                        column
                        for column in _column_vectors(remaining[0], state_rows)
                        if column[2] == fixed_third
                    )
                    if line != expected_line:
                        raise AssertionError("Batch line does not match the daughter slice.")
                    if not line:
                        raise AssertionError("Batch line is empty.")

                    values: list[int] = []
                    for column in line:
                        child = tuple(
                            state_rows[index] - column[index]
                            for index in range(3)
                        )
                        values.append(
                            _multinomial(column)
                            * extrema.longest(child, remaining[1:])
                        )
                    for index in range(1, len(values) - 1):
                        if values[index] ** 2 <= values[index - 1] * values[index + 1]:
                            raise AssertionError(
                                "The claimed line violates strict log-concavity."
                            )

                    peak = _integer(batch["peak_index"], "peak_index")
                    leftmost_peak = max(range(len(values)), key=lambda index: values[index])
                    if peak != leftmost_peak:
                        raise AssertionError("Batch peak index is corrupted.")

                    query_indices: set[int] = set()
                    raw_queries = batch["queries"]
                    if not isinstance(raw_queries, (list, tuple)):
                        raise AssertionError("Batch queries must be a sequence.")
                    for query in raw_queries:
                        if not isinstance(query, dict):
                            raise AssertionError("Malformed batch query.")
                        index = _integer(query["index"], "query index")
                        if index in query_indices or not 0 <= index < len(line):
                            raise AssertionError("Invalid or duplicate batch query index.")
                        query_indices.add(index)
                        if _tuple_vector(query["column"]) != line[index]:
                            raise AssertionError("Batch query column is corrupted.")
                        if _integer(query["branch_longest"], "branch_longest") != values[index]:
                            raise AssertionError("Batch queried value is corrupted.")
                        witness = _tuple_matrix(query["maximizing_columns"])
                        _verify_witness(
                            witness=witness,
                            rows=state_rows,
                            columns=remaining,
                            expected_weight=values[index],
                        )
                        if witness[0] != line[index]:
                            raise AssertionError("Batch witness has the wrong first column.")

                    if peak not in query_indices:
                        raise AssertionError("Batch certificate omits its peak query.")

                    actual_accepted_indices = tuple(
                        index
                        for index, value in enumerate(values)
                        if prefix * value <= threshold
                    )
                    if not actual_accepted_indices:
                        raise AssertionError("A stored batch record accepts no daughters.")
                    expected_columns = tuple(line[index] for index in actual_accepted_indices)
                    if _tuple_matrix(batch["accepted_columns"]) != expected_columns:
                        raise AssertionError("Batch accepted columns are corrupted.")

                    if len(actual_accepted_indices) == len(line):
                        expected_left = len(line) - 1
                        expected_right = None
                    else:
                        left = [index for index in actual_accepted_indices if index < peak]
                        right = [index for index in actual_accepted_indices if index > peak]
                        expected_left = max(left) if left else None
                        expected_right = min(right) if right else None
                    if batch["accepted_left_end"] != expected_left:
                        raise AssertionError("Left batch boundary is corrupted.")
                    if batch["accepted_right_start"] != expected_right:
                        raise AssertionError("Right batch boundary is corrupted.")

                    required = {peak}
                    if expected_left is not None and expected_left < len(line) - 1:
                        required.update((expected_left, expected_left + 1))
                    if expected_right is not None:
                        required.update((expected_right - 1, expected_right))
                    if not required.issubset(query_indices):
                        raise AssertionError("Batch certificate omits a boundary query.")

                    contribution = 0
                    for column in expected_columns:
                        if column in accepted_columns:
                            raise AssertionError("A daughter was accepted twice.")
                        accepted_columns.add(column)
                        child = tuple(
                            state_rows[index] - column[index]
                            for index in range(3)
                        )
                        contribution += (
                            multiplicity
                            * prefix
                            * _multinomial(column)
                            * _multinomial(child)
                        )
                    if _integer(batch["accepted_weight"], "accepted_weight") != contribution:
                        raise AssertionError("Batch contribution is corrupted.")
                    accepted_total += contribution

                for column in _column_vectors(remaining[0], state_rows):
                    if column in accepted_columns:
                        continue
                    child = tuple(
                        state_rows[index] - column[index]
                        for index in range(3)
                    )
                    next_frontier[child][prefix * _multinomial(column)] += multiplicity
        frontier = dict(next_frontier)

    if frontier:
        raise AssertionError("Replay frontier was not exhausted.")
    if len(used_states) != len(state_map):
        raise AssertionError("Certificate has an extra state-decision record.")
    if len(used_batches) != len(raw_batches):
        raise AssertionError("Certificate has an extra or duplicate batch record.")
    if accepted_total != _integer(data["accepted_weight"], "accepted_weight"):
        raise AssertionError("Certificate total accepted mass is corrupted.")

    p_value = Fraction(accepted_total, denominator)
    raw_p = data["p_value"]
    if not isinstance(raw_p, dict):
        raise AssertionError("Malformed p-value record.")
    recorded_p = Fraction(
        _integer(raw_p["numerator"], "p-value numerator"),
        _integer(raw_p["denominator"], "p-value denominator"),
    )
    if p_value != recorded_p:
        raise AssertionError("Certificate p-value is corrupted.")
    return p_value


def verify_certificate(path: str | Path) -> Fraction:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError("Certificate root must be an object.")
    return verify_certificate_data(data)
