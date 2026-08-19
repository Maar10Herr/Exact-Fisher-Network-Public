"""Independent verifier for compact exact-network certificates.

The verifier does not import the optimized engine or either extremum solver.
It checks primal witnesses and exact multiplicative flow duals.  It creates
only forwarded daughters.  Accepted daughter intervals are audited by scalar
mass recurrence and boundary inequalities.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from fractions import Fraction
from math import factorial
from pathlib import Path
from typing import Iterator, Sequence


def _multinomial(parts: Sequence[int]) -> int:
    result = factorial(sum(parts))
    for value in parts:
        result //= factorial(value)
    return result


def _column_vectors(
    total: int,
    upper: tuple[int, ...],
) -> Iterator[tuple[int, ...]]:
    if not upper:
        if total == 0:
            yield ()
        return
    if len(upper) == 1:
        if total <= upper[0]:
            yield (total,)
        return
    lower = max(0, total - sum(upper[1:]))
    high = min(total, upper[0])
    for first in range(lower, high + 1):
        for tail in _column_vectors(total - first, upper[1:]):
            yield (first, *tail)


def _tuple_vector(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise AssertionError("Expected a vector.")
    return tuple(int(item) for item in value)


def _tuple_matrix(value: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise AssertionError("Expected a matrix.")
    return tuple(_tuple_vector(row) for row in value)


def _fraction(value: object) -> Fraction:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise AssertionError("Expected an exact rational.")
    result = Fraction(int(value[0]), int(value[1]))
    if result <= 0:
        raise AssertionError("Flow scalars must be positive.")
    return result


def _observed_weight(table: tuple[tuple[int, ...], ...]) -> int:
    result = 1
    for column in range(len(table[0])):
        result *= _multinomial(
            tuple(table[row][column] for row in range(len(table)))
        )
    return result


def _verify_extremum(
    raw: object,
    *,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    expected_weight: int,
) -> None:
    if not isinstance(raw, dict):
        raise AssertionError("Malformed extremum certificate.")
    witness = _tuple_matrix(raw["columns"])
    if not columns:
        if witness or any(rows) or expected_weight != 1:
            raise AssertionError("Invalid terminal extremum.")
        if raw["row_scalars"] or raw["column_scalars"]:
            raise AssertionError("Terminal extremum has unexpected duals.")
        return
    if len(witness) != len(columns):
        raise AssertionError("Extremum has the wrong number of columns.")
    if any(len(column) != len(rows) for column in witness):
        raise AssertionError("Extremum has the wrong row count.")
    if tuple(sum(column) for column in witness) != columns:
        raise AssertionError("Extremum column margins are wrong.")
    if tuple(
        sum(column[row] for column in witness)
        for row in range(len(rows))
    ) != rows:
        raise AssertionError("Extremum row margins are wrong.")
    weight = 1
    for column in witness:
        weight *= _multinomial(column)
    if weight != expected_weight:
        raise AssertionError("Extremum witness has the wrong weight.")

    row_scalars = tuple(_fraction(value) for value in raw["row_scalars"])
    column_scalars = tuple(
        _fraction(value) for value in raw["column_scalars"]
    )
    if len(row_scalars) != len(rows):
        raise AssertionError("Flow dual has the wrong row count.")
    if len(column_scalars) != len(columns):
        raise AssertionError("Flow dual has the wrong column count.")
    for column_index, witness_column in enumerate(witness):
        for row, cell in enumerate(witness_column):
            fitted = row_scalars[row] / column_scalars[column_index]
            if fitted < cell or fitted > cell + 1:
                raise AssertionError("Flow dual violates a cell interval.")


def _state_key(raw: dict[str, object]) -> tuple[object, ...]:
    return (
        int(raw["layer"]),
        _tuple_vector(raw["residual_rows"]),
        _tuple_vector(raw["remaining_columns"]),
    )


def _batch_key(raw: dict[str, object]) -> tuple[object, ...]:
    return (
        *_state_key(raw),
        int(raw["prefix_weight"]),
        int(raw["prefix_multiplicity"]),
        _tuple_vector(raw["fixed_tail"]),
    )


def _line_bounds(
    rows: tuple[int, ...],
    first_total: int,
    tail: tuple[int, ...],
) -> tuple[int, int]:
    if len(tail) != len(rows) - 2:
        raise AssertionError("A line has the wrong fixed-tail dimension.")
    if any(tail[index] > rows[index + 2] for index in range(len(tail))):
        raise AssertionError("A line tail exceeds its residual margin.")
    pair_total = first_total - sum(tail)
    low = max(0, pair_total - rows[1])
    high = min(pair_total, rows[0])
    if pair_total < 0 or low > high:
        raise AssertionError("A recorded daughter line is infeasible.")
    return low, high


def _all_line_tails(
    rows: tuple[int, ...],
    first_total: int,
) -> Iterator[tuple[int, ...]]:
    upper = rows[2:]
    for tail_total in range(min(first_total, sum(upper)) + 1):
        for tail in _column_vectors(tail_total, upper):
            pair_total = first_total - tail_total
            if max(0, pair_total - rows[1]) <= min(pair_total, rows[0]):
                yield tail


def _column_at(
    first: int,
    first_total: int,
    tail: tuple[int, ...],
) -> tuple[int, ...]:
    return (
        first,
        first_total - sum(tail) - first,
        *tail,
    )


def _verify_query(
    raw: object,
    *,
    index: int,
    column: tuple[int, ...],
    rows: tuple[int, ...],
    remaining: tuple[int, ...],
    extrema: list[object],
    used_extrema: set[int],
) -> int:
    if not isinstance(raw, dict):
        raise AssertionError("Malformed boundary query.")
    if int(raw["index"]) != index:
        raise AssertionError("Boundary query index is inconsistent.")
    if _tuple_vector(raw["column"]) != column:
        raise AssertionError("Boundary query column is inconsistent.")
    branch = int(raw["branch_longest"])
    arc_weight = _multinomial(column)
    if branch % arc_weight:
        raise AssertionError("Branch bound is not divisible by its arc.")
    child = tuple(rows[row] - column[row] for row in range(len(rows)))
    extremum_id = int(raw["suffix_extremum_id"])
    if not 0 <= extremum_id < len(extrema):
        raise AssertionError("Suffix extremum reference is invalid.")
    used_extrema.add(extremum_id)
    _verify_extremum(
        extrema[extremum_id],
        rows=child,
        columns=remaining[1:],
        expected_weight=branch // arc_weight,
    )
    return branch


def verify_compact_certificate_data(data: dict[str, object]) -> Fraction:
    if data.get("schema") != "exact-fisher-compact-v1":
        raise AssertionError("Unknown compact-certificate schema.")
    if data.get("batching") is not True:
        raise AssertionError("Compact certificate does not enable batching.")
    if data.get("reject_pruning") is not False:
        raise AssertionError("Compact certificate must use accept-only pruning.")

    table = _tuple_matrix(data["observed_table"])
    if len(table) < 3 or not table or len(table[0]) < 2:
        raise AssertionError("Compact certificate needs an r x c table, r >= 3.")
    if any(len(row) != len(table[0]) for row in table):
        raise AssertionError("Observed table is ragged.")
    if any(cell < 0 for row in table for cell in row):
        raise AssertionError("Observed table has a negative cell.")
    rows = tuple(sum(row) for row in table)
    columns = tuple(
        sum(table[row][column] for row in range(len(rows)))
        for column in range(len(table[0]))
    )
    threshold = _observed_weight(table)
    denominator = _multinomial(rows)
    if _tuple_vector(data["row_margins"]) != rows:
        raise AssertionError("Row margins do not match.")
    if _tuple_vector(data["column_margins"]) != columns:
        raise AssertionError("Column margins do not match.")
    if int(data["observed_weight"]) != threshold:
        raise AssertionError("Observed weight is corrupted.")
    if int(data["denominator"]) != denominator:
        raise AssertionError("Denominator is corrupted.")

    raw_states = data["states"]
    raw_batches = data["batches"]
    raw_extrema = data["extrema"]
    if (
        not isinstance(raw_states, list)
        or not isinstance(raw_batches, list)
        or not isinstance(raw_extrema, list)
    ):
        raise AssertionError("Compact records must be lists.")
    state_map: dict[tuple[object, ...], dict[str, object]] = {}
    for raw in raw_states:
        if not isinstance(raw, dict):
            raise AssertionError("Malformed compact state.")
        key = _state_key(raw)
        if key in state_map:
            raise AssertionError("Duplicate compact state.")
        state_map[key] = raw
    batch_map: dict[tuple[object, ...], dict[str, object]] = {}
    for raw in raw_batches:
        if not isinstance(raw, dict):
            raise AssertionError("Malformed compact batch.")
        key = _batch_key(raw)
        if key in batch_map:
            raise AssertionError("Duplicate compact batch.")
        batch_map[key] = raw

    used_states: set[tuple[object, ...]] = set()
    used_batches: set[tuple[object, ...]] = set()
    used_extrema: set[int] = set()
    peak_sequences: dict[
        tuple[object, ...],
        dict[int, int],
    ] = defaultdict(dict)
    frontier: dict[tuple[int, ...], Counter[int]] = {
        rows: Counter({1: 1})
    }
    accepted_total = 0

    for layer in range(len(columns) + 1):
        remaining = columns[layer:]
        next_frontier: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for state_rows in sorted(frontier):
            state_key = (layer, state_rows, remaining)
            if state_key not in state_map:
                raise AssertionError("Missing compact state.")
            state = state_map[state_key]
            used_states.add(state_key)
            longest = int(state["longest_completion"])
            extremum_id = int(state["longest_extremum_id"])
            if not 0 <= extremum_id < len(raw_extrema):
                raise AssertionError("State extremum reference is invalid.")
            used_extrema.add(extremum_id)
            _verify_extremum(
                raw_extrema[extremum_id],
                rows=state_rows,
                columns=remaining,
                expected_weight=longest,
            )
            entries = tuple(sorted(frontier[state_rows].items()))
            accepted_entries = tuple(
                entry for entry in entries if entry[0] * longest <= threshold
            )
            expanding_entries = tuple(
                entry
                for entry in entries
                if entry[0] * longest > threshold
                and entry[0] <= threshold
            )
            state_contribution = _multinomial(state_rows) * sum(
                prefix * multiplicity
                for prefix, multiplicity in accepted_entries
            )
            if int(state["accepted_weight"]) != state_contribution:
                raise AssertionError("Compact state mass is corrupted.")
            accepted_total += state_contribution
            if not remaining:
                continue

            for prefix, multiplicity in expanding_entries:
                for tail in _all_line_tails(state_rows, remaining[0]):
                    low, high = _line_bounds(
                        state_rows,
                        remaining[0],
                        tail,
                    )
                    line_length = high - low + 1
                    key = (
                        layer,
                        state_rows,
                        remaining,
                        prefix,
                        multiplicity,
                        tail,
                    )
                    batch = batch_map.get(key)
                    if batch is None:
                        middle = range(line_length)
                    else:
                        used_batches.add(key)
                        if line_length < 4:
                            raise AssertionError("A short line was batched.")
                        if int(batch["line_length"]) != line_length:
                            raise AssertionError("Batch line length is corrupted.")
                        peak = int(batch["peak_index"])
                        if not 0 <= peak < line_length:
                            raise AssertionError("Batch peak is outside the line.")
                        raw_queries = batch["queries"]
                        if not isinstance(raw_queries, (list, tuple)):
                            raise AssertionError("Batch queries are malformed.")
                        query_values: dict[int, int] = {}
                        for raw_query in raw_queries:
                            if not isinstance(raw_query, dict):
                                raise AssertionError("Malformed batch query.")
                            index = int(raw_query["index"])
                            if index in query_values or not 0 <= index < line_length:
                                raise AssertionError("Duplicate or invalid query.")
                            column = _column_at(
                                low + index,
                                remaining[0],
                                tail,
                            )
                            query_values[index] = _verify_query(
                                raw_query,
                                index=index,
                                column=column,
                                rows=state_rows,
                                remaining=remaining,
                                extrema=raw_extrema,
                                used_extrema=used_extrema,
                            )
                        if peak not in query_values:
                            raise AssertionError("Peak query is missing.")
                        peak_value = query_values[peak]
                        if peak > 0:
                            if peak - 1 not in query_values:
                                raise AssertionError("Left peak neighbor is missing.")
                            if query_values[peak - 1] >= peak_value:
                                raise AssertionError("Peak is not the leftmost mode.")
                        if peak + 1 < line_length:
                            if peak + 1 not in query_values:
                                raise AssertionError("Right peak neighbor is missing.")
                            if query_values[peak + 1] > peak_value:
                                raise AssertionError("Peak direction is corrupted.")

                        left_raw = batch["accepted_left_end"]
                        right_raw = batch["accepted_right_start"]
                        left = None if left_raw is None else int(left_raw)
                        right = None if right_raw is None else int(right_raw)
                        if left == line_length - 1:
                            if right is not None:
                                raise AssertionError("All-accepted line has a right range.")
                            if prefix * peak_value > threshold:
                                raise AssertionError("All-accepted peak fails its bound.")
                            middle = range(0)
                        else:
                            if prefix * peak_value <= threshold:
                                raise AssertionError("Partial line has an accepted peak.")
                            if left is None:
                                if 0 not in query_values:
                                    raise AssertionError("Left endpoint query is missing.")
                                if prefix * query_values[0] <= threshold:
                                    raise AssertionError("Missing left interval is accepted.")
                                start = 0
                            else:
                                if not 0 <= left < peak:
                                    raise AssertionError("Left boundary is invalid.")
                                if not {left, left + 1}.issubset(query_values):
                                    raise AssertionError("Left boundary queries are missing.")
                                if prefix * query_values[left] > threshold:
                                    raise AssertionError("Left accepted boundary fails.")
                                if prefix * query_values[left + 1] <= threshold:
                                    raise AssertionError("Left rejected boundary fails.")
                                start = left + 1
                            if right is None:
                                last = line_length - 1
                                if last not in query_values:
                                    raise AssertionError("Right endpoint query is missing.")
                                if prefix * query_values[last] <= threshold:
                                    raise AssertionError("Missing right interval is accepted.")
                                stop = line_length
                            else:
                                if not peak < right < line_length:
                                    raise AssertionError("Right boundary is invalid.")
                                if not {right - 1, right}.issubset(query_values):
                                    raise AssertionError("Right boundary queries are missing.")
                                if prefix * query_values[right - 1] <= threshold:
                                    raise AssertionError("Right rejected boundary fails.")
                                if prefix * query_values[right] > threshold:
                                    raise AssertionError("Right accepted boundary fails.")
                                stop = right
                            middle = range(start, stop)

                        plane_key = (
                            layer,
                            state_rows,
                            remaining,
                            prefix,
                            multiplicity,
                            tail[1:],
                        )
                        peak_coordinate = low + peak
                        prior = peak_sequences[plane_key].get(tail[0])
                        if prior is not None and prior != peak_coordinate:
                            raise AssertionError("A slice has inconsistent peaks.")
                        peak_sequences[plane_key][tail[0]] = peak_coordinate

                        accepted_mass = 0
                        first_column = _column_at(low, remaining[0], tail)
                        first_child = tuple(
                            state_rows[row] - first_column[row]
                            for row in range(len(state_rows))
                        )
                        mass = (
                            _multinomial(first_column)
                            * _multinomial(first_child)
                        )
                        middle_set = set(middle)
                        for index in range(line_length):
                            if index not in middle_set:
                                accepted_mass += mass
                            if index + 1 < line_length:
                                column = _column_at(
                                    low + index,
                                    remaining[0],
                                    tail,
                                )
                                numerator = (
                                    state_rows[0] - column[0]
                                ) * column[1]
                                denominator_step = (
                                    (column[0] + 1)
                                    * (state_rows[1] - column[1] + 1)
                                )
                                product = mass * numerator
                                if (
                                    denominator_step <= 0
                                    or product % denominator_step
                                ):
                                    raise AssertionError(
                                        "Mass recurrence is not integral."
                                    )
                                mass = product // denominator_step
                        contribution = multiplicity * prefix * accepted_mass
                        if int(batch["accepted_weight"]) != contribution:
                            raise AssertionError("Compact interval mass is corrupted.")
                        accepted_total += contribution

                    for index in middle:
                        column = _column_at(
                            low + index,
                            remaining[0],
                            tail,
                        )
                        child = tuple(
                            state_rows[row] - column[row]
                            for row in range(len(state_rows))
                        )
                        next_frontier[child][
                            prefix * _multinomial(column)
                        ] += multiplicity
        frontier = dict(next_frontier)

    for sequence in peak_sequences.values():
        ordered = [sequence[slice_value] for slice_value in sorted(sequence)]
        if any(
            ordered[index + 1] > ordered[index]
            for index in range(len(ordered) - 1)
        ):
            raise AssertionError("Cross-slice peak order is corrupted.")
    if frontier:
        raise AssertionError("Terminal frontier was not exhausted.")
    if used_states != set(state_map):
        raise AssertionError("Compact certificate has an extra state.")
    if used_batches != set(batch_map):
        raise AssertionError("Compact certificate has an extra batch.")
    if used_extrema != set(range(len(raw_extrema))):
        raise AssertionError("Compact certificate has an extra extremum.")
    if accepted_total != int(data["accepted_weight"]):
        raise AssertionError("Compact accepted mass is corrupted.")
    p_value = Fraction(accepted_total, denominator)
    raw_p = data["p_value"]
    if not isinstance(raw_p, dict):
        raise AssertionError("Compact p-value is malformed.")
    recorded = Fraction(
        int(raw_p["numerator"]),
        int(raw_p["denominator"]),
    )
    if p_value != recorded:
        raise AssertionError("Compact p-value is corrupted.")
    return p_value


def verify_compact_certificate(path: str | Path) -> Fraction:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError("Compact certificate root must be an object.")
    return verify_compact_certificate_data(data)
