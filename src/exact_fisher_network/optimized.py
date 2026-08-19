"""Exact state-bound network with optional three-row line batching."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Protocol, Sequence

from .extrema import ExactExtrema, Extremum, column_vectors, multinomial
from .flow_extrema import ExactFlowExtrema


@dataclass(frozen=True)
class BoundQuery:
    index: int
    column: tuple[int, ...]
    branch_longest: int
    maximizing_columns: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BatchRecord:
    layer: int
    residual_rows: tuple[int, ...]
    remaining_columns: tuple[int, ...]
    prefix_weight: int
    prefix_multiplicity: int
    fixed_third: int
    fixed_tail: tuple[int, ...]
    line_columns: tuple[tuple[int, ...], ...]
    queries: tuple[BoundQuery, ...]
    peak_index: int
    accepted_left_end: int | None
    accepted_right_start: int | None
    accepted_columns: tuple[tuple[int, ...], ...]
    accepted_weight: int


@dataclass(frozen=True)
class StateDecisionRecord:
    layer: int
    residual_rows: tuple[int, ...]
    remaining_columns: tuple[int, ...]
    prefix_weight: int
    prefix_multiplicity: int
    longest_completion: int
    longest_columns: tuple[tuple[int, ...], ...]
    shortest_completion: int | None
    shortest_columns: tuple[tuple[int, ...], ...] | None
    decision: str
    accepted_weight: int


@dataclass(frozen=True)
class EngineMetrics:
    residual_states_visited: int
    prefix_state_pairs_checked: int
    pruning_comparisons: int
    arcs_considered: int
    arcs_forwarded: int
    state_accept_all: int
    state_reject_all: int
    state_expand: int
    batched_arcs_accepted: int
    batch_lines_searched: int
    batch_longest_lookups: int
    prefix_batches: int
    unique_daughters_generated: int
    unique_state_line_pairs: int
    repeated_line_searches_avoided: int
    interval_mass_terms: int
    dense_line_searches: int
    sparse_line_searches: int
    monotone_peak_reuses: int
    monotone_peak_steps: int
    longest_state_problems_solved: int
    shortest_state_problems_solved: int
    longest_internal_subproblems: int
    shortest_internal_subproblems: int
    flow_fresh_solves: int
    flow_neighbor_augments: int
    flow_neighbor_fallback_cycles: int
    flow_cycle_augments: int
    flow_cycle_candidates_checked: int
    flow_path_candidates_checked: int
    maximum_prefix_weight_bits: int
    maximum_prefix_multiplicity_bits: int
    maximum_bound_weight_bits: int
    accepted_weight_bits: int
    denominator_bits: int


@dataclass(frozen=True)
class EngineResult:
    observed_table: tuple[tuple[int, ...], ...]
    row_margins: tuple[int, ...]
    column_margins: tuple[int, ...]
    observed_weight: int
    denominator: int
    accepted_weight: int
    p_value: Fraction
    batching: bool
    extrema_cache: str
    reject_pruning: bool
    state_records: tuple[StateDecisionRecord, ...]
    batch_records: tuple[BatchRecord, ...]
    metrics: EngineMetrics


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


def _observed_weight(table: tuple[tuple[int, ...], ...]) -> int:
    result = 1
    for column_index in range(len(table[0])):
        result *= multinomial(
            tuple(table[row][column_index] for row in range(len(table)))
        )
    return result


class _IsolatedExtremaCache:
    """Cache queried states while keeping their recurrence work isolated.

    A production implementation can replace each top-level query by the
    exact flow optimizer of Suzuki--Aoki--Murota.  Isolation prevents a
    query at a parent from silently precomputing every daughter's externally
    visible bound, which would erase the batching ablation.
    """

    def __init__(self) -> None:
        self._longest: dict[tuple[tuple[int, ...], tuple[int, ...]], Extremum] = {}
        self._shortest: dict[tuple[tuple[int, ...], tuple[int, ...]], Extremum] = {}
        self.longest_internal_subproblems = 0
        self.shortest_internal_subproblems = 0

    def longest(
        self,
        rows: tuple[int, ...],
        columns: tuple[int, ...],
    ) -> Extremum:
        key = (rows, columns)
        if key not in self._longest:
            solver = ExactExtrema()
            self._longest[key] = solver.longest(rows, columns)
            self.longest_internal_subproblems += solver.longest_problems_solved
        return self._longest[key]

    def shortest(
        self,
        rows: tuple[int, ...],
        columns: tuple[int, ...],
    ) -> Extremum:
        key = (rows, columns)
        if key not in self._shortest:
            solver = ExactExtrema()
            self._shortest[key] = solver.shortest(rows, columns)
            self.shortest_internal_subproblems += solver.shortest_problems_solved
        return self._shortest[key]

    @property
    def longest_problems_solved(self) -> int:
        return len(self._longest)

    @property
    def shortest_problems_solved(self) -> int:
        return len(self._shortest)


class _ExtremaCache(Protocol):
    longest_internal_subproblems: int
    shortest_internal_subproblems: int

    def longest(
        self,
        rows: tuple[int, ...],
        columns: tuple[int, ...],
    ) -> Extremum: ...

    def shortest(
        self,
        rows: tuple[int, ...],
        columns: tuple[int, ...],
    ) -> Extremum: ...

    @property
    def longest_problems_solved(self) -> int: ...

    @property
    def shortest_problems_solved(self) -> int: ...


class _GlobalExtremaCache:
    """Share each exact recurrence across every queried residual state."""

    def __init__(self) -> None:
        self._solver = ExactExtrema()
        self._longest_queries: set[
            tuple[tuple[int, ...], tuple[int, ...]]
        ] = set()
        self._shortest_queries: set[
            tuple[tuple[int, ...], tuple[int, ...]]
        ] = set()

    def longest(
        self,
        rows: tuple[int, ...],
        columns: tuple[int, ...],
    ) -> Extremum:
        self._longest_queries.add((rows, columns))
        return self._solver.longest(rows, columns)

    def shortest(
        self,
        rows: tuple[int, ...],
        columns: tuple[int, ...],
    ) -> Extremum:
        self._shortest_queries.add((rows, columns))
        return self._solver.shortest(rows, columns)

    @property
    def longest_problems_solved(self) -> int:
        return len(self._longest_queries)

    @property
    def shortest_problems_solved(self) -> int:
        return len(self._shortest_queries)

    @property
    def longest_internal_subproblems(self) -> int:
        return self._solver.longest_problems_solved

    @property
    def shortest_internal_subproblems(self) -> int:
        return self._solver.shortest_problems_solved


class _FlowExtremaCache:
    """Use exact convex transportation for longest completions."""

    def __init__(self) -> None:
        self._longest_solver = ExactFlowExtrema()
        self._shortest_solver = ExactExtrema()
        self._longest_queries: set[
            tuple[tuple[int, ...], tuple[int, ...]]
        ] = set()
        self._shortest_queries: set[
            tuple[tuple[int, ...], tuple[int, ...]]
        ] = set()

    def longest(
        self,
        rows: tuple[int, ...],
        columns: tuple[int, ...],
    ) -> Extremum:
        self._longest_queries.add((rows, columns))
        if not columns:
            if any(rows):
                raise AssertionError("Invalid terminal flow state.")
            return Extremum(1, ())
        return self._longest_solver.longest(rows, columns)

    def shortest(
        self,
        rows: tuple[int, ...],
        columns: tuple[int, ...],
    ) -> Extremum:
        self._shortest_queries.add((rows, columns))
        return self._shortest_solver.shortest(rows, columns)

    @property
    def longest_problems_solved(self) -> int:
        return len(self._longest_queries)

    @property
    def shortest_problems_solved(self) -> int:
        return len(self._shortest_queries)

    @property
    def longest_internal_subproblems(self) -> int:
        return self._longest_solver.problems_solved

    @property
    def shortest_internal_subproblems(self) -> int:
        return self._shortest_solver.shortest_problems_solved

    @property
    def flow_fresh_solves(self) -> int:
        return self._longest_solver.fresh_solves

    @property
    def flow_neighbor_augments(self) -> int:
        return self._longest_solver.neighbor_augments

    @property
    def flow_neighbor_fallback_cycles(self) -> int:
        return self._longest_solver.neighbor_fallback_cycles

    @property
    def flow_cycle_augments(self) -> int:
        return self._longest_solver.cycle_augments

    @property
    def flow_cycle_candidates_checked(self) -> int:
        return self._longest_solver.cycle_candidates_checked

    @property
    def flow_path_candidates_checked(self) -> int:
        return self._longest_solver.path_candidates_checked


class _ReusableDaughterLine:
    """One daughter line shared by every expanding prefix at a state."""

    def __init__(
        self,
        *,
        rows: tuple[int, ...],
        remaining: tuple[int, ...],
        columns: tuple[tuple[int, ...], ...],
        bounds: _ExtremaCache,
        search_mode: Literal["dense", "sparse"],
    ) -> None:
        self.rows = rows
        self.remaining = remaining
        self.columns = columns
        self.bounds = bounds
        self.search_mode = search_mode
        self._children: dict[int, tuple[int, ...]] = {}
        self._column_weights: dict[int, int] = {}
        self.queried: dict[int, tuple[int, Extremum]] = {}
        self.peak_index: int | None = None
        self.boundary_cache: dict[int, tuple[int | None, int | None]] = {}
        self.comparisons = 0
        self._mass_prefix: tuple[int, ...] | None = None
        self._dense_values: tuple[int, ...] | None = None
        self._dense_reversed_right: tuple[int, ...] | None = None
        self.monotone_steps = 0

    @property
    def longest_lookups(self) -> int:
        return len(self.queried)

    def value(self, index: int) -> int:
        if index not in self.queried:
            suffix = self.bounds.longest(
                self.child(index),
                self.remaining[1:],
            )
            self.queried[index] = (
                self.column_weight(index) * suffix.weight,
                suffix,
            )
        return self.queried[index][0]

    def child(self, index: int) -> tuple[int, ...]:
        if index not in self._children:
            column = self.columns[index]
            self._children[index] = tuple(
                self.rows[row] - column[row]
                for row in range(len(self.rows))
            )
        return self._children[index]

    def column_weight(self, index: int) -> int:
        if index not in self._column_weights:
            self._column_weights[index] = multinomial(self.columns[index])
        return self._column_weights[index]

    def peak(self) -> int:
        if self.peak_index is None:
            low = 0
            high = len(self.columns) - 1
            while low < high:
                middle = (low + high) // 2
                self.comparisons += 1
                if self.value(middle) < self.value(middle + 1):
                    low = middle + 1
                else:
                    high = middle
            self.peak_index = low
        return self.peak_index

    def locate_peak_at_or_left(self, upper_first_coordinate: int) -> int:
        """Locate this peak using the previous slice peak as an upper bound."""

        if self.peak_index is not None:
            return self.peak_index
        first = self.columns[0][0]
        last = self.columns[-1][0]
        start = min(upper_first_coordinate, last)
        if start < first:
            raise AssertionError("Monotone peak bound falls left of the line domain.")
        index = start - first
        while index > 0:
            self.comparisons += 1
            self.monotone_steps += 1
            if self.value(index - 1) >= self.value(index):
                index -= 1
            else:
                break
        self.peak_index = index
        return index

    def boundaries(
        self,
        prefix_weight: int,
        threshold: int,
    ) -> tuple[int | None, int | None]:
        cached = self.boundary_cache.get(prefix_weight)
        if cached is not None:
            return cached
        if self.search_mode == "dense":
            return self._dense_boundaries(prefix_weight, threshold)

        peak = self.peak()
        line_size = len(self.columns)
        left_end: int | None = None
        right_start: int | None = None
        self.comparisons += 1
        if prefix_weight * self.value(peak) <= threshold:
            left_end = line_size - 1
        else:
            low = 0
            high = peak
            while low < high:
                middle = (low + high) // 2
                self.comparisons += 1
                if prefix_weight * self.value(middle) <= threshold:
                    low = middle + 1
                else:
                    high = middle
            if low > 0:
                left_end = low - 1

            low = peak + 1
            high = line_size
            while low < high:
                middle = (low + high) // 2
                self.comparisons += 1
                if prefix_weight * self.value(middle) <= threshold:
                    high = middle
                else:
                    low = middle + 1
            if low < line_size:
                right_start = low

        result = (left_end, right_start)
        self.boundary_cache[prefix_weight] = result
        return result

    def _dense_boundaries(
        self,
        prefix_weight: int,
        threshold: int,
    ) -> tuple[int | None, int | None]:
        if self._dense_values is None:
            self._dense_values = tuple(
                self.value(index) for index in range(len(self.columns))
            )
            self.peak_index = max(
                range(len(self.columns)),
                key=self._dense_values.__getitem__,
            )
            self._dense_reversed_right = tuple(
                reversed(self._dense_values[self.peak_index + 1 :])
            )
            self.comparisons += len(self.columns) - 1
        values = self._dense_values
        peak = self.peak()
        cutoff = threshold // prefix_weight
        self.comparisons += 1
        if values[peak] <= cutoff:
            result = (len(values) - 1, None)
        else:
            left_count = bisect_right(values, cutoff, 0, peak)
            if self._dense_reversed_right is None:
                raise AssertionError("Dense right branch was not initialized.")
            right_count = bisect_right(self._dense_reversed_right, cutoff)
            result = (
                left_count - 1 if left_count else None,
                len(values) - right_count if right_count else None,
            )
        self.boundary_cache[prefix_weight] = result
        return result

    def accepted_mass(
        self,
        left_end: int | None,
        right_start: int | None,
    ) -> int:
        mass_prefix = self._get_mass_prefix()
        if left_end == len(self.columns) - 1:
            return mass_prefix[-1]
        result = 0
        if left_end is not None:
            result += mass_prefix[left_end + 1]
        if right_start is not None:
            result += mass_prefix[-1] - mass_prefix[right_start]
        return result

    @property
    def mass_terms(self) -> int:
        return 0 if self._mass_prefix is None else len(self.columns)

    def middle_range(
        self,
        left_end: int | None,
        right_start: int | None,
    ) -> range:
        if left_end == len(self.columns) - 1:
            return range(0)
        start = 0 if left_end is None else left_end + 1
        stop = len(self.columns) if right_start is None else right_start
        return range(start, stop)

    def certificate_queries(
        self,
        left_end: int | None,
        right_start: int | None,
    ) -> tuple[BoundQuery, ...]:
        peak = self.peak()
        required = {peak}
        if peak > 0:
            required.add(peak - 1)
        if peak + 1 < len(self.columns):
            required.add(peak + 1)
        if left_end is None:
            required.add(0)
        elif left_end < len(self.columns) - 1:
            required.update((left_end, left_end + 1))
        if right_start is None:
            required.add(len(self.columns) - 1)
        else:
            required.update((right_start - 1, right_start))
        for index in required:
            self.value(index)
        return tuple(
            BoundQuery(
                index=index,
                column=self.columns[index],
                branch_longest=self.queried[index][0],
                maximizing_columns=(
                    self.columns[index],
                    *self.queried[index][1].columns,
                ),
            )
            for index in sorted(required)
        )

    def _get_mass_prefix(self) -> tuple[int, ...]:
        if self._mass_prefix is not None:
            return self._mass_prefix
        mass = self.column_weight(0) * multinomial(self.child(0))
        prefix = [0, mass]
        for index in range(len(self.columns) - 1):
            column = self.columns[index]
            following = self.columns[index + 1]
            if not (
                following[0] == column[0] + 1
                and following[1] == column[1] - 1
                and following[2:] == column[2:]
            ):
                raise AssertionError("A daughter line is not a unit-transfer line.")
            numerator = (self.rows[0] - column[0]) * column[1]
            denominator = (
                (column[0] + 1)
                * (self.rows[1] - column[1] + 1)
            )
            product = mass * numerator
            if denominator <= 0 or product % denominator != 0:
                raise AssertionError("Exact daughter-mass recurrence is not integral.")
            mass = product // denominator
            prefix.append(prefix[-1] + mass)
        self._mass_prefix = tuple(prefix)
        return self._mass_prefix


def _right_boundary(
    sorted_weights: tuple[int, ...],
    limit: int,
) -> tuple[int, int]:
    """Return ``bisect_right`` and its exact comparison count."""

    low = 0
    high = len(sorted_weights)
    comparisons = 0
    while low < high:
        middle = (low + high) // 2
        comparisons += 1
        if sorted_weights[middle] <= limit:
            low = middle + 1
        else:
            high = middle
    return low, comparisons


def _daughter_planes(
    rows: tuple[int, ...],
    first_total: int,
) -> tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]:
    """Group transfer lines into planes with coordinates 4..r fixed."""

    grouped: dict[
        tuple[int, ...],
        dict[int, list[tuple[int, ...]]],
    ] = defaultdict(lambda: defaultdict(list))
    for column in column_vectors(first_total, rows):
        grouped[column[3:]][column[2]].append(column)
    return tuple(
        tuple(
            tuple(
                sorted(
                    grouped[fixed_other][third],
                    key=lambda item: item[0],
                )
            )
            for third in sorted(grouped[fixed_other])
        )
        for fixed_other in sorted(grouped)
    )


def exact_state_network(
    observed: Sequence[Sequence[int]],
    *,
    batching: bool,
    extrema_cache: Literal["isolated", "global", "flow"] = "isolated",
    line_search: Literal["auto", "dense", "sparse"] = "auto",
    monotone_peaks: bool = True,
    reject_pruning: bool = True,
    collect_certificate: bool = True,
) -> EngineResult:
    """Run the exact network with or without the daughter-line theorem."""

    table = _validate_observed(observed)
    row_margins = tuple(sum(row) for row in table)
    column_margins = tuple(
        sum(table[row][column] for row in range(len(table)))
        for column in range(len(table[0]))
    )
    threshold = _observed_weight(table)
    denominator = multinomial(row_margins)
    if extrema_cache == "isolated":
        bounds: _ExtremaCache = _IsolatedExtremaCache()
    elif extrema_cache == "global":
        bounds = _GlobalExtremaCache()
    elif extrema_cache == "flow":
        bounds = _FlowExtremaCache()
    else:
        raise ValueError(
            "extrema_cache must be 'isolated', 'global', or 'flow'."
        )
    if line_search not in ("auto", "dense", "sparse"):
        raise ValueError("line_search must be 'auto', 'dense', or 'sparse'.")

    frontier: dict[tuple[int, ...], Counter[int]] = {
        row_margins: Counter({1: 1})
    }
    accepted_weight = 0
    batch_records: list[BatchRecord] = []
    state_records: list[StateDecisionRecord] = []
    visited_states: set[tuple[int, tuple[int, ...]]] = set()
    prefix_state_pairs = 0
    pruning_comparisons = 0
    arcs_considered = 0
    arcs_forwarded = 0
    state_accept_all = 0
    state_reject_all = 0
    state_expand = 0
    batched_arcs = 0
    batch_lines = 0
    batch_lookups = 0
    prefix_batches = 0
    unique_daughters = 0
    state_line_pairs = 0
    repeated_line_searches_avoided = 0
    interval_mass_terms = 0
    dense_line_searches = 0
    sparse_line_searches = 0
    monotone_peak_reuses = 0
    monotone_peak_steps = 0
    maximum_prefix_weight_bits = 1
    maximum_prefix_multiplicity_bits = 1
    maximum_bound_weight_bits = 1

    for layer in range(len(column_margins) + 1):
        remaining = column_margins[layer:]
        next_frontier: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for rows in sorted(frontier):
            visited_states.add((layer, rows))
            longest_extremum = bounds.longest(rows, remaining)
            longest = longest_extremum.weight
            entries = tuple(sorted(frontier[rows].items()))
            weights = tuple(prefix for prefix, _ in entries)
            maximum_prefix_weight_bits = max(
                maximum_prefix_weight_bits,
                *(prefix.bit_length() for prefix, _ in entries),
            )
            maximum_prefix_multiplicity_bits = max(
                maximum_prefix_multiplicity_bits,
                *(multiplicity.bit_length() for _, multiplicity in entries),
            )
            maximum_bound_weight_bits = max(
                maximum_bound_weight_bits,
                longest.bit_length(),
            )
            prefix_state_pairs += len(entries)
            prefix_batches += 1

            accept_end, comparisons = _right_boundary(
                weights,
                threshold // longest,
            )
            pruning_comparisons += comparisons
            shortest_extremum: Extremum | None = None
            nonreject_end = accept_end
            if accept_end < len(entries):
                if not remaining:
                    shortest_extremum = longest_extremum
                    nonreject_end = accept_end
                elif reject_pruning:
                    shortest_extremum = bounds.shortest(rows, remaining)
                    nonreject_end, comparisons = _right_boundary(
                        weights,
                        threshold // shortest_extremum.weight,
                    )
                    pruning_comparisons += comparisons
                else:
                    nonreject_end, comparisons = _right_boundary(
                        weights,
                        threshold,
                    )
                    pruning_comparisons += comparisons
            if nonreject_end < accept_end:
                raise AssertionError("State-bound prefix intervals overlap.")

            accepted_entries = entries[:accept_end]
            expanding_entries = entries[accept_end:nonreject_end]
            rejected_entries = entries[nonreject_end:]
            state_accept_all += len(accepted_entries)
            state_expand += len(expanding_entries)
            state_reject_all += len(rejected_entries)

            state_mass = multinomial(rows)
            accepted_weight += state_mass * sum(
                prefix_weight * prefix_multiplicity
                for prefix_weight, prefix_multiplicity in accepted_entries
            )

            if collect_certificate:
                for prefix_weight, prefix_multiplicity in accepted_entries:
                    contribution = (
                        prefix_multiplicity
                        * prefix_weight
                        * state_mass
                    )
                    state_records.append(
                        StateDecisionRecord(
                            layer=layer,
                            residual_rows=rows,
                            remaining_columns=remaining,
                            prefix_weight=prefix_weight,
                            prefix_multiplicity=prefix_multiplicity,
                            longest_completion=longest,
                            longest_columns=longest_extremum.columns,
                            shortest_completion=None,
                            shortest_columns=None,
                            decision="accept_all",
                            accepted_weight=contribution,
                        )
                    )
                if (
                    shortest_extremum is None
                    and rejected_entries
                    and reject_pruning
                ):
                    raise AssertionError("Missing shortest state extremum.")
                for decision, decision_entries in (
                    ("expand", expanding_entries),
                    ("reject_all", rejected_entries),
                ):
                    for prefix_weight, prefix_multiplicity in decision_entries:
                        state_records.append(
                            StateDecisionRecord(
                                layer=layer,
                                residual_rows=rows,
                                remaining_columns=remaining,
                                prefix_weight=prefix_weight,
                                prefix_multiplicity=prefix_multiplicity,
                                longest_completion=longest,
                                longest_columns=longest_extremum.columns,
                                shortest_completion=(
                                    shortest_extremum.weight
                                    if shortest_extremum is not None
                                    else None
                                ),
                                shortest_columns=(
                                    shortest_extremum.columns
                                    if shortest_extremum is not None
                                    else None
                                ),
                                decision=decision,
                                accepted_weight=0,
                            )
                        )

            if not expanding_entries:
                continue
            if not remaining:
                raise AssertionError("A terminal prefix was not classified.")

            if batching and len(rows) >= 3 and len(remaining) >= 2:
                typed_rows = rows
                planes = _daughter_planes(typed_rows, remaining[0])
                lines = tuple(
                    line
                    for plane in planes
                    for line in plane
                )
                unique_daughters += sum(len(line) for line in lines)
                previous_peak_first: int | None = None
                previous_plane: tuple[int, ...] | None = None
                for line_columns in lines:
                    plane = line_columns[0][3:]
                    if plane != previous_plane:
                        previous_peak_first = None
                        previous_plane = plane
                    arcs_considered += len(line_columns) * len(expanding_entries)
                    if len(line_columns) < 4:
                        prepared = tuple(
                            (
                                column,
                                tuple(
                                    rows[index] - column[index]
                                    for index in range(len(rows))
                                ),
                                multinomial(column),
                            )
                            for column in line_columns
                        )
                        for prefix_weight, prefix_multiplicity in expanding_entries:
                            for _, child, column_weight in prepared:
                                arcs_forwarded += 1
                                next_frontier[child][
                                    prefix_weight * column_weight
                                ] += prefix_multiplicity
                        continue

                    if line_search == "auto":
                        sparse_proxy = (
                            1 + 2 * len(expanding_entries)
                        ) * len(line_columns).bit_length()
                        selected_search = (
                            "dense"
                            if (
                                extrema_cache == "global"
                                and len(line_columns) <= sparse_proxy
                            )
                            else "sparse"
                        )
                    else:
                        selected_search = line_search
                    reusable = _ReusableDaughterLine(
                        rows=typed_rows,
                        remaining=remaining,
                        columns=line_columns,
                        bounds=bounds,
                        search_mode=selected_search,
                    )
                    state_line_pairs += 1
                    smallest_prefix = expanding_entries[0][0]
                    pruning_comparisons += 2
                    if (
                        smallest_prefix * reusable.value(0) > threshold
                        and smallest_prefix * reusable.value(
                            len(line_columns) - 1
                        ) > threshold
                    ):
                        for prefix_weight, prefix_multiplicity in expanding_entries:
                            for index in range(len(line_columns)):
                                arcs_forwarded += 1
                                next_frontier[reusable.child(index)][
                                    prefix_weight
                                    * reusable.column_weight(index)
                                ] += prefix_multiplicity
                        batch_lookups += reusable.longest_lookups
                        continue

                    batch_lines += 1
                    if selected_search == "dense":
                        dense_line_searches += 1
                    else:
                        sparse_line_searches += 1
                        if monotone_peaks and previous_peak_first is not None:
                            reusable.locate_peak_at_or_left(
                                previous_peak_first
                            )
                            monotone_peak_reuses += 1
                        else:
                            reusable.peak()
                        previous_peak_first = line_columns[
                            reusable.peak()
                        ][0]
                    repeated_line_searches_avoided += max(
                        0,
                        len(expanding_entries) - 1,
                    )
                    for prefix_weight, prefix_multiplicity in expanding_entries:
                        left_end, right_start = reusable.boundaries(
                            prefix_weight,
                            threshold,
                        )
                        middle = reusable.middle_range(left_end, right_start)
                        accepted_count = len(line_columns) - len(middle)
                        if accepted_count:
                            batched_arcs += accepted_count
                            contribution = (
                                prefix_multiplicity
                                * prefix_weight
                                * reusable.accepted_mass(
                                    left_end,
                                    right_start,
                                )
                            )
                            accepted_weight += contribution
                            if collect_certificate:
                                if left_end == len(line_columns) - 1:
                                    accepted_columns = line_columns
                                else:
                                    accepted_columns = (
                                        line_columns[
                                            : 0 if left_end is None else left_end + 1
                                        ]
                                        + line_columns[
                                            len(line_columns)
                                            if right_start is None
                                            else right_start :
                                        ]
                                    )
                                batch_records.append(
                                    BatchRecord(
                                        layer=layer,
                                        residual_rows=typed_rows,
                                        remaining_columns=remaining,
                                        prefix_weight=prefix_weight,
                                        prefix_multiplicity=prefix_multiplicity,
                                        fixed_third=line_columns[0][2],
                                        fixed_tail=line_columns[0][2:],
                                        line_columns=line_columns,
                                        queries=reusable.certificate_queries(
                                            left_end,
                                            right_start,
                                        ),
                                        peak_index=reusable.peak(),
                                        accepted_left_end=left_end,
                                        accepted_right_start=right_start,
                                        accepted_columns=accepted_columns,
                                        accepted_weight=contribution,
                                    )
                                )
                        for index in middle:
                            arcs_forwarded += 1
                            next_frontier[reusable.child(index)][
                                prefix_weight
                                * reusable.column_weight(index)
                            ] += prefix_multiplicity
                    batch_lookups += reusable.longest_lookups
                    pruning_comparisons += reusable.comparisons
                    interval_mass_terms += reusable.mass_terms
                    monotone_peak_steps += reusable.monotone_steps
            else:
                prepared = tuple(
                    (
                        column,
                        tuple(
                            rows[index] - column[index]
                            for index in range(len(rows))
                        ),
                        multinomial(column),
                    )
                    for column in column_vectors(remaining[0], rows)
                )
                unique_daughters += len(prepared)
                arcs_considered += len(prepared) * len(expanding_entries)
                for prefix_weight, prefix_multiplicity in expanding_entries:
                    for _, child, column_weight in prepared:
                        arcs_forwarded += 1
                        next_frontier[child][
                            prefix_weight * column_weight
                        ] += prefix_multiplicity
        frontier = dict(next_frontier)

    if frontier:
        raise AssertionError("The terminal frontier was not exhausted.")
    if accepted_weight > denominator:
        raise AssertionError("Accepted mass exceeds the conditional denominator.")

    return EngineResult(
        observed_table=table,
        row_margins=row_margins,
        column_margins=column_margins,
        observed_weight=threshold,
        denominator=denominator,
        accepted_weight=accepted_weight,
        p_value=Fraction(accepted_weight, denominator),
        batching=batching,
        extrema_cache=extrema_cache,
        reject_pruning=reject_pruning,
        state_records=tuple(state_records),
        batch_records=tuple(batch_records),
        metrics=EngineMetrics(
            residual_states_visited=len(visited_states),
            prefix_state_pairs_checked=prefix_state_pairs,
            pruning_comparisons=pruning_comparisons,
            arcs_considered=arcs_considered,
            arcs_forwarded=arcs_forwarded,
            state_accept_all=state_accept_all,
            state_reject_all=state_reject_all,
            state_expand=state_expand,
            batched_arcs_accepted=batched_arcs,
            batch_lines_searched=batch_lines,
            batch_longest_lookups=batch_lookups,
            prefix_batches=prefix_batches,
            unique_daughters_generated=unique_daughters,
            unique_state_line_pairs=state_line_pairs,
            repeated_line_searches_avoided=repeated_line_searches_avoided,
            interval_mass_terms=interval_mass_terms,
            dense_line_searches=dense_line_searches,
            sparse_line_searches=sparse_line_searches,
            monotone_peak_reuses=monotone_peak_reuses,
            monotone_peak_steps=monotone_peak_steps,
            longest_state_problems_solved=bounds.longest_problems_solved,
            shortest_state_problems_solved=bounds.shortest_problems_solved,
            longest_internal_subproblems=bounds.longest_internal_subproblems,
            shortest_internal_subproblems=bounds.shortest_internal_subproblems,
            flow_fresh_solves=getattr(bounds, "flow_fresh_solves", 0),
            flow_neighbor_augments=getattr(bounds, "flow_neighbor_augments", 0),
            flow_neighbor_fallback_cycles=getattr(
                bounds,
                "flow_neighbor_fallback_cycles",
                0,
            ),
            flow_cycle_augments=getattr(bounds, "flow_cycle_augments", 0),
            flow_cycle_candidates_checked=getattr(
                bounds,
                "flow_cycle_candidates_checked",
                0,
            ),
            flow_path_candidates_checked=getattr(
                bounds,
                "flow_path_candidates_checked",
                0,
            ),
            maximum_prefix_weight_bits=maximum_prefix_weight_bits,
            maximum_prefix_multiplicity_bits=(
                maximum_prefix_multiplicity_bits
            ),
            maximum_bound_weight_bits=maximum_bound_weight_bits,
            accepted_weight_bits=accepted_weight.bit_length(),
            denominator_bits=denominator.bit_length(),
        ),
    )
