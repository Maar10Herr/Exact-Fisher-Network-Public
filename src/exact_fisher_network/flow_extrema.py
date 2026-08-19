"""Exact maximum-probability tables by convex transportation moves.

For fixed column margins, maximizing the conditional table weight is
equivalent to minimizing ``prod(factorial(cell))``.  A feasible table is
optimal exactly when no unit residual transportation cycle decreases that
integer product.  The implementation compares every move by ``Fraction``.
It never orders candidates with floating-point logarithms.

The neighboring-margin update follows the sensitivity step used by
Suzuki--Aoki--Murota: one minimum-ratio augmenting path changes row margins
by one unit.  A full residual-cycle check is retained as an exact audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
from math import factorial

from .extrema import Extremum, multinomial


Move = tuple[int, int, int]
Table = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FlowSolution:
    row_margins: tuple[int, ...]
    column_margins: tuple[int, ...]
    table: Table
    factorial_product: int
    weight: int

    @property
    def extremum(self) -> Extremum:
        columns = tuple(
            tuple(self.table[row][column] for row in range(len(self.table)))
            for column in range(len(self.column_margins))
        )
        return Extremum(self.weight, columns)


@dataclass(frozen=True)
class FlowPotential:
    """Exact multiplicative primal-dual potential."""

    row_scalars: tuple[Fraction, ...]
    column_scalars: tuple[Fraction, ...]


class ExactFlowExtrema:
    """Cache exact flow optima and reuse neighboring solutions."""

    def __init__(self) -> None:
        self._solutions: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            FlowSolution,
        ] = {}
        self.fresh_solves = 0
        self.neighbor_augments = 0
        self.neighbor_fallback_cycles = 0
        self.cycle_augments = 0
        self.cycle_candidates_checked = 0
        self.path_candidates_checked = 0

    @property
    def problems_solved(self) -> int:
        return len(self._solutions)

    def longest(
        self,
        row_margins: tuple[int, ...],
        column_margins: tuple[int, ...],
    ) -> Extremum:
        return self.solution(row_margins, column_margins).extremum

    def solution(
        self,
        row_margins: tuple[int, ...],
        column_margins: tuple[int, ...],
    ) -> FlowSolution:
        _validate_margins(row_margins, column_margins)
        key = (row_margins, column_margins)
        cached = self._solutions.get(key)
        if cached is not None:
            return cached

        neighbor = self._find_neighbor(row_margins, column_margins)
        if neighbor is None:
            table = _northwest_corner(row_margins, column_margins)
            self.fresh_solves += 1
            table, augmentations = self._cancel_improving_cycles(table)
            self.cycle_augments += augmentations
        else:
            old_rows, old_solution = neighbor
            table = self._augment_neighbor(
                old_solution.table,
                old_rows,
                row_margins,
            )
            self.neighbor_augments += 1
            table, augmentations = self._cancel_improving_cycles(table)
            self.neighbor_fallback_cycles += augmentations
            self.cycle_augments += augmentations

        solution = _make_solution(row_margins, column_margins, table)
        self._solutions[key] = solution
        return solution

    def verify(self, solution: FlowSolution) -> bool:
        """Check margins, exact weight, and the residual-cycle certificate."""

        if _row_sums(solution.table) != solution.row_margins:
            return False
        if _column_sums(solution.table) != solution.column_margins:
            return False
        rebuilt = _make_solution(
            solution.row_margins,
            solution.column_margins,
            solution.table,
        )
        if rebuilt != solution:
            return False
        ratio, _ = self._best_cycle(solution.table, count_metrics=False)
        return (
            (ratio is None or ratio >= 1)
            and verify_flow_potential(
                solution.table,
                exact_flow_potential(solution.table),
            )
        )

    def _find_neighbor(
        self,
        target_rows: tuple[int, ...],
        columns: tuple[int, ...],
    ) -> tuple[tuple[int, ...], FlowSolution] | None:
        for (rows, stored_columns), solution in reversed(
            tuple(self._solutions.items())
        ):
            if stored_columns != columns:
                continue
            differences = tuple(
                target_rows[index] - rows[index]
                for index in range(len(rows))
            )
            if differences.count(1) == 1 and differences.count(-1) == 1:
                if all(value in (-1, 0, 1) for value in differences):
                    return rows, solution
        return None

    def _cancel_improving_cycles(self, table: Table) -> tuple[Table, int]:
        augmentations = 0
        while True:
            ratio, moves = self._best_cycle(table, count_metrics=True)
            if ratio is None or ratio >= 1:
                return table, augmentations
            before = _factorial_product(table)
            table = _apply_moves(table, moves)
            after = _factorial_product(table)
            if after * ratio.denominator != before * ratio.numerator:
                raise AssertionError("Cycle ratio does not match the exact objective.")
            if after >= before:
                raise AssertionError("An improving cycle did not improve the objective.")
            augmentations += 1

    def _best_cycle(
        self,
        table: Table,
        *,
        count_metrics: bool,
    ) -> tuple[Fraction | None, tuple[Move, ...]]:
        row_count = len(table)
        column_count = len(table[0])
        best_ratio: Fraction | None = None
        best_moves: tuple[Move, ...] = ()
        for length in range(2, min(row_count, column_count) + 1):
            for chosen_rows in combinations(range(row_count), length):
                anchor = chosen_rows[0]
                for tail in permutations(chosen_rows[1:]):
                    cycle_rows = (anchor, *tail)
                    for cycle_columns in permutations(range(column_count), length):
                        moves = tuple(
                            (
                                cycle_rows[index],
                                cycle_rows[(index + 1) % length],
                                cycle_columns[index],
                            )
                            for index in range(length)
                        )
                        if any(
                            table[donor][column] == 0
                            for donor, _, column in moves
                        ):
                            continue
                        if count_metrics:
                            self.cycle_candidates_checked += 1
                        ratio = _move_ratio(table, moves)
                        if best_ratio is None or ratio < best_ratio:
                            best_ratio = ratio
                            best_moves = moves
        return best_ratio, best_moves

    def _augment_neighbor(
        self,
        table: Table,
        old_rows: tuple[int, ...],
        target_rows: tuple[int, ...],
    ) -> Table:
        differences = tuple(
            target_rows[index] - old_rows[index]
            for index in range(len(old_rows))
        )
        gain = differences.index(1)
        donor = differences.index(-1)
        other_rows = tuple(
            row
            for row in range(len(old_rows))
            if row not in (donor, gain)
        )

        best_ratio: Fraction | None = None
        best_moves: tuple[Move, ...] = ()
        for intermediate_count in range(len(other_rows) + 1):
            for intermediate in permutations(other_rows, intermediate_count):
                path_rows = (donor, *intermediate, gain)
                edge_count = len(path_rows) - 1
                if edge_count > len(table[0]):
                    continue
                for path_columns in permutations(range(len(table[0])), edge_count):
                    moves = tuple(
                        (
                            path_rows[index],
                            path_rows[index + 1],
                            path_columns[index],
                        )
                        for index in range(edge_count)
                    )
                    if any(
                        table[path_donor][column] == 0
                        for path_donor, _, column in moves
                    ):
                        continue
                    self.path_candidates_checked += 1
                    ratio = _move_ratio(table, moves)
                    if best_ratio is None or ratio < best_ratio:
                        best_ratio = ratio
                        best_moves = moves
        if best_ratio is None:
            raise AssertionError("No feasible neighboring-margin augmentation exists.")

        updated = _apply_moves(table, best_moves)
        if _row_sums(updated) != target_rows:
            raise AssertionError("Neighbor augmentation has the wrong row margins.")
        if _column_sums(updated) != _column_sums(table):
            raise AssertionError("Neighbor augmentation changed column margins.")
        return updated


def _validate_margins(
    rows: tuple[int, ...],
    columns: tuple[int, ...],
) -> None:
    if len(rows) < 2 or len(columns) < 1:
        raise ValueError("A flow state needs at least two rows and one column.")
    if any(value < 0 for value in (*rows, *columns)):
        raise ValueError("Margins must be nonnegative.")
    if sum(rows) != sum(columns):
        raise ValueError("Row and column totals differ.")


def _northwest_corner(
    rows: tuple[int, ...],
    columns: tuple[int, ...],
) -> Table:
    row_remaining = list(rows)
    column_remaining = list(columns)
    table = [[0 for _ in columns] for _ in rows]
    row = 0
    column = 0
    while row < len(rows) and column < len(columns):
        value = min(row_remaining[row], column_remaining[column])
        table[row][column] = value
        row_remaining[row] -= value
        column_remaining[column] -= value
        if row_remaining[row] == 0:
            row += 1
        if column_remaining[column] == 0:
            column += 1
    if any(row_remaining) or any(column_remaining):
        raise AssertionError("Northwest-corner construction failed.")
    return tuple(tuple(table_row) for table_row in table)


def _move_ratio(table: Table, moves: tuple[Move, ...]) -> Fraction:
    ratio = Fraction(1)
    for donor, recipient, column in moves:
        ratio *= Fraction(
            table[recipient][column] + 1,
            table[donor][column],
        )
    return ratio


def _apply_moves(table: Table, moves: tuple[Move, ...]) -> Table:
    updated = [list(row) for row in table]
    for donor, recipient, column in moves:
        if updated[donor][column] <= 0:
            raise AssertionError("A residual move uses a missing donor unit.")
        updated[donor][column] -= 1
        updated[recipient][column] += 1
    return tuple(tuple(row) for row in updated)


def _row_sums(table: Table) -> tuple[int, ...]:
    return tuple(sum(row) for row in table)


def _column_sums(table: Table) -> tuple[int, ...]:
    return tuple(
        sum(table[row][column] for row in range(len(table)))
        for column in range(len(table[0]))
    )


def _factorial_product(table: Table) -> int:
    result = 1
    for row in table:
        for cell in row:
            result *= factorial(cell)
    return result


def _make_solution(
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    table: Table,
) -> FlowSolution:
    factorial_product = _factorial_product(table)
    weight = 1
    for column in range(len(columns)):
        weight *= multinomial(
            tuple(table[row][column] for row in range(len(rows)))
        )
    return FlowSolution(
        row_margins=rows,
        column_margins=columns,
        table=table,
        factorial_product=factorial_product,
        weight=weight,
    )


def exact_flow_potential(table: Table) -> FlowPotential:
    """Construct exact dual scalars by multiplicative Bellman--Ford."""

    row_count = len(table)
    column_count = len(table[0])
    node_count = row_count + column_count
    edges: list[tuple[int, int, Fraction]] = []
    for row in range(row_count):
        for column in range(column_count):
            column_node = row_count + column
            cell = table[row][column]
            edges.append(
                (
                    column_node,
                    row,
                    Fraction(cell + 1),
                )
            )
            if cell > 0:
                edges.append(
                    (
                        row,
                        column_node,
                        Fraction(1, cell),
                    )
                )

    distances = [Fraction(1) for _ in range(node_count)]
    for _ in range(node_count - 1):
        changed = False
        for source, target, ratio in edges:
            candidate = distances[source] * ratio
            if candidate < distances[target]:
                distances[target] = candidate
                changed = True
        if not changed:
            break
    for source, target, ratio in edges:
        if distances[source] * ratio < distances[target]:
            raise ValueError("The table has an improving residual cycle.")

    potential = FlowPotential(
        row_scalars=tuple(distances[:row_count]),
        column_scalars=tuple(
            distances[row_count + column]
            for column in range(column_count)
        ),
    )
    if not verify_flow_potential(table, potential):
        raise AssertionError("Constructed flow potential failed exact verification.")
    return potential


def verify_flow_potential(
    table: Table,
    potential: FlowPotential,
) -> bool:
    """Verify ``x_ij <= alpha_i / beta_j <= x_ij + 1`` exactly."""

    if len(potential.row_scalars) != len(table):
        return False
    if len(potential.column_scalars) != len(table[0]):
        return False
    if any(value <= 0 for value in potential.row_scalars):
        return False
    if any(value <= 0 for value in potential.column_scalars):
        return False
    for row in range(len(table)):
        for column in range(len(table[0])):
            fitted = (
                potential.row_scalars[row]
                / potential.column_scalars[column]
            )
            cell = table[row][column]
            if fitted < cell or fitted > cell + 1:
                return False
    return True
