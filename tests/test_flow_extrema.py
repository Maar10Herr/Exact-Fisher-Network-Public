from __future__ import annotations

import unittest
from fractions import Fraction

from exact_fisher_network.extrema import ExactExtrema
from exact_fisher_network.flow_extrema import (
    ExactFlowExtrema,
    FlowPotential,
    exact_flow_potential,
    verify_flow_potential,
)


class ExactFlowExtremaTests(unittest.TestCase):
    def test_matches_dynamic_programming_on_small_positive_margins(self) -> None:
        for total in range(2, 9):
            for row_count in range(2, min(4, total) + 1):
                for column_count in range(2, min(4, total) + 1):
                    for rows in _positive_compositions(total, row_count):
                        for columns in _positive_compositions(
                            total,
                            column_count,
                        ):
                            flow = ExactFlowExtrema()
                            solution = flow.solution(rows, columns)
                            dynamic = ExactExtrema().longest(rows, columns)
                            self.assertEqual(solution.weight, dynamic.weight)
                            self.assertTrue(flow.verify(solution))

    def test_neighbor_augmentation_matches_dynamic_programming(self) -> None:
        columns = (4, 3, 2, 2)
        row_sequence = (
            (6, 3, 2),
            (5, 4, 2),
            (4, 5, 2),
            (4, 4, 3),
            (3, 5, 3),
        )
        flow = ExactFlowExtrema()
        for rows in row_sequence:
            solution = flow.solution(rows, columns)
            expected = ExactExtrema().longest(rows, columns)
            self.assertEqual(solution.weight, expected.weight)
            self.assertTrue(flow.verify(solution))
        self.assertEqual(flow.fresh_solves, 1)
        self.assertEqual(flow.neighbor_augments, len(row_sequence) - 1)
        self.assertEqual(flow.neighbor_fallback_cycles, 0)

    def test_invalid_margins_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExactFlowExtrema().solution((2, 1), (2, 2))

    def test_exact_dual_potential_detects_corruption(self) -> None:
        solution = ExactFlowExtrema().solution((6, 3, 2), (4, 3, 2, 2))
        potential = exact_flow_potential(solution.table)
        self.assertTrue(verify_flow_potential(solution.table, potential))
        corrupted = FlowPotential(
            row_scalars=(
                potential.row_scalars[0] * Fraction(1000),
                *potential.row_scalars[1:],
            ),
            column_scalars=potential.column_scalars,
        )
        self.assertFalse(verify_flow_potential(solution.table, corrupted))


def _positive_compositions(total: int, parts: int):
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - parts + 2):
        for tail in _positive_compositions(total - first, parts - 1):
            yield (first, *tail)


if __name__ == "__main__":
    unittest.main()
