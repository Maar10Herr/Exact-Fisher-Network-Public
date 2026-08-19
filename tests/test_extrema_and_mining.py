from __future__ import annotations

import unittest

from exact_fisher_network.extrema import (
    ExactExtrema,
    branch_longest_weights,
    three_row_slices,
)
from exact_fisher_network.mining import mine


class ExtremaTests(unittest.TestCase):
    def test_longest_and_shortest_have_exact_witnesses(self) -> None:
        solver = ExactExtrema()
        rows = (3, 3, 3)
        columns = (3, 3, 3)
        longest = solver.longest(rows, columns)
        shortest = solver.shortest(rows, columns)
        self.assertGreaterEqual(longest.weight, shortest.weight)
        self.assertEqual(tuple(map(sum, zip(*longest.columns))), rows)
        self.assertEqual(tuple(map(sum, zip(*shortest.columns))), rows)
        self.assertEqual(tuple(map(sum, longest.columns)), columns)
        self.assertEqual(tuple(map(sum, shortest.columns)), columns)

    def test_branch_slices_are_log_concave(self) -> None:
        weights = branch_longest_weights((4, 3, 2), (3, 2, 4))
        for line in three_row_slices(weights):
            values = [weight for _, weight in line]
            for index in range(1, len(values) - 1):
                self.assertGreaterEqual(
                    values[index] ** 2,
                    values[index - 1] * values[index + 1],
                )

    def test_small_mining_finds_no_log_concavity_counterexample(self) -> None:
        summary = mine(max_total=7, max_columns=4)
        self.assertNotIn(
            "multiplicative_line_log_concavity",
            summary.counterexamples,
        )
        self.assertGreater(summary.adjacent_triples, 0)

    def test_symmetry_reduction_represents_full_ordered_domain(self) -> None:
        full = mine(max_total=7, max_columns=4)
        reduced = mine(
            max_total=7,
            max_columns=4,
            symmetry_reduced=True,
        )
        self.assertEqual(
            reduced.ordered_margin_problems_represented,
            full.margin_problems,
        )
        self.assertLess(reduced.margin_problems, full.margin_problems)

    def test_false_candidates_retain_counterexamples(self) -> None:
        summary = mine(max_total=8, max_columns=4)
        self.assertIn(
            "line_nondecreasing_in_first_coordinate",
            summary.counterexamples,
        )
        self.assertIn(
            "arc_weight_orders_branch_longest",
            summary.counterexamples,
        )
        self.assertNotIn(
            "strict_multiplicative_line_log_concavity",
            summary.counterexamples,
        )


if __name__ == "__main__":
    unittest.main()
