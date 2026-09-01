from __future__ import annotations

from fractions import Fraction
from itertools import combinations
import unittest

from exact_fisher_network.network import exact_network
from exact_fisher_network.optimized import exact_state_network
from exact_fisher_network.oracle import (
    conditional_denominator,
    enumerate_tables,
    exact_enumeration,
    path_weight,
)


def _positive_compositions(total: int, length: int):
    for ordered in combinations(range(1, total), length - 1):
        values = (
            ordered[0],
            *(ordered[index + 1] - ordered[index] for index in range(length - 2)),
            total - ordered[-1],
        )
        yield values


class ExactOracleTests(unittest.TestCase):
    def test_direct_and_network_agree_on_named_examples(self) -> None:
        examples = [
            (((1, 9), (11, 3)), Fraction(41, 14858)),
            (((2, 1, 0), (0, 1, 2)), Fraction(3, 5)),
            (((1, 0), (0, 1), (1, 1)), Fraction(1, 1)),
            (((2, 0, 1), (0, 2, 1), (1, 1, 1)), Fraction(61, 70)),
        ]
        for table, expected in examples:
            with self.subTest(table=table):
                direct = exact_enumeration(table)
                network = exact_network(table)
                self.assertEqual(direct.p_value, expected)
                self.assertEqual(network.p_value, expected)
                self.assertEqual(direct.p_value, network.p_value)
                self.assertEqual(direct.observed_weight, network.observed_weight)
                self.assertEqual(direct.denominator, network.denominator)

    def test_conditional_mass_identity(self) -> None:
        rows = (2, 3, 2)
        columns = (1, 2, 4)
        weights = [path_weight(table) for table in enumerate_tables(rows, columns)]
        self.assertEqual(sum(weights), conditional_denominator(rows))

    def test_all_small_positive_margin_problems(self) -> None:
        """All 2×c and 3×c problems for N≤7 and 2≤c≤4."""

        for row_count in (2, 3):
            for total in range(max(row_count, 2), 8):
                for column_count in range(2, min(4, total) + 1):
                    if total < max(row_count, column_count):
                        continue
                    for rows in _positive_compositions(total, row_count):
                        for columns in _positive_compositions(total, column_count):
                            tables = tuple(enumerate_tables(rows, columns))
                            weights = {
                                table: path_weight(table)
                                for table in tables
                            }
                            denominator = conditional_denominator(rows)
                            representatives = {}
                            for table, weight in weights.items():
                                representatives.setdefault(weight, table)
                            for threshold, observed in representatives.items():
                                direct_p = Fraction(
                                    sum(
                                        weight
                                        for weight in weights.values()
                                        if weight <= threshold
                                    ),
                                    denominator,
                                )
                                network = exact_network(observed)
                                self.assertEqual(
                                    direct_p,
                                    network.p_value,
                                    (rows, columns, observed),
                                )

    def test_instrumentation_records_every_checked_pair(self) -> None:
        result = exact_network(((2, 0, 1), (0, 2, 1), (1, 1, 1)))
        self.assertEqual(
            len(result.pruning_records),
            result.metrics.prefix_state_pairs_checked,
        )
        self.assertGreaterEqual(
            result.metrics.pruning_comparisons,
            len(result.pruning_records),
        )
        self.assertEqual(
            {record.decision for record in result.pruning_records},
            {"accept_all", "reject_all", "expand"},
        )
        for state in result.state_records:
            self.assertEqual(
                state.shortest_completion,
                state.continuation_multiset[0][0],
            )
            self.assertEqual(
                state.longest_completion,
                state.continuation_multiset[-1][0],
            )
            self.assertEqual(
                state.total_completion_weight,
                sum(
                    weight * count
                    for weight, count in state.continuation_multiset
                ),
            )

    def test_nonintegral_and_boolean_cells_are_rejected(self) -> None:
        for invalid in (1.5, True, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    exact_network(((invalid, 1), (1, 1)))
                with self.assertRaises(ValueError):
                    exact_enumeration(((invalid, 1), (1, 1)))
                with self.assertRaises(ValueError):
                    exact_state_network(
                        ((invalid, 1), (1, 1)),
                        batching=False,
                    )

    def test_nonintegral_and_boolean_margins_are_rejected(self) -> None:
        for invalid in (1.5, True, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    tuple(enumerate_tables((invalid, 1), (1, 1)))
                with self.assertRaises(ValueError):
                    tuple(enumerate_tables((1, 1), (invalid, 1)))
                with self.assertRaises(ValueError):
                    conditional_denominator((invalid, 1))


if __name__ == "__main__":
    unittest.main()
