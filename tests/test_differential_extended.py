from __future__ import annotations

import random
import unittest
from itertools import permutations

from benchmarks.workloads import WORKLOADS
from exact_fisher_network.optimized import exact_state_network


class ExtendedDifferentialTests(unittest.TestCase):
    def test_row_and_column_permutations_preserve_exact_rational(self) -> None:
        table = (
            (3, 1, 0, 2),
            (1, 3, 2, 0),
            (2, 0, 3, 1),
        )
        expected = exact_state_network(
            table,
            batching=False,
            extrema_cache="global",
            reject_pruning=False,
            collect_certificate=False,
        ).p_value
        for row_order in permutations(range(3)):
            for column_order in permutations(range(4)):
                permuted = tuple(
                    tuple(table[row][column] for column in column_order)
                    for row in row_order
                )
                actual = exact_state_network(
                    permuted,
                    batching=True,
                    extrema_cache="global",
                    reject_pruning=False,
                    collect_certificate=False,
                ).p_value
                self.assertEqual(actual, expected)

    def test_deterministic_random_positive_margin_tables(self) -> None:
        generator = random.Random(20260724)
        for _ in range(25):
            while True:
                table = tuple(
                    tuple(generator.randrange(5) for _ in range(5))
                    for _ in range(3)
                )
                if all(map(sum, table)) and all(
                    sum(table[row][column] for row in range(3))
                    for column in range(5)
                ):
                    break
            reference = exact_state_network(
                table,
                batching=False,
                extrema_cache="global",
                reject_pruning=True,
                collect_certificate=False,
            )
            for options in (
                {
                    "batching": False,
                    "extrema_cache": "flow",
                    "reject_pruning": True,
                },
                {
                    "batching": False,
                    "extrema_cache": "global",
                    "reject_pruning": False,
                },
                {
                    "batching": True,
                    "extrema_cache": "global",
                    "reject_pruning": False,
                },
            ):
                result = exact_state_network(
                    table,
                    collect_certificate=False,
                    **options,
                )
                self.assertEqual(result.p_value, reference.p_value)
                self.assertEqual(
                    result.accepted_weight,
                    reference.accepted_weight,
                )

    def test_bigint_stress_returns_one_identical_rational(self) -> None:
        table = WORKLOADS["bigint_stress"]["table"]
        results = [
            exact_state_network(
                table,
                collect_certificate=False,
                **options,
            )
            for options in (
                {
                    "batching": False,
                    "extrema_cache": "global",
                    "reject_pruning": True,
                },
                {
                    "batching": False,
                    "extrema_cache": "global",
                    "reject_pruning": False,
                },
                {
                    "batching": True,
                    "extrema_cache": "global",
                    "reject_pruning": False,
                },
            )
        ]
        self.assertTrue(
            all(result.p_value == results[0].p_value for result in results)
        )
        self.assertGreater(results[0].denominator.bit_length(), 64)


if __name__ == "__main__":
    unittest.main()
