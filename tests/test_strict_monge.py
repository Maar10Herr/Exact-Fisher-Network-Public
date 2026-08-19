from __future__ import annotations

import unittest
from itertools import permutations

from benchmarks.workloads import WORKLOADS
from exact_fisher_network.extrema import (
    ExactExtrema,
    branch_longest_weights,
)
from exact_fisher_network.mining import positive_compositions
from exact_fisher_network.optimized import exact_state_network
from exact_fisher_network.stronger_mining import mine_stronger_relations


class StrictMongeTests(unittest.TestCase):
    def test_every_small_feasible_rhombus_is_strict(self) -> None:
        checked = 0
        for total in range(3, 9):
            for column_count in range(2, min(4, total) + 1):
                for columns in positive_compositions(total, column_count):
                    for rows in positive_compositions(total, 3):
                        weights = branch_longest_weights(
                            rows,
                            columns,
                            ExactExtrema(),
                        )
                        daughters = set(weights)
                        for first, donor, second in permutations(range(3), 3):
                            for daughter in daughters:
                                if daughter[donor] < 2:
                                    continue
                                opposite = list(daughter)
                                opposite[first] += 1
                                opposite[donor] -= 2
                                opposite[second] += 1
                                edge_one = list(daughter)
                                edge_one[first] += 1
                                edge_one[donor] -= 1
                                edge_two = list(daughter)
                                edge_two[second] += 1
                                edge_two[donor] -= 1
                                keys = (
                                    tuple(opposite),
                                    tuple(edge_one),
                                    tuple(edge_two),
                                )
                                if not set(keys).issubset(daughters):
                                    continue
                                checked += 1
                                self.assertGreater(
                                    weights[keys[1]] * weights[keys[2]],
                                    weights[daughter] * weights[keys[0]],
                                )
        self.assertGreater(checked, 0)

    def test_miner_retains_boundary_counterexamples(self) -> None:
        result = mine_stronger_relations(
            max_total=8,
            max_columns=4,
            symmetry_reduced=True,
        )
        self.assertIsNone(
            result["strict_multiplicative_rhombus"]["counterexample"]
        )
        self.assertIsNone(
            result["cross_slice_leftmost_peak_nonincrease"]["counterexample"]
        )
        self.assertIsNotNone(
            result["fixed_threshold_boundaries_nonincreasing"][
                "counterexample"
            ]
        )
        self.assertIsNotNone(
            result["fixed_threshold_boundaries_move_at_most_one"][
                "counterexample"
            ]
        )

    def test_monotone_peak_walk_saves_exact_bound_queries(self) -> None:
        table = WORKLOADS["wide_first_column"]["table"]
        independent = exact_state_network(
            table,
            batching=True,
            extrema_cache="flow",
            line_search="sparse",
            monotone_peaks=False,
            collect_certificate=False,
        )
        monotone = exact_state_network(
            table,
            batching=True,
            extrema_cache="flow",
            line_search="sparse",
            monotone_peaks=True,
            collect_certificate=False,
        )
        self.assertEqual(monotone.p_value, independent.p_value)
        self.assertGreater(monotone.metrics.monotone_peak_reuses, 0)
        self.assertLess(
            monotone.metrics.batch_longest_lookups,
            independent.metrics.batch_longest_lookups,
        )


if __name__ == "__main__":
    unittest.main()
