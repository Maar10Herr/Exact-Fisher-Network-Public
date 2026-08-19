from __future__ import annotations

import unittest

from exact_fisher_network.network import exact_network
from exact_fisher_network.optimized import exact_state_network
from exact_fisher_network.oracle import enumerate_tables
from exact_fisher_network.production import exact_production_network


class OptimizedNetworkTests(unittest.TestCase):
    def test_batched_and_unbatched_agree_with_full_network(self) -> None:
        examples = (
            ((2, 0, 1), (0, 2, 1), (1, 1, 1)),
            ((3, 0, 0, 1), (0, 2, 1, 0), (1, 0, 2, 1)),
            ((0, 1, 3), (2, 1, 0), (1, 2, 0)),
        )
        for observed in examples:
            expected = exact_network(observed).p_value
            plain = exact_state_network(observed, batching=False)
            batched = exact_state_network(observed, batching=True)
            self.assertEqual(plain.p_value, expected)
            self.assertEqual(batched.p_value, expected)

    def test_every_small_observation_agrees(self) -> None:
        for total in range(3, 8):
            for rows in _positive_compositions(total, 3):
                for columns in _positive_compositions(total, min(4, total)):
                    for observed in enumerate_tables(rows, columns):
                        plain = exact_state_network(observed, batching=False)
                        batched = exact_state_network(observed, batching=True)
                        self.assertEqual(batched.p_value, plain.p_value)

    def test_batching_has_a_nonzero_network_effect(self) -> None:
        observed = (
            (7, 0, 0, 1),
            (0, 5, 1, 0),
            (0, 0, 4, 2),
        )
        plain = exact_state_network(observed, batching=False)
        batched = exact_state_network(observed, batching=True)
        self.assertEqual(batched.p_value, plain.p_value)
        self.assertGreater(batched.metrics.batched_arcs_accepted, 0)
        self.assertLess(batched.metrics.arcs_forwarded, plain.metrics.arcs_forwarded)

    def test_global_shared_dp_matches_isolated_recurrence(self) -> None:
        examples = (
            ((2, 0, 1), (0, 2, 1), (1, 1, 1)),
            ((7, 0, 0, 1), (0, 5, 1, 0), (0, 0, 4, 2)),
            ((15, 2, 1), (5, 8, 2), (4, 2, 6)),
        )
        for observed in examples:
            for batching in (False, True):
                isolated = exact_state_network(
                    observed,
                    batching=batching,
                    extrema_cache="isolated",
                )
                global_dp = exact_state_network(
                    observed,
                    batching=batching,
                    extrema_cache="global",
                )
                self.assertEqual(global_dp.p_value, isolated.p_value)
                self.assertEqual(
                    global_dp.accepted_weight,
                    isolated.accepted_weight,
                )

    def test_global_shared_dp_avoids_recursive_recomputation(self) -> None:
        observed = (
            (15, 2, 1),
            (5, 8, 2),
            (4, 2, 6),
        )
        isolated = exact_state_network(
            observed,
            batching=True,
            extrema_cache="isolated",
            collect_certificate=False,
        )
        global_dp = exact_state_network(
            observed,
            batching=True,
            extrema_cache="global",
            collect_certificate=False,
        )
        self.assertLess(
            global_dp.metrics.longest_internal_subproblems,
            isolated.metrics.longest_internal_subproblems,
        )
        self.assertLess(
            global_dp.metrics.shortest_internal_subproblems,
            isolated.metrics.shortest_internal_subproblems,
        )

    def test_flow_longest_baseline_matches_dynamic_programming(self) -> None:
        examples = (
            ((2, 0, 1), (0, 2, 1), (1, 1, 1)),
            ((7, 0, 0, 1), (0, 5, 1, 0), (0, 0, 4, 2)),
            ((3, 1, 0, 2), (1, 3, 2, 0), (2, 0, 3, 1)),
        )
        for observed in examples:
            global_dp = exact_state_network(
                observed,
                batching=False,
                extrema_cache="global",
            )
            flow = exact_state_network(
                observed,
                batching=False,
                extrema_cache="flow",
            )
            self.assertEqual(flow.p_value, global_dp.p_value)
            self.assertEqual(flow.accepted_weight, global_dp.accepted_weight)
            self.assertGreater(flow.metrics.flow_fresh_solves, 0)

    def test_fixed_row_generalization_is_exact_for_four_and_five_rows(
        self,
    ) -> None:
        examples = (
            (
                (2, 0, 1),
                (0, 2, 1),
                (1, 1, 0),
                (0, 1, 2),
            ),
            (
                (2, 0, 1),
                (0, 2, 1),
                (1, 1, 0),
                (0, 1, 2),
                (1, 0, 1),
            ),
        )
        for observed in examples:
            plain = exact_state_network(
                observed,
                batching=False,
                extrema_cache="global",
            )
            batched = exact_state_network(
                observed,
                batching=True,
                extrema_cache="global",
            )
            self.assertEqual(batched.p_value, plain.p_value)
            self.assertEqual(batched.accepted_weight, plain.accepted_weight)

    def test_accept_only_mode_avoids_shortest_solver_and_remains_exact(
        self,
    ) -> None:
        observed = (
            (3, 4, 0, 3, 3),
            (1, 3, 1, 4, 2),
            (0, 1, 4, 4, 2),
        )
        standard = exact_state_network(
            observed,
            batching=True,
            extrema_cache="flow",
            reject_pruning=True,
            collect_certificate=False,
        )
        accept_only = exact_state_network(
            observed,
            batching=True,
            extrema_cache="flow",
            reject_pruning=False,
            collect_certificate=False,
        )
        self.assertEqual(accept_only.p_value, standard.p_value)
        self.assertEqual(accept_only.accepted_weight, standard.accepted_weight)
        self.assertEqual(
            accept_only.metrics.shortest_state_problems_solved,
            0,
        )

    def test_unknown_extrema_cache_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            exact_state_network(
                ((1, 0), (0, 1)),
                batching=False,
                extrema_cache="unknown",  # type: ignore[arg-type]
            )

    def test_production_entry_point_uses_exact_measured_mode(self) -> None:
        observed = (
            (7, 0, 0, 1),
            (0, 5, 1, 0),
            (0, 0, 4, 2),
        )
        result = exact_production_network(observed)
        expected = exact_network(observed)
        self.assertEqual(result.p_value, expected.p_value)
        self.assertEqual(result.extrema_cache, "global")
        self.assertFalse(result.reject_pruning)
        self.assertFalse(result.batching)


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
