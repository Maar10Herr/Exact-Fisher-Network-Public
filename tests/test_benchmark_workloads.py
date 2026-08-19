from __future__ import annotations

import unittest

from benchmarks.workloads import WORKLOADS
from exact_fisher_network.optimized import exact_state_network


class BenchmarkWorkloadTests(unittest.TestCase):
    def test_every_frozen_workload_is_exactly_equal(self) -> None:
        for workload in WORKLOADS.values():
            baseline = exact_state_network(workload["table"], batching=False)
            theorem = exact_state_network(workload["table"], batching=True)
            self.assertEqual(theorem.p_value, baseline.p_value)

    def test_frozen_suite_contains_effect_and_no_effect(self) -> None:
        effects = []
        for workload in WORKLOADS.values():
            baseline = exact_state_network(workload["table"], batching=False)
            theorem = exact_state_network(workload["table"], batching=True)
            effects.append(
                theorem.metrics.arcs_forwarded < baseline.metrics.arcs_forwarded
            )
        self.assertIn(True, effects)
        self.assertIn(False, effects)


if __name__ == "__main__":
    unittest.main()
