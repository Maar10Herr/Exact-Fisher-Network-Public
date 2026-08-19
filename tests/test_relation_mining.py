from __future__ import annotations

import unittest

from exact_fisher_network.relation_mining import mine_state_relations


class RelationMiningTests(unittest.TestCase):
    def test_small_domain_classifies_state_relations(self) -> None:
        result = mine_state_relations(max_total=6, max_columns=4)
        self.assertIsNone(
            result["residual_margin_majorization"]["counterexample"]
        )
        self.assertIsNotNone(
            result[
                "continuation_support_inclusion_after_robin_hood_exchange"
            ]["counterexample"]
        )
        self.assertIsNotNone(
            result[
                "normalized_upper_tail_dominance_after_robin_hood_exchange"
            ]["counterexample"]
        )


if __name__ == "__main__":
    unittest.main()
