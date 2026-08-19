from __future__ import annotations

import copy
import unittest

from exact_fisher_network.compact_certificate import compact_certificate_data
from exact_fisher_network.optimized import exact_state_network
from exact_fisher_network.verify_compact_certificate import (
    verify_compact_certificate_data,
)


OBSERVED = (
    (7, 0, 0, 1),
    (0, 5, 1, 0),
    (0, 0, 4, 2),
)


class CompactCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = exact_state_network(
            OBSERVED,
            batching=True,
            extrema_cache="global",
            reject_pruning=False,
            collect_certificate=True,
        )
        cls.data = compact_certificate_data(cls.result)
        if not cls.data["batches"]:
            raise AssertionError("Compact fixture did not batch a line.")

    def test_valid_compact_certificate_replays(self) -> None:
        self.assertEqual(
            verify_compact_certificate_data(copy.deepcopy(self.data)),
            self.result.p_value,
        )

    def test_omitted_daughters_are_not_listed(self) -> None:
        for batch in self.data["batches"]:
            self.assertNotIn("line_columns", batch)
            self.assertNotIn("accepted_columns", batch)

    def test_corrupted_state_dual_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        state = next(
            item for item in data["states"] if item["remaining_columns"]
        )
        extremum = data["extrema"][state["longest_extremum_id"]]
        first = extremum["row_scalars"][0]
        extremum["row_scalars"] = (
            (first[0] * 1000, first[1]),
            *extremum["row_scalars"][1:],
        )
        with self.assertRaises(AssertionError):
            verify_compact_certificate_data(data)

    def test_corrupted_boundary_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        batch = data["batches"][0]
        batch["accepted_left_end"] = None
        batch["accepted_right_start"] = None
        with self.assertRaises(AssertionError):
            verify_compact_certificate_data(data)

    def test_corrupted_query_dual_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        query = data["batches"][0]["queries"][0]
        extremum = data["extrema"][query["suffix_extremum_id"]]
        first = extremum["row_scalars"][0]
        extremum["row_scalars"] = (
            (first[0] * 1000, first[1]),
            *extremum["row_scalars"][1:],
        )
        with self.assertRaises(AssertionError):
            verify_compact_certificate_data(data)

    def test_corrupted_interval_mass_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        data["batches"][0]["accepted_weight"] += 1
        with self.assertRaises(AssertionError):
            verify_compact_certificate_data(data)

    def test_corrupted_fixed_tail_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        tail = data["batches"][0]["fixed_tail"]
        data["batches"][0]["fixed_tail"] = (tail[0] + 1, *tail[1:])
        with self.assertRaises(AssertionError):
            verify_compact_certificate_data(data)

    def test_corrupted_line_length_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        data["batches"][0]["line_length"] += 1
        with self.assertRaises(AssertionError):
            verify_compact_certificate_data(data)

    def test_corrupted_query_branch_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        data["batches"][0]["queries"][0]["branch_longest"] += 1
        with self.assertRaises(AssertionError):
            verify_compact_certificate_data(data)

    def test_corrupted_state_mass_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        data["states"][0]["accepted_weight"] += 1
        with self.assertRaises(AssertionError):
            verify_compact_certificate_data(data)

    def test_corrupted_peak_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        batch = data["batches"][0]
        batch["peak_index"] = (batch["peak_index"] + 1) % batch["line_length"]
        with self.assertRaises(AssertionError):
            verify_compact_certificate_data(data)

    def test_missing_batch_changes_replay_and_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        del data["batches"][0]
        with self.assertRaises(AssertionError):
            verify_compact_certificate_data(data)

    def test_corrupted_final_rational_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        data["p_value"]["numerator"] += 1
        with self.assertRaises(AssertionError):
            verify_compact_certificate_data(data)

    def test_unreferenced_extremum_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        data["extrema"].append(copy.deepcopy(data["extrema"][0]))
        with self.assertRaises(AssertionError):
            verify_compact_certificate_data(data)

    def test_compact_builder_rejects_shortest_pruning_mode(self) -> None:
        standard = exact_state_network(
            OBSERVED,
            batching=True,
            extrema_cache="global",
            reject_pruning=True,
        )
        with self.assertRaises(ValueError):
            compact_certificate_data(standard)


if __name__ == "__main__":
    unittest.main()
