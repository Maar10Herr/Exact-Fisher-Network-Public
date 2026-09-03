from __future__ import annotations

import copy
import unittest

from exact_fisher_network.optimized import exact_state_network
from exact_fisher_network.optimized_certificate import certificate_data
from exact_fisher_network.verify_optimized_certificate import verify_certificate_data


OBSERVED = (
    (7, 0, 0, 1),
    (0, 5, 1, 0),
    (0, 0, 4, 2),
)


class OptimizedCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = exact_state_network(OBSERVED, batching=True)
        cls.data = certificate_data(cls.result)
        if not cls.data["batch_records"]:
            raise AssertionError("Certificate fixture did not exercise batching.")

    def test_valid_certificate_replays(self) -> None:
        self.assertEqual(
            verify_certificate_data(copy.deepcopy(self.data)),
            self.result.p_value,
        )

    def test_non_integer_scalars_are_rejected(self) -> None:
        paths = (
            ("observed_weight",),
            ("state_records", 0, "accepted_weight"),
            ("batch_records", 0, "peak_index"),
        )
        for invalid in (1.5, "1", True):
            for path in paths:
                data = copy.deepcopy(self.data)
                target = data
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = invalid
                with self.subTest(path=path, invalid=invalid):
                    with self.assertRaises(AssertionError):
                        verify_certificate_data(data)

    def test_corrupted_batch_value_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        data["batch_records"][0]["queries"][0]["branch_longest"] += 1
        with self.assertRaises(AssertionError):
            verify_certificate_data(data)

    def test_corrupted_batch_boundary_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        record = data["batch_records"][0]
        record["accepted_left_end"] = None
        record["accepted_right_start"] = None
        with self.assertRaises(AssertionError):
            verify_certificate_data(data)

    def test_corrupted_batch_contribution_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        data["batch_records"][0]["accepted_weight"] += 1
        with self.assertRaises(AssertionError):
            verify_certificate_data(data)

    def test_corrupted_extremal_witness_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        query = data["batch_records"][0]["queries"][0]
        witness = query["maximizing_columns"]
        first = witness[0]
        query["maximizing_columns"] = (
            (first[0] + 1, *first[1:]),
            *witness[1:],
        )
        with self.assertRaises(AssertionError):
            verify_certificate_data(data)

    def test_corrupted_state_decision_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        data["state_records"][0]["decision"] = "reject_all"
        with self.assertRaises(AssertionError):
            verify_certificate_data(data)

    def test_missing_batch_record_changes_replay_and_is_rejected(self) -> None:
        data = copy.deepcopy(self.data)
        del data["batch_records"][0]
        with self.assertRaises(AssertionError):
            verify_certificate_data(data)


if __name__ == "__main__":
    unittest.main()
