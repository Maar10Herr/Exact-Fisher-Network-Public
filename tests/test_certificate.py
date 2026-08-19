from __future__ import annotations

import copy
import unittest

from exact_fisher_network.certificate import certificate_data
from exact_fisher_network.verify_certificate import validate_certificate_data


class CertificateTests(unittest.TestCase):
    @staticmethod
    def certificate() -> dict[str, object]:
        return certificate_data(((2, 0, 1), (0, 2, 1), (1, 1, 1)))

    def assert_rejected(self, data: dict[str, object]) -> None:
        with self.assertRaises(ValueError):
            validate_certificate_data(data)

    def test_valid_certificate_replays(self) -> None:
        self.assertEqual(
            str(validate_certificate_data(self.certificate())),
            "61/70",
        )

    def test_corrupted_threshold_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["result"]["observed_weight"] += 1
        self.assert_rejected(data)

    def test_corrupted_residual_state_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["states"][0]["residual_rows"][0] += 1
        self.assert_rejected(data)

    def test_corrupted_longest_completion_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["states"][0]["longest_completion"] += 1
        self.assert_rejected(data)

    def test_corrupted_continuation_multiset_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["states"][0]["continuation_multiset"][0][1] += 1
        self.assert_rejected(data)

    def test_corrupted_pruning_decision_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["pruning"][0]["decision"] = "reject_all"
        self.assert_rejected(data)

    def test_corrupted_aggregate_contribution_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        accepting = next(
            record
            for record in data["pruning"]
            if record["decision"] == "accept_all"
        )
        accepting["accepted_weight"] += 1
        self.assert_rejected(data)

    def test_missing_record_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["pruning"].pop()
        self.assert_rejected(data)


if __name__ == "__main__":
    unittest.main()

