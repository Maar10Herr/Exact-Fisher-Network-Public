"""Additional state-dominance and continuation-multiset falsification."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from .extrema import ExactExtrema, multinomial
from .mining import positive_compositions, positive_partitions
from .network import _ContinuationOracle


def _majorizes(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    left_descending = sorted(left, reverse=True)
    right_descending = sorted(right, reverse=True)
    return all(
        sum(left_descending[:index]) >= sum(right_descending[:index])
        for index in range(1, len(left))
    )


def _tail_probability(
    multiset: tuple[tuple[int, int], ...],
    denominator: int,
    threshold: int,
) -> Fraction:
    return sum(
        (
            Fraction(weight * multiplicity, denominator)
            for weight, multiplicity in multiset
            if weight >= threshold
        ),
        Fraction(0, 1),
    )


def mine_state_relations(
    *,
    max_total: int,
    max_columns: int,
) -> dict[str, object]:
    majorization_comparisons = 0
    majorization_counterexample = None
    support_counterexample = None
    stochastic_counterexample = None

    for total in range(3, max_total + 1):
        row_partitions = list(positive_partitions(total, 3))
        for column_count in range(2, min(max_columns, total) + 1):
            for columns in positive_compositions(total, column_count):
                extrema = ExactExtrema()
                longest = {
                    rows: extrema.longest(rows, columns).weight
                    for rows in row_partitions
                }
                for left in row_partitions:
                    for right in row_partitions:
                        if not _majorizes(left, right):
                            continue
                        majorization_comparisons += 1
                        if longest[left] > longest[right]:
                            majorization_counterexample = {
                                "column_margins": columns,
                                "more_imbalanced_rows": left,
                                "more_balanced_rows": right,
                                "imbalanced_longest": longest[left],
                                "balanced_longest": longest[right],
                            }
                            break
                    if majorization_counterexample is not None:
                        break
                if majorization_counterexample is not None:
                    break

                if support_counterexample is not None and stochastic_counterexample is not None:
                    continue
                continuation = _ContinuationOracle()
                for rows in positive_compositions(total, 3):
                    for row_out in range(3):
                        for row_in in range(3):
                            if rows[row_out] < rows[row_in] + 2:
                                continue
                            balanced_list = list(rows)
                            balanced_list[row_out] -= 1
                            balanced_list[row_in] += 1
                            balanced = tuple(balanced_list)
                            original_multiset = continuation.counter(rows, columns)
                            balanced_multiset = continuation.counter(balanced, columns)

                            if support_counterexample is None:
                                original_support = {
                                    weight for weight, _ in original_multiset
                                }
                                balanced_support = {
                                    weight for weight, _ in balanced_multiset
                                }
                                missing = sorted(original_support - balanced_support)
                                if missing:
                                    support_counterexample = {
                                        "column_margins": columns,
                                        "original_rows": rows,
                                        "robin_hood_rows": balanced,
                                        "original_multiset": original_multiset,
                                        "balanced_multiset": balanced_multiset,
                                        "missing_weights": missing,
                                    }

                            if stochastic_counterexample is None:
                                thresholds = sorted(
                                    {
                                        weight for weight, _ in original_multiset
                                    }
                                    | {
                                        weight for weight, _ in balanced_multiset
                                    }
                                )
                                for threshold in thresholds:
                                    original_tail = _tail_probability(
                                        original_multiset,
                                        multinomial(rows),
                                        threshold,
                                    )
                                    balanced_tail = _tail_probability(
                                        balanced_multiset,
                                        multinomial(balanced),
                                        threshold,
                                    )
                                    if balanced_tail < original_tail:
                                        stochastic_counterexample = {
                                            "column_margins": columns,
                                            "original_rows": rows,
                                            "robin_hood_rows": balanced,
                                            "threshold": threshold,
                                            "original_multiset": original_multiset,
                                            "balanced_multiset": balanced_multiset,
                                            "original_upper_tail": {
                                                "numerator": original_tail.numerator,
                                                "denominator": original_tail.denominator,
                                            },
                                            "balanced_upper_tail": {
                                                "numerator": balanced_tail.numerator,
                                                "denominator": balanced_tail.denominator,
                                            },
                                        }
                                        break

    return {
        "schema": "exact-fisher-state-relation-mining-v1",
        "domain": {
            "rows": 3,
            "max_total": max_total,
            "max_columns": max_columns,
            "positive_margins": True,
        },
        "residual_margin_majorization": {
            "comparisons": majorization_comparisons,
            "counterexample": majorization_counterexample,
            "classification": (
                "counterexample"
                if majorization_counterexample is not None
                else "survived_but_prior_art"
            ),
        },
        "continuation_support_inclusion_after_robin_hood_exchange": {
            "counterexample": support_counterexample,
            "classification": "counterexample",
        },
        "normalized_upper_tail_dominance_after_robin_hood_exchange": {
            "counterexample": stochastic_counterexample,
            "classification": "counterexample",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-total", type=int, default=14)
    parser.add_argument("--max-columns", type=int, default=6)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = mine_state_relations(
        max_total=arguments.max_total,
        max_columns=arguments.max_columns,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
