"""Counterexample mining for the cross-slice strict Monge candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

from .extrema import ExactExtrema, branch_longest_weights
from .mining import (
    permutation_count,
    positive_compositions,
    positive_partitions,
)


def _first_column_tail_partitions(
    total: int,
    column_count: int,
):
    for first in range(1, total - column_count + 2):
        for tail in positive_partitions(total - first, column_count - 1):
            yield (first, *tail)


def _line_data(
    weights: dict[tuple[int, ...], int],
    first_axis: int,
    slice_axis: int,
) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    grouped: dict[int, list[tuple[int, int]]] = {}
    for daughter, value in weights.items():
        grouped.setdefault(daughter[slice_axis], []).append(
            (daughter[first_axis], value)
        )
    return tuple(
        (
            slice_value,
            tuple(sorted(grouped[slice_value])),
        )
        for slice_value in sorted(grouped)
    )


def mine_stronger_relations(
    *,
    max_total: int,
    max_columns: int,
    symmetry_reduced: bool,
) -> dict[str, object]:
    started_at = datetime.now(timezone.utc)
    started_ns = time.perf_counter_ns()
    margin_problems = 0
    ordered_margin_problems_represented = 0
    rhombi = 0
    adjacent_peak_pairs = 0
    threshold_regions = 0
    strict_rhombus_counterexample = None
    peak_counterexample = None
    monotone_boundary_counterexample = None
    unit_boundary_counterexample = None

    for total in range(3, max_total + 1):
        for column_count in range(2, min(max_columns, total) + 1):
            column_iterable = (
                _first_column_tail_partitions(total, column_count)
                if symmetry_reduced
                else positive_compositions(total, column_count)
            )
            for columns in column_iterable:
                extrema = ExactExtrema()
                row_iterable = (
                    positive_partitions(total, 3)
                    if symmetry_reduced
                    else positive_compositions(total, 3)
                )
                for rows in row_iterable:
                    margin_problems += 1
                    ordered_margin_problems_represented += (
                        permutation_count(rows)
                        * permutation_count(columns[1:])
                        if symmetry_reduced
                        else 1
                    )
                    weights = branch_longest_weights(
                        rows,
                        columns,
                        extrema,
                    )
                    daughters = set(weights)

                    for recipient_one, donor, recipient_two in permutations(
                        range(3),
                        3,
                    ):
                        for daughter in daughters:
                            if daughter[donor] < 2:
                                continue
                            opposite = list(daughter)
                            opposite[recipient_one] += 1
                            opposite[donor] -= 2
                            opposite[recipient_two] += 1
                            edge_one = list(daughter)
                            edge_one[recipient_one] += 1
                            edge_one[donor] -= 1
                            edge_two = list(daughter)
                            edge_two[recipient_two] += 1
                            edge_two[donor] -= 1
                            opposite_key = tuple(opposite)
                            edge_one_key = tuple(edge_one)
                            edge_two_key = tuple(edge_two)
                            if not {
                                opposite_key,
                                edge_one_key,
                                edge_two_key,
                            }.issubset(daughters):
                                continue
                            rhombi += 1
                            lhs = (
                                weights[edge_one_key]
                                * weights[edge_two_key]
                            )
                            rhs = (
                                weights[daughter]
                                * weights[opposite_key]
                            )
                            if (
                                strict_rhombus_counterexample is None
                                and lhs <= rhs
                            ):
                                strict_rhombus_counterexample = {
                                    "total": total,
                                    "row_margins": rows,
                                    "column_margins": columns,
                                    "axes": {
                                        "recipient_one": recipient_one,
                                        "donor": donor,
                                        "recipient_two": recipient_two,
                                    },
                                    "base": daughter,
                                    "opposite": opposite_key,
                                    "edge_one": edge_one_key,
                                    "edge_two": edge_two_key,
                                    "weights": {
                                        "base": weights[daughter],
                                        "opposite": weights[opposite_key],
                                        "edge_one": weights[edge_one_key],
                                        "edge_two": weights[edge_two_key],
                                    },
                                    "strict_side": lhs,
                                    "other_side": rhs,
                                }

                    for first_axis, donor, slice_axis in permutations(
                        range(3),
                        3,
                    ):
                        lines = _line_data(weights, first_axis, slice_axis)
                        modes = tuple(
                            (
                                slice_value,
                                max(
                                    line,
                                    key=lambda item: (item[1], -item[0]),
                                )[0],
                            )
                            for slice_value, line in lines
                        )
                        for left, right in zip(modes, modes[1:]):
                            if right[0] != left[0] + 1:
                                continue
                            adjacent_peak_pairs += 1
                            if peak_counterexample is None and right[1] > left[1]:
                                peak_counterexample = {
                                    "total": total,
                                    "row_margins": rows,
                                    "column_margins": columns,
                                    "axes": {
                                        "first": first_axis,
                                        "donor": donor,
                                        "slice": slice_axis,
                                    },
                                    "left": left,
                                    "right": right,
                                    "lines": lines,
                                }

                        if (
                            monotone_boundary_counterexample is not None
                            and unit_boundary_counterexample is not None
                        ):
                            continue
                        for threshold in sorted(set(weights.values())):
                            threshold_regions += 1
                            intervals = []
                            for slice_value, line in lines:
                                accepted = tuple(
                                    coordinate
                                    for coordinate, value in line
                                    if value > threshold
                                )
                                if accepted:
                                    intervals.append(
                                        (
                                            slice_value,
                                            min(accepted),
                                            max(accepted),
                                        )
                                    )
                            for left, right in zip(intervals, intervals[1:]):
                                if right[0] != left[0] + 1:
                                    continue
                                if (
                                    monotone_boundary_counterexample is None
                                    and (
                                        right[1] > left[1]
                                        or right[2] > left[2]
                                    )
                                ):
                                    monotone_boundary_counterexample = {
                                        "total": total,
                                        "row_margins": rows,
                                        "column_margins": columns,
                                        "threshold": threshold,
                                        "axes": {
                                            "first": first_axis,
                                            "donor": donor,
                                            "slice": slice_axis,
                                        },
                                        "left": left,
                                        "right": right,
                                        "lines": lines,
                                    }
                                if (
                                    unit_boundary_counterexample is None
                                    and (
                                        abs(right[1] - left[1]) > 1
                                        or abs(right[2] - left[2]) > 1
                                    )
                                ):
                                    unit_boundary_counterexample = {
                                        "total": total,
                                        "row_margins": rows,
                                        "column_margins": columns,
                                        "threshold": threshold,
                                        "axes": {
                                            "first": first_axis,
                                            "donor": donor,
                                            "slice": slice_axis,
                                        },
                                        "left": left,
                                        "right": right,
                                        "lines": lines,
                                    }

    source_path = Path(__file__)
    ended_at = datetime.now(timezone.utc)
    return {
        "schema": "exact-fisher-stronger-mining-v1",
        "domain": {
            "rows": 3,
            "minimum_columns": 2,
            "maximum_columns": max_columns,
            "maximum_total": max_total,
            "positive_margins": True,
            "symmetry_reduced": symmetry_reduced,
        },
        "counts": {
            "margin_problems": margin_problems,
            "ordered_margin_problems_represented": (
                ordered_margin_problems_represented
            ),
            "strict_rhombi": rhombi,
            "adjacent_peak_pairs": adjacent_peak_pairs,
            "threshold_regions_until_false_candidates_closed": (
                threshold_regions
            ),
        },
        "strict_multiplicative_rhombus": {
            "counterexample": strict_rhombus_counterexample,
            "classification": (
                "counterexample"
                if strict_rhombus_counterexample is not None
                else "survived"
            ),
        },
        "cross_slice_leftmost_peak_nonincrease": {
            "counterexample": peak_counterexample,
            "classification": (
                "counterexample"
                if peak_counterexample is not None
                else "survived"
            ),
        },
        "fixed_threshold_boundaries_nonincreasing": {
            "counterexample": monotone_boundary_counterexample,
            "classification": "counterexample",
        },
        "fixed_threshold_boundaries_move_at_most_one": {
            "counterexample": unit_boundary_counterexample,
            "classification": "counterexample",
        },
        "provenance": {
            "started_at_utc": started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "elapsed_ns": time.perf_counter_ns() - started_ns,
            "python": sys.version,
            "platform": platform.platform(),
            "source_file": str(source_path),
            "source_sha256": hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest(),
            "argv": sys.argv,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-total", type=int, default=14)
    parser.add_argument("--max-columns", type=int, default=6)
    parser.add_argument("--symmetry-reduced", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = mine_stronger_relations(
        max_total=arguments.max_total,
        max_columns=arguments.max_columns,
        symmetry_reduced=arguments.symmetry_reduced,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
        print(arguments.output)


if __name__ == "__main__":
    main()
