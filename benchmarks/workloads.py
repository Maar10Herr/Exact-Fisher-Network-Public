"""Small deterministic workloads used by tests and examples.

These tables are stable implementation fixtures for correctness checks and
reproducible examples.
"""

WORKLOADS = {
    "root_accept_equal": {
        "tags": ["balanced", "root-pruned", "tie"],
        "table": ((2, 1, 1), (1, 2, 1), (1, 1, 2)),
    },
    "exact_boundary_equality": {
        "tags": ["adversarial", "threshold-equality", "boundary"],
        "table": ((4, 0, 0), (2, 1, 1), (0, 2, 2)),
    },
    "unbalanced_small": {
        "tags": ["unbalanced", "small"],
        "table": ((6, 2, 1, 1), (1, 1, 1, 0), (0, 0, 1, 1)),
    },
    "sparse_separated": {
        "tags": ["sparse", "separated", "slower-risk"],
        "table": ((7, 0, 0, 1), (0, 5, 1, 0), (0, 0, 4, 2)),
    },
    "mixed_five_columns": {
        "tags": ["mixed", "five-columns", "many-prefixes"],
        "table": ((3, 1, 0, 2, 1), (1, 3, 2, 0, 1), (2, 0, 3, 1, 1)),
    },
    "wide_first_column": {
        "tags": ["wide-daughter-triangle", "structural-win"],
        "table": ((15, 2, 1), (5, 8, 2), (4, 2, 6)),
    },
    "repeated_prefix_heavy": {
        "tags": ["deterministic-random", "five-columns", "repeated-prefix-heavy"],
        "table": ((3, 4, 0, 3, 3), (1, 3, 1, 4, 2), (0, 1, 4, 4, 2)),
    },
    "four_row_sparse": {
        "tags": ["four-row", "sparse", "fixed-r-generalization"],
        "table": ((2, 0, 1), (0, 2, 1), (1, 1, 0), (0, 1, 2)),
    },
    "five_row_small": {
        "tags": ["five-row", "fixed-r-generalization"],
        "table": ((2, 0, 1), (0, 2, 1), (1, 1, 0), (0, 1, 2), (1, 0, 1)),
    },
    "bigint_stress": {
        "tags": ["bigint", "sparse", "unbalanced"],
        "table": ((12, 0, 3, 1), (2, 10, 1, 2), (1, 3, 11, 2)),
    },
}
