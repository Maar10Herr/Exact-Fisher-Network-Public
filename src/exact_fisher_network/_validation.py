"""Shared validation for exact-network public inputs."""

from __future__ import annotations

from operator import index


def integer(value: object, label: str) -> int:
    """Return an integer-like value without coercing invalid inputs."""

    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not a boolean.")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError(f"{label} must be an integer.") from error
