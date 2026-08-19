"""Frozen production choices for the exact Fisher network."""

from __future__ import annotations

from typing import Sequence

from .optimized import EngineResult, exact_state_network


def exact_production_network(
    observed: Sequence[Sequence[int]],
    *,
    collect_certificate: bool = False,
) -> EngineResult:
    """Run the measured production winner.

    The global longest recurrence is shared across states.  The engine uses
    the exact universal continuation lower bound one and does not build the
    shortest recurrence.  Strict-Monge daughter batching remains available
    as an auditable opt-in because it has not passed the wall-time gate.
    """

    return exact_state_network(
        observed,
        batching=False,
        extrema_cache="global",
        reject_pruning=False,
        collect_certificate=collect_certificate,
    )
