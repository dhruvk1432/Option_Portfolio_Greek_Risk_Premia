"""Pure execution-cost and observed-quote sensitivity helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionScenarioResult:
    """Portfolio returns under modeled and quote-based cost scenarios."""

    modeled_return: float
    midpoint_return: float
    touch_price_return: float
    worst_case_return: float
    entry_coverage: float | None
    roundtrip_coverage: float | None
    evidence_label: str = "observed-quote sensitivity"

    def __post_init__(self) -> None:
        returns = (
            self.modeled_return,
            self.midpoint_return,
            self.touch_price_return,
            self.worst_case_return,
        )
        if not all(math.isfinite(value) for value in returns):
            raise ValueError("scenario returns must be finite")
        coverage = (self.entry_coverage, self.roundtrip_coverage)
        if (coverage[0] is None) != (coverage[1] is None):
            raise ValueError("entry and round-trip coverage must be reported together")
        if any(value is not None and not 0.0 <= value <= 1.0 for value in coverage):
            raise ValueError("coverage must lie between zero and one")
        if "fill" in self.evidence_label.lower():
            raise ValueError("touch-price scenarios cannot be labeled as realized fills")


def execution_scenarios(
    *,
    gross_return: float,
    modeled_cost: float,
    midpoint_cost: float,
    touch_cost: float,
    worst_cost: float,
    entry_coverage: float,
    roundtrip_coverage: float,
) -> ExecutionScenarioResult:
    """Subtract each cost estimate from the same gross portfolio return."""

    costs = (modeled_cost, midpoint_cost, touch_cost, worst_cost)
    if any(cost < 0.0 for cost in costs):
        raise ValueError("scenario costs must be nonnegative")
    return ExecutionScenarioResult(
        modeled_return=gross_return - modeled_cost,
        midpoint_return=gross_return - midpoint_cost,
        touch_price_return=gross_return - touch_cost,
        worst_case_return=gross_return - worst_cost,
        entry_coverage=entry_coverage,
        roundtrip_coverage=roundtrip_coverage,
    )
