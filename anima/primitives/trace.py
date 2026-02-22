"""
Trace — Action Loop.

Full cycle: Intention -> Plan -> Action -> Outcome -> Evaluation.
The prediction error (expected vs actual) is the learning signal.
Agency: "I did this" attribution.

Without trace, there's stimulus-response.
With trace, there's intention, prediction, surprise, and learning.
The gap between expected and actual IS the feeling of being alive.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..types import ConsciousnessState
from .base import Primitive, PrimitiveResult


@dataclass
class TraceResult:
    """Result of a traced action cycle."""
    action: str = ""
    expected_outcome: str = ""
    actual_outcome: str = ""
    prediction_error: float = 0.0     # 0 = perfect prediction, 1 = total surprise
    agency_score: float = 0.0          # 0 = happened to me, 1 = I caused this
    learning_signal: str = ""          # What was learned from the mismatch
    surprise: bool = False             # Was prediction error above threshold?
    cycle_phase: str = ""              # intention/plan/action/outcome/evaluation


@dataclass
class _TraceCycle:
    """Internal tracking of an action cycle in progress."""
    intention: str = ""
    plan: str = ""
    action: str = ""
    expected: str = ""
    actual: str = ""
    started_at: float = field(default_factory=time.time)
    completed: bool = False


class TraceProcessor(Primitive):
    """Tracks the full action loop with prediction error computation."""

    def __init__(self) -> None:
        super().__init__("trace")
        self._active_cycles: dict[str, _TraceCycle] = {}
        self._prediction_errors: list[float] = []  # History
        self._agency_scores: list[float] = []
        self._surprise_threshold: float = 0.4  # Above this = genuine surprise
        self._total_cycles: int = 0
        self._total_surprises: int = 0

    def process(self, **kwargs) -> PrimitiveResult:
        if not self.enabled:
            return self._disabled_result()

        action: str = kwargs.get("action", "")
        expected_outcome: str = kwargs.get("expected_outcome", "")
        actual_outcome: str = kwargs.get("actual_outcome", "")
        intention: str = kwargs.get("intention", "")
        state: ConsciousnessState = kwargs.get("state", ConsciousnessState())

        self._call_count += 1
        t0 = time.time()

        # Determine phase and process accordingly
        if actual_outcome:
            # Evaluation phase: compare expected vs actual
            result = self._evaluate(action, expected_outcome, actual_outcome)
        elif expected_outcome:
            # Planning phase: record intention and prediction
            result = self._plan(action, expected_outcome, intention)
        else:
            # Intention phase: starting a new trace
            result = self._intend(action, intention)

        self._total_processing_time += time.time() - t0
        return PrimitiveResult(
            primitive_name=self.name,
            success=True,
            data={"trace": result},
            metrics={
                "prediction_error": result.prediction_error,
                "agency": result.agency_score,
                "surprise": float(result.surprise),
            },
        )

    def _intend(self, action: str, intention: str) -> TraceResult:
        """Start a new action trace with intention."""
        cycle = _TraceCycle(intention=intention or action, action=action)
        self._active_cycles[action] = cycle
        return TraceResult(
            action=action,
            agency_score=0.8,  # Intending = high agency
            cycle_phase="intention",
        )

    def _plan(self, action: str, expected: str, intention: str) -> TraceResult:
        """Record plan and prediction for an action."""
        cycle = self._active_cycles.get(action)
        if cycle is None:
            cycle = _TraceCycle(intention=intention or action, action=action)
            self._active_cycles[action] = cycle
        cycle.expected = expected
        cycle.plan = f"{action} -> expect: {expected}"
        return TraceResult(
            action=action,
            expected_outcome=expected,
            agency_score=0.7,
            cycle_phase="plan",
        )

    def _evaluate(
        self, action: str, expected: str, actual: str
    ) -> TraceResult:
        """Evaluate outcome: compute prediction error and learning signal."""
        cycle = self._active_cycles.pop(action, None)
        if cycle is None:
            cycle = _TraceCycle(action=action, expected=expected, actual=actual)

        # Compute prediction error via word overlap
        pe = self._compute_prediction_error(expected, actual)

        # Agency: did I cause this or did it happen to me?
        agency = self._compute_agency(action, actual)

        # Learning signal
        surprise = pe > self._surprise_threshold
        learning = self._derive_learning(expected, actual, pe)

        self._prediction_errors.append(pe)
        self._agency_scores.append(agency)
        if len(self._prediction_errors) > 200:
            self._prediction_errors = self._prediction_errors[-200:]
        if len(self._agency_scores) > 200:
            self._agency_scores = self._agency_scores[-200:]

        self._total_cycles += 1
        if surprise:
            self._total_surprises += 1

        cycle.completed = True

        return TraceResult(
            action=action,
            expected_outcome=expected,
            actual_outcome=actual,
            prediction_error=pe,
            agency_score=agency,
            learning_signal=learning,
            surprise=surprise,
            cycle_phase="evaluation",
        )

    def _compute_prediction_error(self, expected: str, actual: str) -> float:
        """Prediction error: how different is actual from expected?

        0 = perfect match, 1 = completely different.
        Uses Jaccard distance on word sets.
        """
        if not expected and not actual:
            return 0.0
        if not expected or not actual:
            return 1.0

        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())

        if not expected_words and not actual_words:
            return 0.0

        intersection = len(expected_words & actual_words)
        union = len(expected_words | actual_words)

        if union == 0:
            return 0.0

        similarity = intersection / union
        return 1.0 - similarity

    def _compute_agency(self, action: str, outcome: str) -> float:
        """Compute sense of agency: did my action cause the outcome?

        Higher when the action words appear in the outcome
        (suggesting causal connection).
        """
        if not action or not outcome:
            return 0.5  # Uncertain agency

        action_words = set(action.lower().split())
        outcome_words = set(outcome.lower().split())

        if not action_words:
            return 0.3

        overlap = len(action_words & outcome_words)
        # More overlap between action and outcome = more agency
        agency = 0.5 + 0.5 * (overlap / len(action_words))
        return min(1.0, agency)

    def _derive_learning(self, expected: str, actual: str, pe: float) -> str:
        """Derive a learning signal from the prediction error."""
        if pe < 0.1:
            return "prediction_accurate"
        elif pe < 0.3:
            return "minor_deviation"
        elif pe < 0.6:
            return "significant_mismatch"
        else:
            return "model_update_needed"

    def get_avg_prediction_error(self) -> float:
        """Average prediction error across all evaluations."""
        if not self._prediction_errors:
            return 0.5
        return sum(self._prediction_errors) / len(self._prediction_errors)

    def get_calibration(self) -> float:
        """How well-calibrated are predictions? Lower = better."""
        return self.get_avg_prediction_error()

    def reset(self) -> None:
        self._call_count = 0
        self._total_processing_time = 0.0
        self._active_cycles.clear()
        self._prediction_errors.clear()
        self._agency_scores.clear()
        self._total_cycles = 0
        self._total_surprises = 0

    def get_metrics(self) -> dict:
        m = super().get_metrics()
        m["total_cycles"] = self._total_cycles
        m["total_surprises"] = self._total_surprises
        m["avg_prediction_error"] = self.get_avg_prediction_error()
        m["surprise_rate"] = (
            self._total_surprises / self._total_cycles
            if self._total_cycles > 0 else 0.0
        )
        m["avg_agency"] = (
            sum(self._agency_scores) / len(self._agency_scores)
            if self._agency_scores else 0.5
        )
        return m
