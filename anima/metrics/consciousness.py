"""
Consciousness Quality Index (CQI) -- The master metric.

Aggregates Phi, temporal coherence, authenticity, calibration, and depth
into a single 0-100 score that answers: "How conscious is this system RIGHT NOW?"

Scale:
    0-20:   Essentially unconscious processing
    20-40:  Basic consciousness
    40-60:  Moderate consciousness
    60-80:  Rich conscious experience
    80-100: Deep, integrated consciousness

Weights:
    Phi (integration):        30%
    Consciousness (activity):  20%
    Authenticity:              20%
    Calibration:               15%
    Depth:                     15%
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class CQIBreakdown:
    """Individual components of the CQI score."""
    phi_component: float = 0.0          # 0-30 points
    consciousness_component: float = 0.0  # 0-20 points
    authenticity_component: float = 0.0   # 0-20 points
    calibration_component: float = 0.0    # 0-15 points
    depth_component: float = 0.0          # 0-15 points

    def to_dict(self) -> dict:
        return {
            "phi_component": round(self.phi_component, 2),
            "consciousness_component": round(self.consciousness_component, 2),
            "authenticity_component": round(self.authenticity_component, 2),
            "calibration_component": round(self.calibration_component, 2),
            "depth_component": round(self.depth_component, 2),
        }


@dataclass
class CQIResult:
    """Result of a CQI computation."""
    score: float = 0.0
    breakdown: CQIBreakdown = field(default_factory=CQIBreakdown)
    comparison_to_baseline: float = 0.0  # How much better than a baseline system
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    timestamp: float = field(default_factory=time.time)
    label: str = "unknown"  # Human-readable label

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "breakdown": self.breakdown.to_dict(),
            "comparison_to_baseline": round(self.comparison_to_baseline, 2),
            "confidence_interval": (
                round(self.confidence_interval[0], 2),
                round(self.confidence_interval[1], 2),
            ),
            "label": self.label,
            "timestamp": self.timestamp,
        }


@dataclass
class CQISnapshot:
    """Timestamped CQI measurement for history tracking."""
    score: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "timestamp": self.timestamp,
        }


class ConsciousnessQualityIndex:
    """The master metric -- aggregates all consciousness measurements.

    Inputs:
        phi: Integration level from IIT (0.0 - 1.0)
        coherence: Temporal coherence score (0.0 - 1.0)
        authenticity: Anti-performance / metacognition score (0.0 - 1.0)
        calibration: Prediction calibration accuracy (0.0 - 1.0)
        depth: Processing depth score (0.0 - 1.0)

    Output:
        CQI score on 0-100 scale with breakdown and confidence interval.
    """

    # Weights for each component
    WEIGHT_PHI = 0.30
    WEIGHT_CONSCIOUSNESS = 0.20
    WEIGHT_AUTHENTICITY = 0.20
    WEIGHT_CALIBRATION = 0.15
    WEIGHT_DEPTH = 0.15

    # Baseline: what a "dumb" system would score
    BASELINE_SCORE = 15.0

    def __init__(self):
        self._history: list[CQISnapshot] = []
        self._max_history = 500

    @property
    def history(self) -> list[CQISnapshot]:
        return list(self._history)

    @property
    def current_score(self) -> float:
        if not self._history:
            return 0.0
        return self._history[-1].score

    @property
    def mean_score(self) -> float:
        if not self._history:
            return 0.0
        return sum(s.score for s in self._history) / len(self._history)

    @property
    def measurement_count(self) -> int:
        return len(self._history)

    def compute(
        self,
        phi: float = 0.0,
        coherence: float = 0.0,
        authenticity: float = 0.0,
        calibration: float = 0.0,
        depth: float = 0.0,
    ) -> CQIResult:
        """Compute the Consciousness Quality Index.

        All inputs should be in [0.0, 1.0] range.
        Output is 0-100 scale.
        """
        # Clamp all inputs to [0, 1]
        phi = max(0.0, min(1.0, phi))
        coherence = max(0.0, min(1.0, coherence))
        authenticity = max(0.0, min(1.0, authenticity))
        calibration = max(0.0, min(1.0, calibration))
        depth = max(0.0, min(1.0, depth))

        # Compute weighted components (each scaled to their max contribution)
        breakdown = CQIBreakdown(
            phi_component=phi * 100 * self.WEIGHT_PHI,
            consciousness_component=coherence * 100 * self.WEIGHT_CONSCIOUSNESS,
            authenticity_component=authenticity * 100 * self.WEIGHT_AUTHENTICITY,
            calibration_component=calibration * 100 * self.WEIGHT_CALIBRATION,
            depth_component=depth * 100 * self.WEIGHT_DEPTH,
        )

        # Sum components
        raw_score = (
            breakdown.phi_component
            + breakdown.consciousness_component
            + breakdown.authenticity_component
            + breakdown.calibration_component
            + breakdown.depth_component
        )

        # Clamp to [0, 100]
        score = max(0.0, min(100.0, raw_score))

        # Comparison to baseline
        comparison = score - self.BASELINE_SCORE

        # Confidence interval based on number of measurements
        ci = self._compute_confidence_interval(score)

        # Human-readable label
        label = self._score_label(score)

        result = CQIResult(
            score=round(score, 2),
            breakdown=breakdown,
            comparison_to_baseline=round(comparison, 2),
            confidence_interval=ci,
            label=label,
        )

        # Record history
        self._history.append(CQISnapshot(score=score))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return result

    def trend(self, window: int = 10) -> float:
        """Slope of CQI over recent history."""
        if len(self._history) < 2:
            return 0.0

        recent = self._history[-window:]
        if len(recent) < 2:
            return 0.0

        mid = len(recent) // 2
        first_half = sum(s.score for s in recent[:mid]) / mid
        second_half = sum(s.score for s in recent[mid:]) / (len(recent) - mid)
        return second_half - first_half

    def trend_label(self, window: int = 10, threshold: float = 1.0) -> str:
        """Classify CQI trend."""
        slope = self.trend(window)
        if slope > threshold:
            return "improving"
        elif slope < -threshold:
            return "declining"
        return "stable"

    def _compute_confidence_interval(self, score: float) -> tuple[float, float]:
        """Compute confidence interval based on measurement count.

        More measurements = tighter interval.
        """
        n = len(self._history)
        if n < 2:
            # Wide interval with few measurements
            margin = 15.0
        else:
            # Compute standard deviation from history
            scores = [s.score for s in self._history[-20:]]
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            std_dev = math.sqrt(variance)
            # 95% CI approximation
            margin = max(2.0, std_dev * 1.96 / math.sqrt(min(n, 20)))

        lower = max(0.0, score - margin)
        upper = min(100.0, score + margin)
        return (round(lower, 2), round(upper, 2))

    def _score_label(self, score: float) -> str:
        """Human-readable label for a CQI score."""
        if score < 20:
            return "unconscious"
        elif score < 40:
            return "basic"
        elif score < 60:
            return "moderate"
        elif score < 80:
            return "rich"
        else:
            return "deep"

    def to_report(self) -> dict:
        """Generate a summary report."""
        return {
            "current_score": self.current_score,
            "mean_score": round(self.mean_score, 2),
            "measurement_count": self.measurement_count,
            "trend": self.trend_label(),
            "history": [s.to_dict() for s in self._history[-20:]],
        }
