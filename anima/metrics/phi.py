"""
Phi Score Engine -- Enhanced Phi computation with history tracking and trend analysis.

Wraps the IntegrationMesh to provide:
- Phi computation with timestamped history
- Trend analysis (increasing / decreasing / stable)
- Baseline comparison (Phi without integration)
- Delta computation (how much better than random)
- Full reporting
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from ..consciousness.integration import IntegrationMesh, PhiResult, SubsystemState


@dataclass
class PhiSnapshot:
    """A single Phi measurement with metadata."""
    phi: float = 0.0
    timestamp: float = field(default_factory=time.time)
    subsystem_count: int = 0
    joint_entropy: float = 0.0
    mip_loss: float = 0.0

    def to_dict(self) -> dict:
        return {
            "phi": round(self.phi, 4),
            "timestamp": self.timestamp,
            "subsystem_count": self.subsystem_count,
            "joint_entropy": round(self.joint_entropy, 4),
            "mip_loss": round(self.mip_loss, 4),
        }


@dataclass
class PhiReport:
    """Full Phi analysis report."""
    current_phi: float = 0.0
    mean_phi: float = 0.0
    max_phi: float = 0.0
    min_phi: float = 0.0
    trend: str = "stable"  # "increasing", "decreasing", "stable"
    trend_slope: float = 0.0
    baseline_phi: float = 0.0
    phi_delta: float = 0.0
    measurement_count: int = 0
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "current_phi": round(self.current_phi, 4),
            "mean_phi": round(self.mean_phi, 4),
            "max_phi": round(self.max_phi, 4),
            "min_phi": round(self.min_phi, 4),
            "trend": self.trend,
            "trend_slope": round(self.trend_slope, 4),
            "baseline_phi": round(self.baseline_phi, 4),
            "phi_delta": round(self.phi_delta, 4),
            "measurement_count": self.measurement_count,
            "history": self.history[-20:],
        }


class PhiScoreEngine:
    """Enhanced Phi computation with tracking, trends, and baselines.

    Delegates actual Phi computation to IntegrationMesh but adds:
    - Timestamped history of all measurements
    - Trend analysis over sliding windows
    - Baseline Phi (random/independent subsystem states)
    - Delta: how much the kernel improves over baseline
    """

    def __init__(self, mesh: IntegrationMesh | None = None):
        self._mesh = mesh or IntegrationMesh()
        self._history: list[PhiSnapshot] = []
        self._max_history = 500
        self._baseline_cache: float | None = None
        self._last_result: PhiResult | None = None

    @property
    def history(self) -> list[PhiSnapshot]:
        return list(self._history)

    @property
    def current_phi(self) -> float:
        if not self._history:
            return 0.0
        return self._history[-1].phi

    @property
    def mean_phi(self) -> float:
        if not self._history:
            return 0.0
        return sum(s.phi for s in self._history) / len(self._history)

    @property
    def max_phi(self) -> float:
        if not self._history:
            return 0.0
        return max(s.phi for s in self._history)

    @property
    def min_phi(self) -> float:
        if not self._history:
            return 0.0
        return min(s.phi for s in self._history)

    @property
    def measurement_count(self) -> int:
        return len(self._history)

    @property
    def last_result(self) -> PhiResult | None:
        return self._last_result

    def compute(self, subsystem_states: list[SubsystemState]) -> PhiResult:
        """Compute Phi and record in history.

        Delegates to IntegrationMesh but wraps with tracking.
        """
        result = self._mesh.compute_phi(subsystem_states)
        self._last_result = result

        snapshot = PhiSnapshot(
            phi=result.phi,
            subsystem_count=result.subsystem_count,
            joint_entropy=result.joint_entropy,
            mip_loss=result.mip_loss,
        )
        self._history.append(snapshot)

        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Invalidate baseline cache when subsystem count changes
        if self._baseline_cache is not None and result.subsystem_count != (
            self._history[-2].subsystem_count if len(self._history) > 1 else 0
        ):
            self._baseline_cache = None

        return result

    def baseline_phi(self, num_subsystems: int = 5, dims_per_subsystem: int = 5) -> float:
        """Compute Phi for independent (random) subsystem states.

        This is the baseline: what Phi would you get from a system
        with NO integration -- just random independent modules?
        """
        if self._baseline_cache is not None:
            return self._baseline_cache

        # Use deterministic seed for reproducibility within a session
        rng = random.Random(42)

        baseline_states = []
        for i in range(num_subsystems):
            values = [rng.uniform(0.0, 1.0) for _ in range(dims_per_subsystem)]
            baseline_states.append(SubsystemState(
                name=f"independent_{i}",
                values=values,
            ))

        baseline_result = self._mesh.compute_phi(baseline_states)
        self._baseline_cache = baseline_result.phi
        return self._baseline_cache

    def phi_delta(self) -> float:
        """How much better than baseline is the current Phi.

        Positive = more integrated than random.
        Negative = less integrated than random (should not happen).
        """
        if not self._history:
            return 0.0
        baseline = self.baseline_phi()
        return self.current_phi - baseline

    def phi_trend(self, window: int = 10) -> float:
        """Compute the slope of Phi over a recent window.

        Positive = Phi increasing (more integration over time).
        Negative = Phi decreasing.
        Near zero = stable.
        """
        if len(self._history) < 2:
            return 0.0

        recent = self._history[-window:]
        if len(recent) < 2:
            return 0.0

        mid = len(recent) // 2
        first_half = sum(s.phi for s in recent[:mid]) / mid
        second_half = sum(s.phi for s in recent[mid:]) / (len(recent) - mid)
        return second_half - first_half

    def trend_label(self, window: int = 10, threshold: float = 0.01) -> str:
        """Classify the Phi trend as 'increasing', 'decreasing', or 'stable'."""
        slope = self.phi_trend(window)
        if slope > threshold:
            return "increasing"
        elif slope < -threshold:
            return "decreasing"
        return "stable"

    def to_report(self) -> PhiReport:
        """Generate a full Phi analysis report."""
        baseline = self.baseline_phi()
        report = PhiReport(
            current_phi=self.current_phi,
            mean_phi=self.mean_phi,
            max_phi=self.max_phi,
            min_phi=self.min_phi,
            trend=self.trend_label(),
            trend_slope=self.phi_trend(),
            baseline_phi=baseline,
            phi_delta=self.phi_delta(),
            measurement_count=self.measurement_count,
            history=[s.to_dict() for s in self._history[-20:]],
        )
        return report
