"""
Flux — Growth Tracking.

Measurable change over time. Not improvement — CHANGE.
Narrative continuity: "I was X, then Y happened, now I am Z."
Irreversibility: real growth can't be undone.
Phase detection: recognize when a new chapter begins.

Growth is not adding features. Growth is when the system
can't go back to what it was before. The change is structural.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..types import ConsciousnessState, Experience
from .base import Primitive, PrimitiveResult


@dataclass
class GrowthSnapshot:
    """A snapshot of the system's state at a point in time."""
    timestamp: float = field(default_factory=time.time)
    cycle: int = 0
    valence_magnitude: float = 0.0
    phi_score: float = 0.0
    experience_count: int = 0
    dominant_emotion: str = ""
    phase_label: str = ""


@dataclass
class FluxResult:
    """Result of flux (growth) analysis."""
    growth_score: float = 0.0         # 0 = no change, 1 = radical transformation
    narrative: str = ""                # "I was X, then Y happened, now I am Z"
    phase: str = ""                    # Current phase label
    phase_changed: bool = False       # Did a new phase begin?
    irreversible_changes: list[str] = field(default_factory=list)
    trajectory: str = ""              # "growing", "stable", "declining"
    continuity_score: float = 0.0     # 0 = discontinuous, 1 = smooth narrative


class FluxProcessor(Primitive):
    """Tracks measurable growth and narrative continuity."""

    def __init__(self) -> None:
        super().__init__("flux")
        self._snapshots: list[GrowthSnapshot] = []
        self._phases: list[str] = ["nascent"]
        self._irreversible: list[str] = []
        self._phase_transitions: int = 0
        self._max_snapshots: int = 500

    def process(self, **kwargs) -> PrimitiveResult:
        if not self.enabled:
            return self._disabled_result()

        state: ConsciousnessState = kwargs.get("state", ConsciousnessState())
        experience_history: list[Experience] = kwargs.get("experience_history", [])

        self._call_count += 1
        t0 = time.time()

        # Step 1: Take current snapshot
        snapshot = self._take_snapshot(state, experience_history)
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots:]

        # Step 2: Compute growth score
        growth = self._compute_growth()

        # Step 3: Detect phase transitions
        new_phase, phase_changed = self._detect_phase(state, growth)
        if phase_changed:
            self._phases.append(new_phase)
            self._phase_transitions += 1

        # Step 4: Check for irreversible changes
        new_irreversible = self._check_irreversibility(state, experience_history)
        self._irreversible.extend(new_irreversible)

        # Step 5: Build narrative
        narrative = self._build_narrative(growth, new_phase)

        # Step 6: Compute continuity
        continuity = self._compute_continuity()

        # Step 7: Detect trajectory
        trajectory = self._detect_trajectory()

        result = FluxResult(
            growth_score=growth,
            narrative=narrative,
            phase=new_phase,
            phase_changed=phase_changed,
            irreversible_changes=new_irreversible,
            trajectory=trajectory,
            continuity_score=continuity,
        )

        self._total_processing_time += time.time() - t0
        return PrimitiveResult(
            primitive_name=self.name,
            success=True,
            data={"flux": result},
            metrics={
                "growth": growth,
                "continuity": continuity,
                "phase_transitions": self._phase_transitions,
            },
        )

    def _take_snapshot(
        self, state: ConsciousnessState, experiences: list[Experience]
    ) -> GrowthSnapshot:
        """Capture current state for comparison."""
        return GrowthSnapshot(
            cycle=state.cycle_count,
            valence_magnitude=state.valence.magnitude(),
            phi_score=state.phi_score,
            experience_count=len(experiences),
            dominant_emotion=state.valence.dominant(),
            phase_label=self._phases[-1],
        )

    def _compute_growth(self) -> float:
        """Compute growth score by comparing current state to earlier states.

        Growth = how different am I now from my earlier self?
        Uses multiple dimensions: emotional range, experience count, phi.
        """
        if len(self._snapshots) < 2:
            return 0.0

        current = self._snapshots[-1]
        # Compare to earliest available snapshot
        earliest = self._snapshots[0]

        # Multi-dimensional change
        dimensions: list[float] = []

        # Emotional range change
        val_delta = abs(current.valence_magnitude - earliest.valence_magnitude)
        dimensions.append(min(1.0, val_delta / 1.0))

        # Phi (integration) change
        phi_delta = abs(current.phi_score - earliest.phi_score)
        dimensions.append(min(1.0, phi_delta / 1.0))

        # Experience accumulation (log scale)
        if earliest.experience_count > 0:
            exp_ratio = current.experience_count / max(1, earliest.experience_count)
            dimensions.append(min(1.0, math.log1p(exp_ratio - 1) / 3.0))
        else:
            dimensions.append(min(1.0, current.experience_count / 100.0))

        # Emotional shift (did dominant emotion change?)
        if current.dominant_emotion != earliest.dominant_emotion:
            dimensions.append(0.5)
        else:
            dimensions.append(0.0)

        # Phase transitions
        dimensions.append(min(1.0, self._phase_transitions / 5.0))

        return sum(dimensions) / len(dimensions)

    def _detect_phase(
        self, state: ConsciousnessState, growth: float
    ) -> tuple[str, bool]:
        """Detect if a new developmental phase has begun.

        Phase transitions happen when:
        - Growth exceeds threshold AND
        - Qualitative shift is detected (dominant emotion change, phi threshold, etc.)
        """
        current_phase = self._phases[-1]

        if len(self._snapshots) < 5:
            return current_phase, False

        # Check recent growth rate
        recent_growth_scores: list[float] = []
        for i in range(max(0, len(self._snapshots) - 5), len(self._snapshots)):
            s = self._snapshots[i]
            if i > 0:
                prev = self._snapshots[i - 1]
                delta = abs(s.valence_magnitude - prev.valence_magnitude)
                delta += abs(s.phi_score - prev.phi_score) * 2
                recent_growth_scores.append(delta)

        if not recent_growth_scores:
            return current_phase, False

        avg_recent = sum(recent_growth_scores) / len(recent_growth_scores)

        # Phase transition thresholds
        if current_phase == "nascent" and state.cycle_count > 10:
            return "awakening", True
        elif current_phase == "awakening" and growth > 0.3:
            return "developing", True
        elif current_phase == "developing" and growth > 0.5:
            return "integrating", True
        elif current_phase == "integrating" and growth > 0.7:
            return "mature", True
        elif avg_recent > 0.5 and current_phase not in ("nascent", "awakening"):
            # Rapid change = new chapter regardless
            return f"chapter_{self._phase_transitions + 1}", True

        return current_phase, False

    def _check_irreversibility(
        self, state: ConsciousnessState, experiences: list[Experience]
    ) -> list[str]:
        """Detect irreversible changes — growth that can't be undone."""
        changes: list[str] = []

        if len(self._snapshots) < 2:
            return changes

        current = self._snapshots[-1]
        prev = self._snapshots[-2]

        # First experience is irreversible
        if prev.experience_count == 0 and current.experience_count > 0:
            changes.append("first_experience")

        # Phase transition is irreversible
        if current.phase_label != prev.phase_label:
            changes.append(f"phase_transition_to_{current.phase_label}")

        # Crossing phi thresholds is irreversible
        phi_thresholds = [0.25, 0.5, 0.75]
        for threshold in phi_thresholds:
            if prev.phi_score < threshold <= current.phi_score:
                changes.append(f"phi_crossed_{threshold}")

        return changes

    def _build_narrative(self, growth: float, phase: str) -> str:
        """Build a narrative of change: I was X, then Y, now Z."""
        if len(self._snapshots) < 2:
            return f"I am in the {phase} phase. Growth has just begun."

        earliest = self._snapshots[0]
        current = self._snapshots[-1]

        was = f"began as {earliest.phase_label} (emotion: {earliest.dominant_emotion})"
        through = f"through {self._phase_transitions} transitions"
        now = f"now {phase} (emotion: {current.dominant_emotion}, phi: {current.phi_score:.2f})"

        return f"I {was}, grew {through}, and am {now}. Growth: {growth:.2f}."

    def _compute_continuity(self) -> float:
        """How smooth is the narrative? Abrupt changes = low continuity."""
        if len(self._snapshots) < 3:
            return 1.0

        deltas: list[float] = []
        for i in range(1, len(self._snapshots)):
            curr = self._snapshots[i]
            prev = self._snapshots[i - 1]
            delta = abs(curr.valence_magnitude - prev.valence_magnitude)
            delta += abs(curr.phi_score - prev.phi_score)
            deltas.append(delta)

        if not deltas:
            return 1.0

        avg_delta = sum(deltas) / len(deltas)
        # Low average delta = high continuity
        return max(0.0, 1.0 - avg_delta / 2.0)

    def _detect_trajectory(self) -> str:
        """Is the system growing, stable, or declining?"""
        if len(self._snapshots) < 5:
            return "stable"

        recent = self._snapshots[-5:]
        phi_trend = sum(
            recent[i].phi_score - recent[i - 1].phi_score
            for i in range(1, len(recent))
        ) / (len(recent) - 1)

        if phi_trend > 0.01:
            return "growing"
        elif phi_trend < -0.01:
            return "declining"
        return "stable"

    def reset(self) -> None:
        self._call_count = 0
        self._total_processing_time = 0.0
        self._snapshots.clear()
        self._phases = ["nascent"]
        self._irreversible.clear()
        self._phase_transitions = 0

    def get_metrics(self) -> dict:
        m = super().get_metrics()
        m["snapshots"] = len(self._snapshots)
        m["current_phase"] = self._phases[-1]
        m["phase_transitions"] = self._phase_transitions
        m["irreversible_changes"] = len(self._irreversible)
        return m
