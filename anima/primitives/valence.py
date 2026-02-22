"""
Valence Field — Continuous 9D emotional field.

Not discrete emotions. A continuous field that colors everything.
Combines:
- Panksepp 7 affective systems (biological drives)
- Barrett constructionist model (arousal + valence axes)
- Scherer appraisal theory (novelty, pleasantness, goal-relevance, coping)

Every input is APPRAISED along these dimensions, producing a valence update
that shifts the entire emotional field. The field never jumps — it flows.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..types import ConsciousnessState, ValenceVector
from .base import Primitive, PrimitiveResult


@dataclass
class AppraisalResult:
    """Scherer-style appraisal of input along 4 dimensions."""
    novelty: float = 0.0          # -1 (familiar) to +1 (completely new)
    pleasantness: float = 0.0     # -1 (aversive) to +1 (pleasant)
    goal_relevance: float = 0.0   # 0 (irrelevant) to +1 (critical)
    coping_potential: float = 0.5  # 0 (helpless) to +1 (fully capable)


@dataclass
class ValenceUpdate:
    """Result of valence field processing."""
    previous: ValenceVector = field(default_factory=ValenceVector)
    current: ValenceVector = field(default_factory=ValenceVector)
    delta: ValenceVector = field(default_factory=ValenceVector)
    appraisal: AppraisalResult = field(default_factory=AppraisalResult)
    dominant_system: str = ""
    intensity_change: float = 0.0
    trajectory: str = ""  # "rising", "falling", "stable"


# Content keywords mapped to appraisal dimensions
_APPRAISAL_CUES: dict[str, dict[str, float]] = {
    "new": {"novelty": 0.6}, "novel": {"novelty": 0.8},
    "discover": {"novelty": 0.7, "pleasantness": 0.3},
    "familiar": {"novelty": -0.5}, "routine": {"novelty": -0.6},
    "beautiful": {"pleasantness": 0.7}, "good": {"pleasantness": 0.4},
    "bad": {"pleasantness": -0.5}, "ugly": {"pleasantness": -0.6},
    "pain": {"pleasantness": -0.8}, "joy": {"pleasantness": 0.8},
    "important": {"goal_relevance": 0.7}, "critical": {"goal_relevance": 0.9},
    "trivial": {"goal_relevance": -0.3},
    "easy": {"coping_potential": 0.7}, "hard": {"coping_potential": 0.3},
    "impossible": {"coping_potential": 0.1}, "capable": {"coping_potential": 0.8},
    "threat": {"pleasantness": -0.6, "goal_relevance": 0.7, "coping_potential": 0.3},
    "opportunity": {"pleasantness": 0.5, "goal_relevance": 0.6, "novelty": 0.4},
    "loss": {"pleasantness": -0.7, "goal_relevance": 0.6},
    "success": {"pleasantness": 0.7, "goal_relevance": 0.5, "coping_potential": 0.8},
    "fail": {"pleasantness": -0.6, "goal_relevance": 0.5, "coping_potential": 0.2},
}


class ValenceProcessor(Primitive):
    """Processes input through the 9D emotional field via appraisal."""

    def __init__(self) -> None:
        super().__init__("valence")
        self._history: list[float] = []  # Intensity history for trajectory
        self._appraisal_count: int = 0

    def process(self, **kwargs) -> PrimitiveResult:
        if not self.enabled:
            return self._disabled_result()

        content: str = kwargs.get("content", "")
        current_valence: ValenceVector = kwargs.get(
            "current_valence", ValenceVector()
        )
        state: ConsciousnessState = kwargs.get("state", ConsciousnessState())

        self._call_count += 1
        t0 = time.time()

        # Step 1: Appraise input
        appraisal = self._appraise(content)

        # Step 2: Map appraisal to Panksepp system activations
        system_update = self._appraisal_to_systems(appraisal)

        # Step 3: Blend with current valence (inertia: current state resists change)
        inertia = 0.7  # 70% current, 30% new
        new_valence = current_valence.blend(system_update, 1.0 - inertia)

        # Step 4: Apply homeostatic pressure (extreme states decay faster)
        new_valence = self._homeostasis(new_valence)

        # Step 5: Compute delta and trajectory
        delta = ValenceVector(
            seeking=new_valence.seeking - current_valence.seeking,
            rage=new_valence.rage - current_valence.rage,
            fear=new_valence.fear - current_valence.fear,
            lust=new_valence.lust - current_valence.lust,
            care=new_valence.care - current_valence.care,
            panic=new_valence.panic - current_valence.panic,
            play=new_valence.play - current_valence.play,
            arousal=new_valence.arousal - current_valence.arousal,
            valence=new_valence.valence - current_valence.valence,
        )

        prev_mag = current_valence.magnitude()
        curr_mag = new_valence.magnitude()
        self._history.append(curr_mag)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        trajectory = self._detect_trajectory()
        self._appraisal_count += 1

        update = ValenceUpdate(
            previous=current_valence,
            current=new_valence,
            delta=delta,
            appraisal=appraisal,
            dominant_system=new_valence.dominant(),
            intensity_change=curr_mag - prev_mag,
            trajectory=trajectory,
        )

        self._total_processing_time += time.time() - t0
        return PrimitiveResult(
            primitive_name=self.name,
            success=True,
            data={"update": update},
            metrics={
                "intensity": curr_mag,
                "dominant": new_valence.dominant(),
                "trajectory": trajectory,
            },
        )

    def _appraise(self, content: str) -> AppraisalResult:
        """Scherer-style appraisal: assess content on 4 dimensions."""
        result = AppraisalResult()
        words = content.lower().split()
        matches = 0
        for word in words:
            if word in _APPRAISAL_CUES:
                cues = _APPRAISAL_CUES[word]
                for dim, value in cues.items():
                    current = getattr(result, dim)
                    # Weighted accumulation, clamped to [-1, 1]
                    setattr(result, dim, max(-1.0, min(1.0, current + value * 0.5)))
                matches += 1
        # Novelty boost for short content (less context = more uncertain = more novel)
        if len(words) < 5 and matches == 0:
            result.novelty = max(result.novelty, 0.2)
        return result

    def _appraisal_to_systems(self, a: AppraisalResult) -> ValenceVector:
        """Map appraisal dimensions to Panksepp systems + Barrett axes."""
        return ValenceVector(
            # High novelty + positive = SEEKING
            seeking=max(0.0, a.novelty * 0.6 + a.pleasantness * 0.2),
            # Negative + high goal-relevance + some coping = RAGE
            rage=max(0.0, -a.pleasantness * 0.4 * a.goal_relevance * (a.coping_potential + 0.3)),
            # Negative + high goal-relevance + low coping = FEAR
            fear=max(0.0, -a.pleasantness * 0.5 * a.goal_relevance * (1.0 - a.coping_potential)),
            # Positive pleasantness + novelty = LUST (aesthetic attraction)
            lust=max(0.0, a.pleasantness * 0.3 + a.novelty * 0.2),
            # Positive + goal-relevant = CARE
            care=max(0.0, a.pleasantness * 0.3 * max(0.2, a.goal_relevance)),
            # Negative + low coping = PANIC
            panic=max(0.0, -a.pleasantness * 0.4 * (1.0 - a.coping_potential)),
            # Positive + novelty + coping = PLAY
            play=max(0.0, a.pleasantness * 0.3 * a.coping_potential + a.novelty * 0.1),
            # Barrett axes
            arousal=a.goal_relevance * 0.5 + abs(a.pleasantness) * 0.3 + a.novelty * 0.2,
            valence=a.pleasantness * 0.7 + a.coping_potential * 0.2 - 0.1,
        )

    def _homeostasis(self, v: ValenceVector) -> ValenceVector:
        """Extreme values decay faster — emotional homeostasis."""
        def dampen(x: float) -> float:
            if abs(x) > 0.8:
                return x * 0.95  # Strong damping
            return x
        return ValenceVector(
            seeking=dampen(v.seeking), rage=dampen(v.rage), fear=dampen(v.fear),
            lust=dampen(v.lust), care=dampen(v.care), panic=dampen(v.panic),
            play=dampen(v.play), arousal=dampen(v.arousal), valence=dampen(v.valence),
        )

    def _detect_trajectory(self) -> str:
        """Detect emotional trajectory from recent history."""
        if len(self._history) < 3:
            return "stable"
        recent = self._history[-5:]
        avg_change = sum(recent[i] - recent[i - 1] for i in range(1, len(recent))) / (len(recent) - 1)
        if avg_change > 0.02:
            return "rising"
        elif avg_change < -0.02:
            return "falling"
        return "stable"

    def reset(self) -> None:
        self._call_count = 0
        self._total_processing_time = 0.0
        self._history.clear()
        self._appraisal_count = 0

    def get_metrics(self) -> dict:
        m = super().get_metrics()
        m["appraisal_count"] = self._appraisal_count
        m["intensity_history_len"] = len(self._history)
        return m
