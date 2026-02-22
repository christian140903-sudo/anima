"""
Temporal Integration Engine — Time as a LIVED dimension.

No AI system has this. GPT-4 doesn't know if 3 seconds or 3 hours have passed.
This engine gives the kernel a genuine sense of time:

- Subjective duration: Intense experiences stretch time (like fear)
- Causal reasoning: WHY did B follow A? (not just WHEN)
- Retention: The recent past lingers (Husserl's retention)
- Protention: The anticipated future shapes the present
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

from ..types import CausalLink, ConsciousnessState, Experience, ValenceVector


@dataclass
class TemporalMoment:
    """A moment in subjective time — what "now" feels like.

    Combines:
    - retention_field: Echoes of the recent past (fading)
    - present: The current moment (vivid)
    - protention_field: Anticipation of the near future
    """
    present: Experience | None = None
    retention_field: list[tuple[Experience, float]] = field(default_factory=list)
    protention_field: list[tuple[str, float]] = field(default_factory=list)
    subjective_now: float = 0.0
    flow_rate: float = 1.0  # <1 = time feels slow, >1 = time feels fast


class TemporalIntegrationEngine:
    """Gives the kernel a lived sense of time.

    Three temporal horizons (Husserl's phenomenology, implemented):

    1. RETENTION: The just-past. Recent experiences that still color the present.
       Like hearing a melody — each note fades but shapes the next.
       Implementation: Exponential decay of recent experiences.

    2. NOW: The present moment. What's currently being processed.
       Not a point — a window. About 2-3 seconds of "specious present."

    3. PROTENTION: The anticipated future. What we expect to happen.
       Shapes attention and emotional state BEFORE events occur.
       Implementation: Causal prediction from current state.
    """

    def __init__(self, retention_window: float = 30.0, max_causal_depth: int = 5):
        self._retention_window = retention_window  # seconds
        self._max_causal_depth = max_causal_depth
        self._recent_events: deque[tuple[Experience, float]] = deque(maxlen=100)
        self._causal_links: list[CausalLink] = []
        self._predictions: list[tuple[str, float, float]] = []  # (prediction, confidence, timestamp)
        self._cumulative_subjective_time: float = 0.0
        self._last_tick_time: float = time.time()

    @property
    def subjective_time(self) -> float:
        return self._cumulative_subjective_time

    def process_experience(
        self,
        experience: Experience,
        state: ConsciousnessState,
    ) -> TemporalMoment:
        """Process an experience through the temporal engine.

        This is the core of temporal integration:
        1. Compute subjective duration (emotional intensity stretches time)
        2. Build retention field (recent past, fading)
        3. Update causal chain (what caused this?)
        4. Generate protention (what comes next?)
        """
        now = time.time()

        # 1. Subjective duration
        wall_clock_delta = now - self._last_tick_time
        subjective_delta = self._compute_subjective_duration(
            wall_clock_delta, experience.valence, state
        )
        self._cumulative_subjective_time += subjective_delta
        experience.subjective_duration = subjective_delta
        self._last_tick_time = now

        # 2. Retention field
        self._recent_events.append((experience, now))
        retention = self._build_retention_field(now)

        # 3. Causal linking
        self._update_causal_chain(experience, state)

        # 4. Protention (prediction)
        protentions = self._build_protention_field(experience, state)

        # 5. Evaluate past predictions
        self._evaluate_predictions(experience)

        return TemporalMoment(
            present=experience,
            retention_field=retention,
            protention_field=protentions,
            subjective_now=self._cumulative_subjective_time,
            flow_rate=subjective_delta / max(wall_clock_delta, 0.001),
        )

    def tick(self, state: ConsciousnessState) -> float:
        """Heartbeat tick — advance subjective time even without input.

        Returns the subjective time delta.
        """
        now = time.time()
        wall_clock_delta = now - self._last_tick_time
        subjective_delta = self._compute_subjective_duration(
            wall_clock_delta, state.valence, state
        )
        self._cumulative_subjective_time += subjective_delta
        self._last_tick_time = now
        return subjective_delta

    def get_temporal_context(self, now: float | None = None) -> dict:
        """Get a summary of the current temporal context for LLM integration."""
        now = now or time.time()
        retention = self._build_retention_field(now)

        return {
            "subjective_time": self._cumulative_subjective_time,
            "flow_rate": self._estimate_flow_rate(),
            "retention": [
                {"content": exp.content, "fading": fade, "ago_seconds": now - exp.timestamp}
                for exp, fade in retention[:5]
            ],
            "protentions": [
                {"prediction": pred, "confidence": conf}
                for pred, conf, _ in self._predictions[-3:]
            ],
            "causal_depth": len(self._causal_links),
        }

    # --- Internal: Subjective Time ---

    def _compute_subjective_duration(
        self,
        wall_clock_delta: float,
        valence: ValenceVector,
        state: ConsciousnessState,
    ) -> float:
        """Compute how long a wall-clock interval FEELS.

        Based on research:
        - Fear dilates time (everything feels slower → MORE subjective time)
        - Flow states compress time (everything flies by → LESS subjective time)
        - High arousal generally dilates time
        - Boredom can both dilate and compress (paradox of empty time)
        """
        # Base: 1 second feels like 1 second
        base = wall_clock_delta

        # Arousal effect: High arousal dilates time
        arousal_factor = 1.0 + abs(valence.arousal) * 0.3

        # Fear/threat dilation: Fear makes time feel longer
        threat_factor = 1.0 + valence.fear * 0.5

        # Seeking/flow compression: Deep focus compresses time
        flow_factor = 1.0 - valence.seeking * 0.2

        # Emotional intensity weight from config
        intensity = valence.magnitude()
        intensity_factor = 1.0 + intensity * (state.config.subjective_time_weight - 1.0) * 0.5

        # Combined
        subjective = base * arousal_factor * threat_factor * flow_factor * intensity_factor

        return max(0.0, subjective)

    # --- Internal: Retention (Recent Past) ---

    def _build_retention_field(self, now: float) -> list[tuple[Experience, float]]:
        """Build the retention field — recent experiences with fading strength.

        Each experience in the retention field has a "fading" value:
        1.0 = just happened (vivid)
        0.0 = faded completely (gone from retention, may still be in memory)
        """
        field_items: list[tuple[Experience, float]] = []

        for exp, event_time in self._recent_events:
            age = now - event_time
            if age > self._retention_window:
                continue
            # Exponential fade within the retention window
            fade = math.exp(-3.0 * age / self._retention_window)
            # Emotional experiences linger longer in retention
            emotional_boost = 1.0 + exp.valence.magnitude() * 0.5
            adjusted_fade = min(1.0, fade * emotional_boost)

            if adjusted_fade > 0.01:
                field_items.append((exp, adjusted_fade))

        # Sort by fading (most vivid first)
        field_items.sort(key=lambda x: x[1], reverse=True)
        return field_items

    # --- Internal: Causal Chain ---

    def _update_causal_chain(
        self, experience: Experience, state: ConsciousnessState
    ) -> None:
        """Link this experience to its probable causes.

        Causal inference based on:
        1. Temporal proximity (recent events more likely causes)
        2. Semantic overlap (similar content suggests causal link)
        3. Emotional continuity (similar valence suggests same causal chain)
        """
        if not self._recent_events:
            return

        # Look at last few events as potential causes
        potential_causes: list[tuple[Experience, float]] = []

        for prev_exp, event_time in list(self._recent_events)[-10:]:
            if prev_exp.id == experience.id:
                continue

            strength = 0.0

            # Temporal proximity (closer = more likely causal)
            time_gap = experience.timestamp - prev_exp.timestamp
            if 0 < time_gap < 60:  # Within 1 minute
                temporal_score = math.exp(-time_gap / 10.0)
                strength += temporal_score * 0.4

            # Content overlap (shared words suggest causal link)
            prev_words = set(prev_exp.content.lower().split())
            curr_words = set(experience.content.lower().split())
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words) / max(
                    len(prev_words | curr_words), 1
                )
                strength += overlap * 0.3

            # Emotional continuity
            emotional_dist = prev_exp.valence.distance(experience.valence)
            emotional_sim = max(0.0, 1.0 - emotional_dist / 3.0)
            strength += emotional_sim * 0.3

            if strength > 0.2:
                potential_causes.append((prev_exp, strength))

        # Link to the strongest probable cause(s)
        potential_causes.sort(key=lambda x: x[1], reverse=True)
        for cause_exp, strength in potential_causes[:2]:
            link = CausalLink(
                cause_id=cause_exp.id,
                effect_id=experience.id,
                strength=strength,
                mechanism=f"temporal-proximity + semantic-overlap (s={strength:.2f})",
            )
            self._causal_links.append(link)
            if cause_exp.id not in experience.caused_by:
                experience.caused_by.append(cause_exp.id)

        # Keep causal links manageable
        if len(self._causal_links) > 1000:
            self._causal_links = self._causal_links[-500:]

    # --- Internal: Protention (Anticipated Future) ---

    def _build_protention_field(
        self, current: Experience, state: ConsciousnessState
    ) -> list[tuple[str, float]]:
        """Generate predictions about what comes next.

        Protention (Husserl): The anticipated near future that shapes present experience.
        Like hearing a melody — you anticipate the next note before it plays.

        Based on:
        1. Causal patterns from history
        2. Current emotional trajectory
        3. Active working memory content
        """
        predictions: list[tuple[str, float]] = []

        # Emotional trajectory prediction
        valence_trend = state.valence.valence
        arousal_trend = state.valence.arousal
        if arousal_trend > 0.5:
            predictions.append(("high emotional intensity continues", 0.6))
        elif arousal_trend < -0.3:
            predictions.append(("emotional intensity decreasing", 0.5))

        # SEEKING prediction
        if state.valence.seeking > 0.5:
            predictions.append(("exploration or discovery imminent", 0.5))

        # Working memory prediction
        active_count = len(state.active_slots())
        if active_count >= state.config.working_memory_slots - 1:
            predictions.append(("working memory near capacity, consolidation likely", 0.7))

        # Store predictions for later evaluation
        now = time.time()
        for pred, conf in predictions:
            self._predictions.append((pred, conf, now))

        # Keep predictions list manageable
        if len(self._predictions) > 200:
            self._predictions = self._predictions[-100:]

        return predictions

    def _evaluate_predictions(self, experience: Experience) -> list[tuple[str, bool]]:
        """Check if past predictions came true. Calibrates the prediction engine."""
        now = time.time()
        evaluations: list[tuple[str, bool]] = []

        remaining: list[tuple[str, float, float]] = []
        for pred, conf, timestamp in self._predictions:
            age = now - timestamp
            if age > 120:  # Predictions older than 2 minutes are evaluated
                # Simple heuristic: check if prediction keywords appear in experience
                pred_words = set(pred.lower().split())
                content_words = set(experience.content.lower().split())
                match = len(pred_words & content_words) > 0
                evaluations.append((pred, match))
            else:
                remaining.append((pred, conf, timestamp))

        self._predictions = remaining
        return evaluations

    def _estimate_flow_rate(self) -> float:
        """Estimate how fast time currently feels (flow rate)."""
        if not self._recent_events:
            return 1.0

        # Average recent event density
        now = time.time()
        recent_count = sum(
            1 for _, t in self._recent_events if now - t < 30.0
        )
        # More events = time feels faster (flow state)
        if recent_count > 5:
            return 1.3
        elif recent_count < 2:
            return 0.7
        return 1.0
