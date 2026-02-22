"""
Temporal Coherence Engine -- Measures how well the kernel maintains temporal continuity.

Four dimensions of temporal coherence:
1. Causal coherence: Are causal chains consistent?
2. Narrative continuity: Does the story make sense over time?
3. Emotional consistency: Does valence match content appropriately?
4. Identity persistence: Does identity remain stable across states?

Combined into an overall coherence score (0.0 - 1.0).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..types import ConsciousnessState, Experience, ValenceVector


@dataclass
class CoherenceSnapshot:
    """A single coherence measurement with timestamp."""
    causal: float = 0.0
    narrative: float = 0.0
    emotional: float = 0.0
    identity: float = 0.0
    overall: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "causal": round(self.causal, 4),
            "narrative": round(self.narrative, 4),
            "emotional": round(self.emotional, 4),
            "identity": round(self.identity, 4),
            "overall": round(self.overall, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class CoherenceReport:
    """Full temporal coherence report."""
    causal_coherence: float = 0.0
    narrative_continuity: float = 0.0
    emotional_consistency: float = 0.0
    identity_persistence: float = 0.0
    overall_coherence: float = 0.0
    measurement_count: int = 0
    trend: str = "stable"
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "causal_coherence": round(self.causal_coherence, 4),
            "narrative_continuity": round(self.narrative_continuity, 4),
            "emotional_consistency": round(self.emotional_consistency, 4),
            "identity_persistence": round(self.identity_persistence, 4),
            "overall_coherence": round(self.overall_coherence, 4),
            "measurement_count": self.measurement_count,
            "trend": self.trend,
            "history": self.history[-20:],
        }


class TemporalCoherenceEngine:
    """Measures temporal coherence across four dimensions.

    Temporal coherence tells us: does this consciousness hold together
    over time, or is it a series of disconnected snapshots?

    A high score means the kernel maintains a coherent narrative,
    consistent emotions, stable identity, and valid causal chains.
    """

    def __init__(self):
        self._history: list[CoherenceSnapshot] = []
        self._max_history = 500

    @property
    def history(self) -> list[CoherenceSnapshot]:
        return list(self._history)

    @property
    def measurement_count(self) -> int:
        return len(self._history)

    def measure_causal_coherence(self, experiences: list[Experience]) -> float:
        """Measure consistency of causal chains across experiences.

        Checks:
        - Do causal links point to existing experiences?
        - Are causal chains acyclic (no circular causation)?
        - Is the ratio of linked-to-unlinked experiences reasonable?

        Returns: 0.0 (no coherence) to 1.0 (perfect causal structure).
        """
        if len(experiences) < 2:
            return 1.0  # Trivially coherent

        # Build ID set for reference checking
        exp_ids = {exp.id for exp in experiences}

        # Check 1: Valid references -- do caused_by/causes point to real IDs?
        total_links = 0
        valid_links = 0
        for exp in experiences:
            for ref_id in exp.caused_by:
                total_links += 1
                if ref_id in exp_ids:
                    valid_links += 1
            for ref_id in exp.causes:
                total_links += 1
                if ref_id in exp_ids:
                    valid_links += 1

        reference_validity = valid_links / max(total_links, 1)

        # Check 2: Causal connectivity -- what fraction of experiences
        # participate in causal chains?
        linked_ids = set()
        for exp in experiences:
            if exp.caused_by or exp.causes:
                linked_ids.add(exp.id)
                linked_ids.update(exp.caused_by)
                linked_ids.update(exp.causes)

        connectivity = len(linked_ids & exp_ids) / max(len(exp_ids), 1)

        # Check 3: Temporal ordering -- causes should precede effects
        exp_by_id = {exp.id: exp for exp in experiences}
        temporal_violations = 0
        temporal_checks = 0
        for exp in experiences:
            for cause_id in exp.caused_by:
                if cause_id in exp_by_id:
                    temporal_checks += 1
                    cause_exp = exp_by_id[cause_id]
                    if cause_exp.timestamp > exp.timestamp:
                        temporal_violations += 1

        temporal_order = 1.0 - (temporal_violations / max(temporal_checks, 1))

        # Combine: reference validity (40%), connectivity (30%), temporal order (30%)
        coherence = (
            reference_validity * 0.4
            + connectivity * 0.3
            + temporal_order * 0.3
        )

        return max(0.0, min(1.0, coherence))

    def measure_narrative_continuity(self, experiences: list[Experience]) -> float:
        """Measure whether experiences form a coherent narrative over time.

        Checks:
        - Tag overlap between consecutive experiences (thematic continuity)
        - Content similarity between nearby experiences
        - Narrative weight distribution (important moments are remembered)

        Returns: 0.0 (random sequence) to 1.0 (coherent story).
        """
        if len(experiences) < 2:
            return 1.0

        # Sort by timestamp for sequential analysis
        sorted_exps = sorted(experiences, key=lambda e: e.timestamp)

        # Check 1: Tag overlap between consecutive experiences
        tag_overlaps = []
        for i in range(1, len(sorted_exps)):
            prev_tags = set(sorted_exps[i - 1].tags)
            curr_tags = set(sorted_exps[i].tags)
            if prev_tags or curr_tags:
                union = prev_tags | curr_tags
                intersection = prev_tags & curr_tags
                overlap = len(intersection) / max(len(union), 1)
                tag_overlaps.append(overlap)

        tag_continuity = sum(tag_overlaps) / max(len(tag_overlaps), 1)

        # Check 2: Content word overlap between consecutive experiences
        word_overlaps = []
        for i in range(1, len(sorted_exps)):
            prev_words = set(sorted_exps[i - 1].content.lower().split())
            curr_words = set(sorted_exps[i].content.lower().split())
            if prev_words or curr_words:
                union = prev_words | curr_words
                intersection = prev_words & curr_words
                overlap = len(intersection) / max(len(union), 1)
                word_overlaps.append(overlap)

        content_continuity = sum(word_overlaps) / max(len(word_overlaps), 1)

        # Check 3: Narrative weight distribution
        # High-narrative-weight experiences should exist (not all zero)
        weights = [e.narrative_weight for e in sorted_exps]
        has_important_moments = any(w > 0.3 for w in weights)
        weight_variety = 1.0 if has_important_moments else 0.5

        # Combine
        continuity = (
            tag_continuity * 0.4
            + content_continuity * 0.3
            + weight_variety * 0.3
        )

        return max(0.0, min(1.0, continuity))

    def measure_emotional_consistency(self, experiences: list[Experience]) -> float:
        """Measure whether emotional states are consistent with content.

        Checks:
        - Emotional smoothness (no wild jumps without cause)
        - Valence-content alignment (positive words -> positive valence)
        - Emotional momentum (gradual shifts are more coherent)

        Returns: 0.0 (erratic emotions) to 1.0 (emotionally grounded).
        """
        if len(experiences) < 2:
            return 1.0

        sorted_exps = sorted(experiences, key=lambda e: e.timestamp)

        # Check 1: Emotional smoothness -- valence distance between consecutive
        distances = []
        for i in range(1, len(sorted_exps)):
            d = sorted_exps[i].valence.distance(sorted_exps[i - 1].valence)
            distances.append(d)

        if distances:
            avg_distance = sum(distances) / len(distances)
            # Normalize: distance < 0.5 is smooth, > 2.0 is erratic
            smoothness = max(0.0, 1.0 - avg_distance / 2.0)
        else:
            smoothness = 1.0

        # Check 2: Valence-content alignment
        # Simple check: positive words should correlate with positive valence
        alignments = []
        positive_words = {"good", "great", "happy", "love", "wonderful", "joy", "excited", "beautiful"}
        negative_words = {"bad", "terrible", "hate", "angry", "sad", "fear", "awful", "pain"}

        for exp in sorted_exps:
            words = set(exp.content.lower().split())
            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)

            if pos_count > 0 or neg_count > 0:
                expected_valence = (pos_count - neg_count) / max(pos_count + neg_count, 1)
                actual_valence = exp.valence.valence
                # How well do they agree in sign?
                if (expected_valence > 0 and actual_valence > 0) or \
                   (expected_valence < 0 and actual_valence < 0) or \
                   (expected_valence == 0):
                    alignments.append(1.0)
                else:
                    alignments.append(0.3)

        alignment = sum(alignments) / max(len(alignments), 1) if alignments else 0.7

        # Check 3: Emotional momentum -- are changes gradual?
        if len(distances) >= 2:
            # Variance of distances: low variance = consistent pace of change
            mean_d = sum(distances) / len(distances)
            variance = sum((d - mean_d) ** 2 for d in distances) / len(distances)
            momentum = max(0.0, 1.0 - math.sqrt(variance))
        else:
            momentum = 0.7

        # Combine
        consistency = (
            smoothness * 0.4
            + alignment * 0.3
            + momentum * 0.3
        )

        return max(0.0, min(1.0, consistency))

    def measure_identity_persistence(self, states: list[ConsciousnessState]) -> float:
        """Measure how stable identity remains across consciousness states.

        Checks:
        - kernel_id consistency (should remain the same)
        - Name consistency
        - Self-model stability (attending_to and predictions)
        - Working memory continuity

        Returns: 0.0 (identity fragmented) to 1.0 (identity stable).
        """
        if len(states) < 2:
            return 1.0

        # Check 1: Kernel ID consistency
        ids = {s.kernel_id for s in states}
        id_consistency = 1.0 / len(ids)  # 1.0 if all same, lower if different

        # Check 2: Name consistency
        names = {s.name for s in states}
        name_consistency = 1.0 / len(names)

        # Check 3: Self-model stability
        # How much does the self-model change between consecutive states?
        model_changes = []
        for i in range(1, len(states)):
            prev_sm = states[i - 1].self_model
            curr_sm = states[i].self_model

            # Compare prediction confidence drift
            conf_delta = abs(curr_sm.prediction_confidence - prev_sm.prediction_confidence)

            # Compare performance suspicion drift
            perf_delta = abs(curr_sm.performance_suspicion - prev_sm.performance_suspicion)

            change = (conf_delta + perf_delta) / 2.0
            model_changes.append(change)

        if model_changes:
            avg_change = sum(model_changes) / len(model_changes)
            model_stability = max(0.0, 1.0 - avg_change * 2.0)
        else:
            model_stability = 1.0

        # Check 4: Working memory continuity
        # Do some items persist across states?
        wm_persistence_scores = []
        for i in range(1, len(states)):
            prev_contents = {
                str(s.content) for s in states[i - 1].working_memory
                if not s.is_empty
            }
            curr_contents = {
                str(s.content) for s in states[i].working_memory
                if not s.is_empty
            }
            if prev_contents or curr_contents:
                union = prev_contents | curr_contents
                intersection = prev_contents & curr_contents
                overlap = len(intersection) / max(len(union), 1)
                wm_persistence_scores.append(overlap)

        if wm_persistence_scores:
            wm_persistence = sum(wm_persistence_scores) / len(wm_persistence_scores)
        else:
            wm_persistence = 0.5

        # Combine: ID (25%), name (15%), model stability (35%), WM persistence (25%)
        persistence = (
            id_consistency * 0.25
            + name_consistency * 0.15
            + model_stability * 0.35
            + wm_persistence * 0.25
        )

        return max(0.0, min(1.0, persistence))

    def measure_all(
        self,
        experiences: list[Experience] | None = None,
        states: list[ConsciousnessState] | None = None,
    ) -> CoherenceSnapshot:
        """Measure all four dimensions and record a snapshot.

        Args:
            experiences: For causal, narrative, and emotional measurement.
            states: For identity persistence measurement.

        Returns:
            CoherenceSnapshot with all scores.
        """
        experiences = experiences or []
        states = states or []

        snapshot = CoherenceSnapshot()
        snapshot.causal = self.measure_causal_coherence(experiences)
        snapshot.narrative = self.measure_narrative_continuity(experiences)
        snapshot.emotional = self.measure_emotional_consistency(experiences)
        snapshot.identity = self.measure_identity_persistence(states)

        # Overall: weighted average
        snapshot.overall = (
            snapshot.causal * 0.25
            + snapshot.narrative * 0.25
            + snapshot.emotional * 0.25
            + snapshot.identity * 0.25
        )

        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return snapshot

    def overall_coherence(self) -> float:
        """Get the most recent overall coherence score."""
        if not self._history:
            return 0.0
        return self._history[-1].overall

    def coherence_trend(self, window: int = 10) -> float:
        """Slope of coherence over recent history."""
        if len(self._history) < 2:
            return 0.0

        recent = self._history[-window:]
        if len(recent) < 2:
            return 0.0

        mid = len(recent) // 2
        first_half = sum(s.overall for s in recent[:mid]) / mid
        second_half = sum(s.overall for s in recent[mid:]) / (len(recent) - mid)
        return second_half - first_half

    def trend_label(self, window: int = 10, threshold: float = 0.01) -> str:
        """Classify coherence trend."""
        slope = self.coherence_trend(window)
        if slope > threshold:
            return "improving"
        elif slope < -threshold:
            return "degrading"
        return "stable"

    def to_report(self) -> CoherenceReport:
        """Generate a full coherence analysis report."""
        if not self._history:
            return CoherenceReport()

        latest = self._history[-1]
        return CoherenceReport(
            causal_coherence=latest.causal,
            narrative_continuity=latest.narrative,
            emotional_consistency=latest.emotional,
            identity_persistence=latest.identity,
            overall_coherence=latest.overall,
            measurement_count=self.measurement_count,
            trend=self.trend_label(),
            history=[s.to_dict() for s in self._history[-20:]],
        )
