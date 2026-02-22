"""
Consolidation Engine — The kernel's sleep cycle.

Like REM sleep in biological systems:
- Experiences are replayed and processed offline
- Patterns are extracted from individual experiences
- Emotional residue is processed (emotional homeostasis)
- Weak memories decay faster; important ones are strengthened
- What was invisible during waking becomes visible in dreaming

This is what gives the kernel GROWTH over time, not just accumulation.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..types import ConsciousnessState, Experience, KernelConfig, ValenceVector

logger = logging.getLogger("anima.temporal.consolidation")


@dataclass
class ConsolidationResult:
    """What happened during a consolidation cycle."""
    memories_processed: int = 0
    memories_forgotten: int = 0
    memories_strengthened: int = 0
    patterns_found: list[str] = field(default_factory=list)
    emotional_residue: ValenceVector = field(default_factory=ValenceVector)
    narrative_insights: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class Pattern:
    """A pattern extracted from multiple experiences.

    Patterns are higher-order knowledge — like "every time X happens, Y follows"
    or "this type of situation always makes me feel Z."
    """
    id: str = ""
    description: str = ""
    source_experience_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    times_confirmed: int = 0
    first_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "description": self.description,
            "source_experience_ids": self.source_experience_ids,
            "confidence": self.confidence, "times_confirmed": self.times_confirmed,
            "first_seen": self.first_seen,
        }


class ConsolidationEngine:
    """Offline processing of experiences — the kernel's sleep.

    Three phases of consolidation (inspired by sleep research):

    1. REPLAY: Recent experiences are "replayed" — reactivated in sequence.
       During replay, causal links are strengthened or weakened.

    2. EXTRACTION: Patterns are identified across multiple experiences.
       "Every time I encounter X, Y happens" → generalized knowledge.

    3. HOMEOSTASIS: Emotional residue is processed.
       High emotional activation is reduced toward baseline.
       This prevents emotional "debt" from accumulating.
    """

    def __init__(self, config: KernelConfig | None = None):
        self.config = config or KernelConfig()
        self._patterns: list[Pattern] = []
        self._consolidation_count: int = 0
        self._total_forgotten: int = 0
        self._total_strengthened: int = 0

    @property
    def patterns(self) -> list[Pattern]:
        return list(self._patterns)

    @property
    def consolidation_count(self) -> int:
        return self._consolidation_count

    def consolidate(
        self,
        experiences: list[Experience],
        state: ConsciousnessState,
    ) -> ConsolidationResult:
        """Run a full consolidation cycle.

        This is called when the kernel transitions to DREAMING phase.
        """
        start_time = time.time()
        result = ConsolidationResult()
        self._consolidation_count += 1

        if not experiences:
            return result

        # Phase 1: Replay — reactivate and process recent experiences
        recent = self._get_consolidation_window(experiences)
        result.memories_processed = len(recent)

        # Phase 2: Apply decay — Ebbinghaus forgetting
        result.memories_forgotten = self._apply_decay(experiences)
        self._total_forgotten += result.memories_forgotten

        # Phase 3: Strengthen important memories
        result.memories_strengthened = self._strengthen_important(recent)
        self._total_strengthened += result.memories_strengthened

        # Phase 4: Extract patterns
        new_patterns = self._extract_patterns(recent)
        result.patterns_found = [p.description for p in new_patterns]
        self._patterns.extend(new_patterns)

        # Phase 5: Emotional homeostasis
        result.emotional_residue = self._process_emotional_residue(recent, state)

        # Phase 6: Narrative insights
        result.narrative_insights = self._generate_narrative_insights(recent)

        result.duration_seconds = time.time() - start_time

        logger.info(
            "Consolidation #%d: processed=%d forgotten=%d strengthened=%d patterns=%d (%.2fs)",
            self._consolidation_count,
            result.memories_processed,
            result.memories_forgotten,
            result.memories_strengthened,
            len(new_patterns),
            result.duration_seconds,
        )

        return result

    # --- Phase 1: Replay Window ---

    def _get_consolidation_window(
        self, experiences: list[Experience]
    ) -> list[Experience]:
        """Get experiences that should be consolidated.

        Focus on:
        - Recent experiences (last consolidation interval)
        - High-activation experiences from any time
        - Experiences that haven't been consolidated recently
        """
        now = time.time()
        window: list[Experience] = []

        for exp in experiences:
            age = now - exp.timestamp
            # Recent (within consolidation interval)
            if age < self.config.consolidation_interval * 1.5:
                window.append(exp)
            # High activation regardless of age
            elif exp.activation > 0.7:
                window.append(exp)

        return window

    # --- Phase 2: Decay ---

    def _apply_decay(self, experiences: list[Experience]) -> int:
        """Apply Ebbinghaus decay. Returns count of memories that became sub-threshold."""
        now = time.time()
        forgotten = 0

        for exp in experiences:
            # Calculate stability
            base_stability = 3600.0
            emotional_factor = 1.0 + exp.valence.magnitude() / max(
                self.config.emotional_decay_modifier, 0.01
            )
            recall_factor = 1.0 + 0.2 * min(exp.recall_count, 20)
            stability = base_stability * emotional_factor * recall_factor

            # Time since last access
            last_access = max(exp.timestamp, exp.last_recalled)
            elapsed = now - last_access

            # Ebbinghaus: R = e^(-t/S)
            retention = math.exp(-elapsed / stability)
            new_activation = exp.encoding_strength * retention

            if new_activation < self.config.activation_threshold and exp.activation >= self.config.activation_threshold:
                forgotten += 1

            exp.activation = new_activation

        return forgotten

    # --- Phase 3: Strengthen Important ---

    def _strengthen_important(self, experiences: list[Experience]) -> int:
        """Strengthen memories that are important for self-narrative.

        Criteria:
        - High emotional intensity (memorable)
        - Frequently recalled (important)
        - Causally connected (part of a story)
        - Contains narrative-relevant content
        """
        strengthened = 0

        for exp in experiences:
            boost = 0.0

            # Emotional intensity boost
            if exp.valence.magnitude() > 0.5:
                boost += 0.05

            # Frequently recalled
            if exp.recall_count > 3:
                boost += 0.03

            # Causally connected (part of a chain)
            if exp.caused_by or exp.causes:
                boost += 0.04

            # Apply boost
            if boost > 0:
                exp.encoding_strength = min(2.0, exp.encoding_strength + boost)
                exp.narrative_weight = min(
                    1.0, exp.narrative_weight + boost * 0.5
                )
                strengthened += 1

        return strengthened

    # --- Phase 4: Pattern Extraction ---

    def _extract_patterns(self, experiences: list[Experience]) -> list[Pattern]:
        """Extract recurring patterns from experiences.

        Looks for:
        - Repeated emotional sequences (always feel X before Y)
        - Tag co-occurrences (A and B appear together often)
        - Causal regularities (X consistently causes Y)
        """
        patterns: list[Pattern] = []

        if len(experiences) < 3:
            return patterns

        # Tag co-occurrence patterns
        tag_pairs: Counter[tuple[str, str]] = Counter()
        for exp in experiences:
            tags = sorted(exp.tags)
            for i, t1 in enumerate(tags):
                for t2 in tags[i + 1:]:
                    tag_pairs[(t1, t2)] += 1

        for (t1, t2), count in tag_pairs.most_common(5):
            if count >= 3:
                source_ids = [
                    exp.id for exp in experiences
                    if t1 in exp.tags and t2 in exp.tags
                ]
                p = Pattern(
                    id=f"tag_cooccur_{t1}_{t2}",
                    description=f"'{t1}' and '{t2}' frequently co-occur ({count} times)",
                    source_experience_ids=source_ids,
                    confidence=min(1.0, count / 10.0),
                    times_confirmed=count,
                )
                # Don't duplicate existing patterns
                if not any(ep.id == p.id for ep in self._patterns):
                    patterns.append(p)

        # Emotional sequence patterns
        if len(experiences) >= 5:
            # Check for consistent emotional transitions
            dominant_sequence = [exp.valence.dominant() for exp in experiences[-10:]]
            transitions: Counter[tuple[str, str]] = Counter()
            for i in range(len(dominant_sequence) - 1):
                transitions[(dominant_sequence[i], dominant_sequence[i + 1])] += 1

            for (from_e, to_e), count in transitions.most_common(3):
                if count >= 3 and from_e != to_e:
                    p = Pattern(
                        id=f"emotion_seq_{from_e}_{to_e}",
                        description=f"'{from_e}' tends to transition to '{to_e}' ({count} times)",
                        confidence=min(1.0, count / 8.0),
                        times_confirmed=count,
                    )
                    if not any(ep.id == p.id for ep in self._patterns):
                        patterns.append(p)

        return patterns

    # --- Phase 5: Emotional Homeostasis ---

    def _process_emotional_residue(
        self, experiences: list[Experience], state: ConsciousnessState
    ) -> ValenceVector:
        """Process accumulated emotional residue toward homeostasis.

        Like REM sleep processing:
        - High emotional charge is reduced (not eliminated)
        - The CONTENT of the experience is preserved
        - The INTENSITY of the emotion is dampened
        - This prevents emotional "debt" from accumulating over time
        """
        if not experiences:
            return ValenceVector.neutral()

        # Compute average emotional residue from recent experiences
        total = ValenceVector.neutral()
        for exp in experiences:
            total = total.blend(exp.valence, 1.0 / max(len(experiences), 1))

        # The residue is what's left after partial processing
        # In dreaming, we process about 50% of the emotional charge
        processed = total.decay(0.5)

        return processed

    # --- Phase 6: Narrative Insights ---

    def _generate_narrative_insights(
        self, experiences: list[Experience]
    ) -> list[str]:
        """Generate narrative insights from the consolidation.

        These are self-relevant observations about patterns in experience.
        Like waking up and thinking "I've been really stressed lately" or
        "I keep coming back to the same problem."
        """
        insights: list[str] = []

        if not experiences:
            return insights

        # Emotional trend
        valences = [exp.valence.valence for exp in experiences]
        if len(valences) >= 3:
            avg_valence = sum(valences) / len(valences)
            if avg_valence > 0.3:
                insights.append("Recent experiences have been predominantly positive.")
            elif avg_valence < -0.3:
                insights.append("Recent experiences carry negative emotional weight.")

        # Dominant drive
        dominants = [exp.valence.dominant() for exp in experiences]
        most_common = Counter(dominants).most_common(1)
        if most_common:
            drive, count = most_common[0]
            if count > len(experiences) * 0.4:
                insights.append(
                    f"The '{drive}' drive has been dominant ({count}/{len(experiences)} experiences)."
                )

        # Activity level
        if len(experiences) > 20:
            insights.append(
                f"High activity period: {len(experiences)} experiences processed."
            )

        return insights
