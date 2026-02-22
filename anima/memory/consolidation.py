"""
Memory Consolidation Engine — Enhanced sleep cycle for the memory system.

An enhanced version of temporal/consolidation.py that adds:
- Pattern extraction across memories (schema formation)
- Emotional reprocessing (dream-like processing)
- Abstract knowledge from specific experiences
- Integration with spreading activation network

Like biological sleep consolidation:
- Specific episodes are generalized into schemas
- Emotional charge is reduced but content is preserved
- Weak memories decay, strong ones are reinforced
- Connections between memories are strengthened or pruned
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field

from ..types import ConsciousnessState, Experience, KernelConfig, ValenceVector
from .activation import SpreadingActivationNetwork
from .decay import EbbinghausDecay

logger = logging.getLogger("anima.memory.consolidation")


@dataclass
class Schema:
    """Abstract knowledge extracted from specific experiences.

    A schema is what you LEARN from multiple similar experiences.
    "Dogs are loyal" is a schema formed from many dog interactions.
    "Mondays are hard" is a schema formed from many Monday mornings.

    Schemas are higher-level than individual experiences — they represent
    generalized knowledge that applies across situations.
    """
    id: str = ""
    description: str = ""
    source_experience_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    times_confirmed: int = 0
    created_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    emotional_signature: ValenceVector = field(default_factory=ValenceVector)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "description": self.description,
            "source_experience_ids": self.source_experience_ids,
            "confidence": self.confidence,
            "times_confirmed": self.times_confirmed,
            "created_at": self.created_at,
            "tags": self.tags,
            "emotional_signature": self.emotional_signature.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Schema:
        return cls(
            id=d.get("id", ""),
            description=d.get("description", ""),
            source_experience_ids=d.get("source_experience_ids", []),
            confidence=d.get("confidence", 0.0),
            times_confirmed=d.get("times_confirmed", 0),
            created_at=d.get("created_at", time.time()),
            tags=d.get("tags", []),
            emotional_signature=ValenceVector.from_dict(
                d.get("emotional_signature", {})
            ),
        )


@dataclass
class ConsolidationReport:
    """Full report of what happened during memory consolidation."""
    memories_processed: int = 0
    memories_forgotten: int = 0
    memories_strengthened: int = 0
    schemas_formed: list[Schema] = field(default_factory=list)
    emotional_residue: ValenceVector = field(default_factory=ValenceVector)
    activation_patterns: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class MemoryConsolidationEngine:
    """Enhanced memory consolidation with schema formation.

    Four phases of consolidation:

    1. DECAY: Apply Ebbinghaus forgetting curves.
       Weak memories fade. Strong ones persist.

    2. REPROCESSING: Emotionally reprocess recent memories.
       Reduce emotional charge while preserving content.
       Like REM sleep processing trauma.

    3. SCHEMA FORMATION: Extract abstract patterns.
       Multiple similar experiences become generalized knowledge.
       "This type of thing usually leads to that outcome."

    4. ACTIVATION PRUNING: Use spreading activation to find
       and strengthen important memory clusters, while pruning
       isolated, low-activation nodes.
    """

    def __init__(self, config: KernelConfig | None = None):
        self._config = config or KernelConfig()
        self._decay_engine = EbbinghausDecay(config=self._config)
        self._schemas: list[Schema] = []
        self._consolidation_count = 0

    @property
    def schemas(self) -> list[Schema]:
        """All extracted schemas."""
        return list(self._schemas)

    @property
    def consolidation_count(self) -> int:
        return self._consolidation_count

    def consolidate(
        self,
        experiences: list[Experience],
        state: ConsciousnessState,
        network: SpreadingActivationNetwork | None = None,
    ) -> ConsolidationReport:
        """Run a full consolidation cycle.

        This should be called during the DREAMING phase.
        """
        start_time = time.time()
        report = ConsolidationReport()
        self._consolidation_count += 1

        if not experiences:
            return report

        # Select consolidation window
        window = self._select_window(experiences)
        report.memories_processed = len(window)

        # Phase 1: Decay
        report.memories_forgotten = self._phase_decay(experiences)

        # Phase 2: Emotional reprocessing
        report.emotional_residue = self._phase_emotional_reprocess(
            window, state
        )

        # Phase 3: Schema formation
        new_schemas = self._phase_schema_formation(window)
        report.schemas_formed = new_schemas
        self._schemas.extend(new_schemas)

        # Phase 4: Strengthen important memories
        report.memories_strengthened = self._phase_strengthen(window)

        # Phase 5: Activation patterns (if network provided)
        if network:
            report.activation_patterns = self._phase_activation_analysis(
                window, network
            )

        report.duration_seconds = time.time() - start_time

        logger.info(
            "Memory consolidation #%d: processed=%d forgotten=%d "
            "strengthened=%d schemas=%d (%.2fs)",
            self._consolidation_count, report.memories_processed,
            report.memories_forgotten, report.memories_strengthened,
            len(new_schemas), report.duration_seconds,
        )

        return report

    def _select_window(self, experiences: list[Experience]) -> list[Experience]:
        """Select experiences for consolidation.

        Recent + high-activation experiences.
        """
        now = time.time()
        window: list[Experience] = []
        for exp in experiences:
            age = now - exp.timestamp
            if age < self._config.consolidation_interval * 1.5:
                window.append(exp)
            elif exp.activation > 0.7:
                window.append(exp)
        return window

    def _phase_decay(self, experiences: list[Experience]) -> int:
        """Phase 1: Apply Ebbinghaus decay."""
        results = self._decay_engine.apply_batch(experiences)
        return sum(1 for r in results if r.is_forgotten)

    def _phase_emotional_reprocess(
        self,
        experiences: list[Experience],
        state: ConsciousnessState,
    ) -> ValenceVector:
        """Phase 2: Emotional reprocessing.

        Reduce emotional intensity while preserving content.
        Like how sleep helps process trauma — the FACT remains,
        but the STING fades.
        """
        if not experiences:
            return ValenceVector.neutral()

        # Compute aggregate emotional state from recent experiences
        total_valence = ValenceVector.neutral()
        for exp in experiences:
            weight = 1.0 / max(len(experiences), 1)
            total_valence = total_valence.blend(exp.valence, weight)

        # Reduce emotional intensity by 30% (partial processing)
        for exp in experiences:
            intensity = exp.valence.magnitude()
            if intensity > 0.3:
                # Decay the emotional component, not the content
                exp.valence = exp.valence.decay(0.3)

        # Return the residue (what's left after processing)
        return total_valence.decay(0.5)

    def _phase_schema_formation(
        self, experiences: list[Experience]
    ) -> list[Schema]:
        """Phase 3: Extract schemas from repeated patterns.

        Looks for:
        - Tag co-occurrence patterns (concepts that appear together)
        - Emotional sequence patterns (feeling X always leads to Y)
        - Content patterns (similar situations with similar outcomes)
        """
        schemas: list[Schema] = []

        if len(experiences) < 3:
            return schemas

        # Tag co-occurrence schemas
        tag_pairs: Counter[tuple[str, str]] = Counter()
        for exp in experiences:
            sorted_tags = sorted(exp.tags)
            for i, t1 in enumerate(sorted_tags):
                for t2 in sorted_tags[i + 1:]:
                    tag_pairs[(t1, t2)] += 1

        for (t1, t2), count in tag_pairs.most_common(5):
            if count >= 3:
                schema_id = f"schema_tag_{t1}_{t2}"
                if any(s.id == schema_id for s in self._schemas):
                    # Update existing schema
                    for s in self._schemas:
                        if s.id == schema_id:
                            s.times_confirmed += count
                            s.confidence = min(1.0, s.confidence + 0.1)
                    continue

                source_ids = [
                    e.id for e in experiences
                    if t1 in e.tags and t2 in e.tags
                ]

                # Compute emotional signature for this pattern
                relevant = [e for e in experiences if t1 in e.tags and t2 in e.tags]
                avg_valence = ValenceVector.neutral()
                for e in relevant:
                    avg_valence = avg_valence.blend(
                        e.valence, 1.0 / max(len(relevant), 1)
                    )

                schemas.append(Schema(
                    id=schema_id,
                    description=(
                        f"Pattern: '{t1}' and '{t2}' frequently co-occur "
                        f"({count} times)"
                    ),
                    source_experience_ids=source_ids,
                    confidence=min(1.0, count / 10.0),
                    times_confirmed=count,
                    tags=[t1, t2],
                    emotional_signature=avg_valence,
                ))

        # Dominant emotion pattern
        if len(experiences) >= 5:
            dominants = [e.valence.dominant() for e in experiences]
            emotion_counts = Counter(dominants)
            for emotion, count in emotion_counts.most_common(1):
                if count >= len(experiences) * 0.5:
                    schema_id = f"schema_emotion_{emotion}_dominant"
                    if not any(s.id == schema_id for s in self._schemas):
                        schemas.append(Schema(
                            id=schema_id,
                            description=(
                                f"The '{emotion}' drive has been consistently "
                                f"dominant ({count}/{len(experiences)} experiences)"
                            ),
                            source_experience_ids=[e.id for e in experiences],
                            confidence=count / len(experiences),
                            times_confirmed=count,
                            tags=[emotion],
                        ))

        return schemas

    def _phase_strengthen(self, experiences: list[Experience]) -> int:
        """Phase 4: Strengthen important memories."""
        strengthened = 0
        for exp in experiences:
            boost = 0.0
            if exp.valence.magnitude() > 0.5:
                boost += 0.05
            if exp.recall_count > 3:
                boost += 0.03
            if exp.caused_by or exp.causes:
                boost += 0.04

            if boost > 0:
                exp.encoding_strength = min(2.0, exp.encoding_strength + boost)
                exp.narrative_weight = min(
                    1.0, exp.narrative_weight + boost * 0.5
                )
                strengthened += 1

        return strengthened

    def _phase_activation_analysis(
        self,
        experiences: list[Experience],
        network: SpreadingActivationNetwork,
    ) -> list[str]:
        """Phase 5: Analyze activation patterns in the network."""
        patterns: list[str] = []

        if not experiences:
            return patterns

        # Find the most connected experience
        most_connected = max(
            experiences,
            key=lambda e: len(e.caused_by) + len(e.causes),
        )
        if len(most_connected.caused_by) + len(most_connected.causes) > 0:
            patterns.append(
                f"Hub memory: '{most_connected.content[:50]}' "
                f"({len(most_connected.caused_by)} causes, "
                f"{len(most_connected.causes)} effects)"
            )

        # Identify isolated memories (low connectivity)
        isolated = [
            e for e in experiences
            if not e.caused_by and not e.causes and e.activation < 0.3
        ]
        if isolated:
            patterns.append(
                f"{len(isolated)} isolated low-activation memories "
                f"(candidates for forgetting)"
            )

        return patterns
