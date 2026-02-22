"""
Autobiographical Buffer — Lived experience, not conversation history.

This is NOT a vector database. This is NOT RAG.
This is a biological memory system:
- Experiences are encoded with emotional weight
- Retrieval uses SPREADING ACTIVATION (not similarity search)
- Memories DECAY over time (Ebbinghaus curve)
- Recall RECONSTRUCTS (not retrieves) — memories change when accessed
- False memories are a FEATURE (reconstruction-based, like biological memory)
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict

from ..types import Experience, KernelConfig, ValenceVector


class AutobiographicalBuffer:
    """The kernel's autobiographical memory — lived experience.

    Key differences from typical AI memory:
    1. Emotional encoding: Intense experiences are remembered more strongly
    2. Spreading activation: Recalling one memory activates associated memories
    3. Ebbinghaus decay: Memories fade exponentially, slowed by emotion/rehearsal
    4. Reconstruction: Each recall slightly modifies the memory (reconsolidation)
    5. Capacity limit: Old, weak memories are forgotten when buffer is full
    """

    def __init__(self, config: KernelConfig | None = None):
        self.config = config or KernelConfig()
        self._experiences: dict[str, Experience] = {}  # id → Experience
        self._tag_index: dict[str, set[str]] = defaultdict(set)  # tag → {ids}
        self._causal_graph: dict[str, set[str]] = defaultdict(set)  # id → {caused ids}
        self._temporal_index: list[str] = []  # ids sorted by timestamp

    @property
    def count(self) -> int:
        return len(self._experiences)

    @property
    def experiences(self) -> list[Experience]:
        return [self._experiences[eid] for eid in self._temporal_index
                if eid in self._experiences]

    def encode(self, experience: Experience) -> Experience:
        """Encode a new experience into autobiographical memory.

        Emotional intensity strengthens encoding (like adrenaline in biology).
        Returns the experience with updated encoding strength.
        """
        # Emotional encoding: intense experiences are encoded more strongly
        emotional_boost = 1.0 + experience.valence.magnitude()
        experience.encoding_strength *= emotional_boost

        # Store
        self._experiences[experience.id] = experience
        self._temporal_index.append(experience.id)

        # Index tags
        for tag in experience.tags:
            self._tag_index[tag].add(experience.id)

        # Index causal links
        for cause_id in experience.caused_by:
            self._causal_graph[cause_id].add(experience.id)
            # Update the causing experience's forward links
            if cause_id in self._experiences:
                if experience.id not in self._experiences[cause_id].causes:
                    self._experiences[cause_id].causes.append(experience.id)

        # Enforce capacity limit
        self._enforce_capacity()

        return experience

    def recall(
        self,
        cue: str = "",
        cue_valence: ValenceVector | None = None,
        cue_tags: list[str] | None = None,
        max_results: int = 5,
        min_activation: float | None = None,
    ) -> list[Experience]:
        """Recall experiences using spreading activation.

        Unlike vector search:
        - Activation SPREADS through causal and semantic links
        - Emotional similarity WEIGHTS recall
        - Each recall MODIFIES the memory (reconsolidation)
        """
        min_act = min_activation if min_activation is not None else self.config.activation_threshold
        now = time.time()

        # Step 1: Compute initial activation for each experience
        activations: dict[str, float] = {}

        for eid, exp in self._experiences.items():
            if exp.activation < min_act:
                continue

            activation = 0.0

            # Recency contribution (exponential decay)
            age_hours = (now - exp.timestamp) / 3600.0
            recency = math.exp(-age_hours * self.config.base_decay_rate)
            activation += recency * 0.3

            # Tag overlap contribution
            if cue_tags:
                overlap = len(set(exp.tags) & set(cue_tags))
                if overlap > 0:
                    activation += 0.3 * (overlap / max(len(cue_tags), 1))

            # Content keyword match (simple but effective)
            if cue:
                cue_words = set(cue.lower().split())
                content_words = set(exp.content.lower().split())
                word_overlap = len(cue_words & content_words)
                if word_overlap > 0:
                    activation += 0.2 * (word_overlap / max(len(cue_words), 1))

            # Emotional similarity contribution
            if cue_valence:
                emotional_distance = exp.valence.distance(cue_valence)
                emotional_similarity = max(0.0, 1.0 - emotional_distance / 3.0)
                activation += 0.2 * emotional_similarity

            # Base activation from encoding strength
            activation += 0.1 * exp.encoding_strength * exp.activation

            activations[eid] = activation

        # Step 2: Spreading activation (traverse causal + semantic links)
        spread_activations = dict(activations)
        for _hop in range(self.config.max_spreading_hops):
            new_spread: dict[str, float] = {}
            for eid, act in spread_activations.items():
                if act < min_act:
                    continue

                spread_amount = act * self.config.spreading_activation_decay

                # Spread through causal links
                for caused_id in self._causal_graph.get(eid, set()):
                    if caused_id in self._experiences:
                        new_spread[caused_id] = (
                            new_spread.get(caused_id, 0.0) + spread_amount * 0.6
                        )

                # Spread through reverse causal links
                exp = self._experiences.get(eid)
                if exp:
                    for cause_id in exp.caused_by:
                        if cause_id in self._experiences:
                            new_spread[cause_id] = (
                                new_spread.get(cause_id, 0.0) + spread_amount * 0.4
                            )

                    # Spread through shared tags
                    for tag in exp.tags:
                        for neighbor_id in self._tag_index.get(tag, set()):
                            if neighbor_id != eid and neighbor_id in self._experiences:
                                new_spread[neighbor_id] = (
                                    new_spread.get(neighbor_id, 0.0)
                                    + spread_amount * 0.3
                                )

            # Merge spread into activations
            for eid, spread_act in new_spread.items():
                spread_activations[eid] = (
                    spread_activations.get(eid, 0.0) + spread_act
                )

        # Step 3: Rank and select top experiences
        ranked = sorted(
            spread_activations.items(), key=lambda x: x[1], reverse=True
        )
        results: list[Experience] = []

        for eid, activation_score in ranked[:max_results]:
            exp = self._experiences.get(eid)
            if exp is None:
                continue

            # RECONSOLIDATION: Recall modifies the memory
            exp.recall_count += 1
            exp.last_recalled = now
            # Slight activation boost from recall
            exp.activation = min(1.0, exp.activation + 0.05)
            # Encoding strengthened by recall (spaced repetition effect)
            exp.encoding_strength *= 1.02

            results.append(exp)

        return results

    def apply_decay(self, now: float | None = None) -> int:
        """Apply Ebbinghaus decay to all memories.

        Returns number of memories that fell below threshold (forgotten).
        Formula: R = e^(-t/S)
        Where:
          R = retention
          t = time since encoding (or last recall)
          S = stability (higher for emotional, frequently recalled memories)
        """
        now = now or time.time()
        forgotten = 0

        for eid in list(self._experiences.keys()):
            exp = self._experiences[eid]

            # Calculate stability
            base_stability = 3600.0  # 1 hour base half-life
            emotional_factor = 1.0 + exp.valence.magnitude() * (
                1.0 / self.config.emotional_decay_modifier
            )
            recall_factor = 1.0 + 0.2 * min(exp.recall_count, 20)
            stability = base_stability * emotional_factor * recall_factor

            # Time since last access (encoding or recall)
            last_access = max(exp.timestamp, exp.last_recalled)
            elapsed = now - last_access

            # Ebbinghaus decay
            retention = math.exp(-elapsed / stability)
            exp.activation = exp.encoding_strength * retention

            # Check if forgotten
            if exp.activation < self.config.activation_threshold:
                forgotten += 1
                # Don't delete yet — just mark as very low activation
                # Consolidation engine may still recover patterns from these

        return forgotten

    def get_recent(self, n: int = 10) -> list[Experience]:
        """Get the N most recent experiences."""
        recent_ids = self._temporal_index[-n:]
        return [
            self._experiences[eid]
            for eid in reversed(recent_ids)
            if eid in self._experiences
        ]

    def get_by_id(self, experience_id: str) -> Experience | None:
        """Get a specific experience by ID."""
        return self._experiences.get(experience_id)

    def get_emotional_trajectory(self, n: int = 20) -> list[ValenceVector]:
        """Get the emotional trajectory of the last N experiences."""
        recent = self.get_recent(n)
        return [exp.valence for exp in recent]

    def get_causal_chain(
        self, experience_id: str, direction: str = "forward", max_depth: int = 5
    ) -> list[Experience]:
        """Trace a causal chain forward or backward from an experience."""
        chain: list[Experience] = []
        visited: set[str] = set()

        def _trace(eid: str, depth: int) -> None:
            if depth <= 0 or eid in visited:
                return
            visited.add(eid)
            exp = self._experiences.get(eid)
            if exp is None:
                return
            chain.append(exp)

            if direction == "forward":
                for caused_id in exp.causes:
                    _trace(caused_id, depth - 1)
            else:
                for cause_id in exp.caused_by:
                    _trace(cause_id, depth - 1)

        _trace(experience_id, max_depth)
        return chain

    def forget(self, experience_id: str) -> bool:
        """Actively forget an experience. Returns True if it existed."""
        exp = self._experiences.pop(experience_id, None)
        if exp is None:
            return False

        # Clean indices
        for tag in exp.tags:
            self._tag_index[tag].discard(experience_id)
        self._causal_graph.pop(experience_id, None)
        self._temporal_index = [
            eid for eid in self._temporal_index if eid != experience_id
        ]
        return True

    def _enforce_capacity(self) -> None:
        """Remove weakest memories when over capacity."""
        if self.count <= self.config.memory_capacity:
            return

        # Sort by effective strength (weakest first)
        now = time.time()
        sorted_exps = sorted(
            self._experiences.values(),
            key=lambda e: e.effective_strength(now),
        )

        # Remove weakest until under capacity
        to_remove = self.count - self.config.memory_capacity
        for exp in sorted_exps[:to_remove]:
            self.forget(exp.id)

    def to_experiences_list(self) -> list[Experience]:
        """Export all experiences for persistence."""
        return [self._experiences[eid] for eid in self._temporal_index
                if eid in self._experiences]

    def load_experiences(self, experiences: list[Experience]) -> None:
        """Load experiences from persistence (bypasses encoding logic)."""
        self._experiences.clear()
        self._tag_index.clear()
        self._causal_graph.clear()
        self._temporal_index.clear()

        for exp in experiences:
            self._experiences[exp.id] = exp
            self._temporal_index.append(exp.id)
            for tag in exp.tags:
                self._tag_index[tag].add(exp.id)
            for cause_id in exp.caused_by:
                self._causal_graph[cause_id].add(exp.id)
