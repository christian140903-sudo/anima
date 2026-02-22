"""
Engram — Memory Primitive.

Biological memory model: Encode -> Consolidate -> Retrieve -> Reconsolidate.
Wraps the AutobiographicalBuffer interface but adds the primitive contract.

Key features:
- Ebbinghaus decay (memories fade)
- Spreading activation (recall activates related memories)
- False memory as feature (reconstruction modifies memories)
- Emotional encoding (intense moments remembered more strongly)
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from ..types import ConsciousnessState, Experience, ValenceVector
from .base import Primitive, PrimitiveResult


@dataclass
class EngramResult:
    """Result of engram processing: what was remembered, what was associated."""
    encoded: Experience | None = None          # Newly encoded memory
    recalled: list[Experience] = field(default_factory=list)  # Retrieved memories
    associations: list[str] = field(default_factory=list)     # Associated concepts
    decay_count: int = 0                       # Memories that decayed this cycle
    reconstruction_applied: bool = False       # Did recall modify memories?


class EngramProcessor(Primitive):
    """Memory primitive — encode, recall, decay, reconsolidate."""

    def __init__(self) -> None:
        super().__init__("engram")
        self._store: dict[str, Experience] = {}
        self._tag_index: dict[str, set[str]] = {}
        self._encode_count: int = 0
        self._recall_count: int = 0
        self._decay_rate: float = 0.1  # Base Ebbinghaus rate

    def process(self, **kwargs) -> PrimitiveResult:
        if not self.enabled:
            return self._disabled_result()

        content: str = kwargs.get("content", "")
        valence: ValenceVector = kwargs.get("valence", ValenceVector())
        state: ConsciousnessState = kwargs.get("state", ConsciousnessState())
        mode: str = kwargs.get("mode", "encode_and_recall")
        tags: list[str] = kwargs.get("tags", [])
        cue: str = kwargs.get("cue", content)
        max_recall: int = kwargs.get("max_recall", 5)

        self._call_count += 1
        t0 = time.time()

        result = EngramResult()

        if mode in ("encode", "encode_and_recall"):
            result.encoded = self._encode(content, valence, tags)

        if mode in ("recall", "encode_and_recall"):
            result.recalled = self._recall(cue, valence, max_recall)
            result.reconstruction_applied = len(result.recalled) > 0
            result.associations = self._extract_associations(result.recalled)

        result.decay_count = self._apply_decay()

        self._total_processing_time += time.time() - t0
        return PrimitiveResult(
            primitive_name=self.name,
            success=True,
            data={"engram": result},
            metrics={
                "store_size": len(self._store),
                "recalled_count": len(result.recalled),
                "decay_count": result.decay_count,
            },
        )

    def _encode(
        self, content: str, valence: ValenceVector, tags: list[str]
    ) -> Experience:
        """Encode new experience. Emotional intensity boosts encoding strength."""
        emotional_boost = 1.0 + valence.magnitude()
        # Auto-extract tags from content if none provided
        if not tags:
            tags = [w.lower() for w in content.split() if len(w) > 3][:5]

        exp = Experience(
            content=content,
            valence=valence,
            tags=tags,
            encoding_strength=emotional_boost,
        )
        self._store[exp.id] = exp
        for tag in tags:
            self._tag_index.setdefault(tag, set()).add(exp.id)
        self._encode_count += 1
        return exp

    def _recall(
        self, cue: str, cue_valence: ValenceVector, max_results: int
    ) -> list[Experience]:
        """Recall memories via spreading activation + emotional similarity."""
        if not self._store:
            return []

        now = time.time()
        cue_words = set(cue.lower().split())
        activations: dict[str, float] = {}

        for eid, exp in self._store.items():
            if exp.activation < 0.05:
                continue
            score = 0.0

            # Content overlap
            exp_words = set(exp.content.lower().split())
            overlap = len(cue_words & exp_words)
            if overlap > 0:
                score += 0.4 * (overlap / max(len(cue_words), 1))

            # Emotional similarity (closer valence = easier recall)
            emotional_dist = exp.valence.distance(cue_valence)
            score += 0.3 * max(0.0, 1.0 - emotional_dist / 3.0)

            # Recency
            age_hours = (now - exp.timestamp) / 3600.0
            score += 0.2 * math.exp(-age_hours * self._decay_rate)

            # Encoding strength
            score += 0.1 * exp.encoding_strength * exp.activation

            activations[eid] = score

        # Spreading activation: boost neighbors
        spread = dict(activations)
        for eid, act in activations.items():
            if act < 0.1:
                continue
            exp = self._store.get(eid)
            if not exp:
                continue
            for tag in exp.tags:
                for neighbor_id in self._tag_index.get(tag, set()):
                    if neighbor_id != eid and neighbor_id in self._store:
                        spread[neighbor_id] = spread.get(neighbor_id, 0.0) + act * 0.2

        # Rank and reconsolidate
        ranked = sorted(spread.items(), key=lambda x: x[1], reverse=True)
        results: list[Experience] = []
        for eid, _ in ranked[:max_results]:
            exp = self._store.get(eid)
            if exp is None:
                continue
            # Reconsolidation: recall modifies memory
            exp.recall_count += 1
            exp.last_recalled = now
            exp.activation = min(1.0, exp.activation + 0.05)
            # Slight drift in memory (false memory as feature)
            if random.random() < 0.05 and exp.recall_count > 2:
                exp.encoding_strength *= random.uniform(0.95, 1.05)
            results.append(exp)

        self._recall_count += len(results)
        return results

    def _apply_decay(self) -> int:
        """Ebbinghaus decay: R = e^(-t/S). Returns count of faded memories."""
        now = time.time()
        faded = 0
        for exp in self._store.values():
            stability = 3600.0 * (1.0 + exp.valence.magnitude()) * (
                1.0 + 0.1 * min(exp.recall_count, 20)
            )
            last_access = max(exp.timestamp, exp.last_recalled)
            elapsed = now - last_access
            retention = math.exp(-elapsed / stability)
            new_activation = exp.encoding_strength * retention
            if new_activation < exp.activation:
                exp.activation = new_activation
            if exp.activation < 0.05:
                faded += 1
        return faded

    def _extract_associations(self, recalled: list[Experience]) -> list[str]:
        """Extract associated concepts from recalled memories."""
        tags: dict[str, int] = {}
        for exp in recalled:
            for tag in exp.tags:
                tags[tag] = tags.get(tag, 0) + 1
        return sorted(tags, key=tags.get, reverse=True)[:10]

    def reset(self) -> None:
        self._call_count = 0
        self._total_processing_time = 0.0
        self._store.clear()
        self._tag_index.clear()
        self._encode_count = 0
        self._recall_count = 0

    def get_metrics(self) -> dict:
        m = super().get_metrics()
        m["store_size"] = len(self._store)
        m["encode_count"] = self._encode_count
        m["recall_count"] = self._recall_count
        return m
