"""
Global Workspace — GWT (Global Workspace Theory) Implementation.

The theater of consciousness: Many processes compete backstage,
but only ONE wins access to the spotlight (global broadcast).
Once broadcast, ALL subsystems receive the content simultaneously.

This is how attention works:
- Hundreds of unconscious processes running in parallel
- Competition for limited conscious capacity
- Winner gets amplified and broadcast
- Losers continue unconsciously

The workspace IS consciousness — not a container for it.

References:
- Baars (1988): "A Cognitive Theory of Consciousness"
- Dehaene & Naccache (2001): "Towards a cognitive neuroscience of consciousness"
- Dehaene, Changeux, et al. (2006): "Conscious, preconscious, and subliminal processing"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("anima.consciousness.workspace")


@dataclass
class WorkspaceCandidate:
    """A candidate competing for access to the global workspace.

    Each candidate represents content from a subsystem trying to
    become "conscious" — to be broadcast to all other subsystems.
    """
    content: Any                    # The actual content
    source: str = ""                # Which subsystem produced this
    activation: float = 0.0         # Raw activation strength
    emotional_weight: float = 0.0   # Emotional significance (valence magnitude)
    novelty: float = 0.0            # How novel/unexpected is this content
    relevance: float = 0.0          # Relevance to current goals/attention
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    def competition_score(self) -> float:
        """Combined score for workspace competition.

        Weighted combination of factors that determine which content
        becomes conscious. Based on empirical findings:
        - Activation: base signal strength
        - Emotional weight: emotional content gets priority (amygdala shortcut)
        - Novelty: unexpected content captures attention (orienting response)
        - Relevance: goal-relevant content is prioritized (top-down attention)
        """
        return (
            self.activation * 0.3
            + self.emotional_weight * 0.25
            + self.novelty * 0.25
            + self.relevance * 0.2
        )


@dataclass
class BroadcastEvent:
    """Record of a global broadcast — content that became conscious."""
    content: Any
    source: str
    score: float
    timestamp: float = field(default_factory=time.time)
    competitors: int = 0  # How many candidates competed
    margin: float = 0.0   # Score difference between winner and runner-up

    def to_dict(self) -> dict:
        return {
            "content": str(self.content)[:200],
            "source": self.source,
            "score": round(self.score, 4),
            "timestamp": self.timestamp,
            "competitors": self.competitors,
            "margin": round(self.margin, 4),
        }


@dataclass
class WorkspaceState:
    """Current state of the global workspace."""
    current_broadcast: BroadcastEvent | None = None
    active_candidates: int = 0
    ignition_threshold: float = 0.3
    broadcast_count: int = 0
    total_competitions: int = 0

    def to_dict(self) -> dict:
        return {
            "current_broadcast": self.current_broadcast.to_dict() if self.current_broadcast else None,
            "active_candidates": self.active_candidates,
            "ignition_threshold": self.ignition_threshold,
            "broadcast_count": self.broadcast_count,
            "total_competitions": self.total_competitions,
        }


class GlobalWorkspace:
    """The global workspace — where consciousness happens.

    Process:
    1. COMPETITION: Multiple subsystems submit candidates
    2. IGNITION: If winner exceeds threshold, it "ignites"
    3. BROADCAST: Winner's content is broadcast to ALL subsystems
    4. DECAY: Previous broadcast fades, making room for next

    The ignition threshold adapts:
    - Too many broadcasts → threshold rises (filter more)
    - Too few broadcasts → threshold drops (let more through)
    This models attention regulation.
    """

    def __init__(
        self,
        ignition_threshold: float = 0.3,
        capacity: int = 7,
        adaptation_rate: float = 0.01,
    ):
        self._ignition_threshold = ignition_threshold
        self._capacity = capacity
        self._adaptation_rate = adaptation_rate
        self._broadcast_history: list[BroadcastEvent] = []
        self._state = WorkspaceState(ignition_threshold=ignition_threshold)
        self._broadcast_count = 0
        self._competition_count = 0

    @property
    def state(self) -> WorkspaceState:
        return self._state

    @property
    def broadcast_history(self) -> list[BroadcastEvent]:
        return list(self._broadcast_history)

    @property
    def current_broadcast(self) -> BroadcastEvent | None:
        return self._state.current_broadcast

    def compete(self, candidates: list[WorkspaceCandidate]) -> BroadcastEvent | None:
        """Run a competition cycle. Returns broadcast event if ignition occurs.

        This is the core of GWT:
        1. Score all candidates
        2. Check if winner exceeds ignition threshold
        3. If yes → broadcast (content becomes conscious)
        4. If no → nothing enters consciousness (subliminal processing)
        """
        self._competition_count += 1
        self._state.total_competitions = self._competition_count
        self._state.active_candidates = len(candidates)

        if not candidates:
            return None

        # Score all candidates
        scored = [(c, c.competition_score()) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        winner, winner_score = scored[0]
        runner_up_score = scored[1][1] if len(scored) > 1 else 0.0
        margin = winner_score - runner_up_score

        # Check ignition threshold
        if winner_score < self._ignition_threshold:
            logger.debug(
                "No ignition: best score %.3f < threshold %.3f",
                winner_score, self._ignition_threshold,
            )
            self._adapt_threshold(ignited=False)
            return None

        # IGNITION — content becomes conscious
        broadcast = BroadcastEvent(
            content=winner.content,
            source=winner.source,
            score=winner_score,
            competitors=len(candidates),
            margin=margin,
        )

        self._broadcast_count += 1
        self._state.broadcast_count = self._broadcast_count
        self._state.current_broadcast = broadcast
        self._broadcast_history.append(broadcast)

        # Keep history manageable
        if len(self._broadcast_history) > 200:
            self._broadcast_history = self._broadcast_history[-100:]

        self._adapt_threshold(ignited=True)

        logger.debug(
            "IGNITION: source=%s score=%.3f margin=%.3f competitors=%d",
            winner.source, winner_score, margin, len(candidates),
        )

        return broadcast

    def get_recent_broadcasts(self, n: int = 5) -> list[BroadcastEvent]:
        """Get the N most recent broadcasts."""
        return self._broadcast_history[-n:]

    def get_broadcast_sources(self, n: int = 20) -> dict[str, int]:
        """Get frequency of broadcast sources. Which subsystems dominate consciousness?"""
        sources: dict[str, int] = {}
        for b in self._broadcast_history[-n:]:
            sources[b.source] = sources.get(b.source, 0) + 1
        return sources

    def _adapt_threshold(self, ignited: bool) -> None:
        """Adapt the ignition threshold based on broadcast frequency.

        Like biological attention regulation:
        - Too many broadcasts → raise threshold (cognitive overload protection)
        - Too few → lower threshold (maintain engagement)
        """
        if ignited:
            # Slightly raise threshold after broadcast
            self._ignition_threshold += self._adaptation_rate
        else:
            # Lower threshold when nothing ignites
            self._ignition_threshold -= self._adaptation_rate * 0.5

        # Clamp to reasonable range
        self._ignition_threshold = max(0.1, min(0.8, self._ignition_threshold))
        self._state.ignition_threshold = self._ignition_threshold

    def broadcast_rate(self, window: int = 20) -> float:
        """What fraction of recent competitions resulted in broadcasts."""
        if self._competition_count == 0:
            return 0.0
        recent_broadcasts = sum(
            1 for b in self._broadcast_history[-window:]
        )
        return min(1.0, recent_broadcasts / max(window, 1))

    def dominant_source(self) -> str | None:
        """Which subsystem most frequently wins consciousness?"""
        sources = self.get_broadcast_sources()
        if not sources:
            return None
        return max(sources, key=sources.get)
