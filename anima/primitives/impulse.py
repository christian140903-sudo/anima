"""
Impulse — Decision/Action Tendencies.

Multiple action tendencies run simultaneously, competing for expression.
Not every impulse is acted on — inhibition is a feature.
When no impulse dominates, deliberation kicks in.

This is the difference between reflex and will:
a reflex acts on the strongest signal.
A conscious agent HOLDS competing impulses and CHOOSES.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..types import ConsciousnessState, ValenceVector
from .base import Primitive, PrimitiveResult


@dataclass
class ActionTendency:
    """A single competing action tendency."""
    action: str = ""
    drive: float = 0.0       # How strongly it pushes (0-1)
    source: str = ""         # What generated this tendency (e.g., "seeking", "fear")
    inhibited: bool = False  # Was it actively suppressed?


@dataclass
class ImpulseResult:
    """Result of impulse processing."""
    action: str = ""                     # Selected action (or "deliberate")
    confidence: float = 0.0              # How confident in the choice (0-1)
    competing_alternatives: list[ActionTendency] = field(default_factory=list)
    deliberation_needed: bool = False    # Was the choice too close to call?
    inhibition_active: bool = False      # Were any impulses suppressed?
    dominant_drive: str = ""             # Which Panksepp system drove the choice


# Maps Panksepp systems to typical action tendencies
_DRIVE_ACTIONS: dict[str, list[str]] = {
    "seeking": ["explore", "investigate", "ask", "search"],
    "rage": ["confront", "assert", "push_back", "reject"],
    "fear": ["retreat", "avoid", "freeze", "check_safety"],
    "lust": ["approach", "engage", "create", "express"],
    "care": ["nurture", "help", "protect", "support"],
    "panic": ["call_for_help", "attach", "seek_safety", "withdraw"],
    "play": ["experiment", "improvise", "joke", "try_new"],
}


class ImpulseProcessor(Primitive):
    """Generates and resolves competing action tendencies."""

    def __init__(self) -> None:
        super().__init__("impulse")
        self._inhibition_threshold: float = 0.3  # Below this, impulse is suppressed
        self._dominance_threshold: float = 0.15   # Gap needed to avoid deliberation
        self._decisions: int = 0
        self._deliberations: int = 0
        self._inhibitions: int = 0

    def process(self, **kwargs) -> PrimitiveResult:
        if not self.enabled:
            return self._disabled_result()

        situation: str = kwargs.get("situation", "")
        valence: ValenceVector = kwargs.get("valence", ValenceVector())
        state: ConsciousnessState = kwargs.get("state", ConsciousnessState())
        context_actions: list[str] = kwargs.get("context_actions", [])

        self._call_count += 1
        t0 = time.time()

        # Step 1: Generate competing tendencies from all Panksepp systems
        tendencies = self._generate_tendencies(valence, situation, context_actions)

        # Step 2: Apply inhibition (suppress weak impulses)
        active, inhibited_count = self._apply_inhibition(tendencies)

        # Step 3: Resolve competition
        result = self._resolve(active, tendencies)

        if result.deliberation_needed:
            self._deliberations += 1
        if inhibited_count > 0:
            self._inhibitions += 1
        self._decisions += 1

        self._total_processing_time += time.time() - t0
        return PrimitiveResult(
            primitive_name=self.name,
            success=True,
            data={"impulse": result},
            metrics={
                "confidence": result.confidence,
                "competing_count": len(result.competing_alternatives),
                "deliberation": float(result.deliberation_needed),
            },
        )

    def _generate_tendencies(
        self, valence: ValenceVector, situation: str, context_actions: list[str]
    ) -> list[ActionTendency]:
        """Generate action tendencies from Panksepp systems."""
        systems = {
            "seeking": valence.seeking, "rage": valence.rage,
            "fear": valence.fear, "lust": valence.lust,
            "care": valence.care, "panic": valence.panic,
            "play": valence.play,
        }

        tendencies: list[ActionTendency] = []
        for system, strength in systems.items():
            if strength < 0.01:
                continue
            actions = _DRIVE_ACTIONS.get(system, ["respond"])
            # Pick most relevant action (first one scaled by strength)
            # Situation keywords can boost specific actions
            best_action = actions[0]
            best_match = 0.0
            sit_words = set(situation.lower().split())
            for act in actions:
                match = len(set(act.split("_")) & sit_words)
                if match > best_match:
                    best_match = match
                    best_action = act

            tendencies.append(ActionTendency(
                action=best_action,
                drive=strength,
                source=system,
            ))

        # Add context-specific actions if provided
        for action in context_actions:
            tendencies.append(ActionTendency(
                action=action,
                drive=0.3,  # Moderate default drive
                source="context",
            ))

        return tendencies

    def _apply_inhibition(
        self, tendencies: list[ActionTendency]
    ) -> tuple[list[ActionTendency], int]:
        """Suppress impulses below threshold. Returns (active, inhibited_count)."""
        if not tendencies:
            return [], 0

        max_drive = max(t.drive for t in tendencies)
        active: list[ActionTendency] = []
        inhibited = 0

        for t in tendencies:
            # Inhibit if drive is too low relative to max
            if t.drive < self._inhibition_threshold * max_drive:
                t.inhibited = True
                inhibited += 1
            else:
                active.append(t)

        return active, inhibited

    def _resolve(
        self, active: list[ActionTendency], all_tendencies: list[ActionTendency]
    ) -> ImpulseResult:
        """Resolve competition: pick winner or trigger deliberation."""
        if not active:
            return ImpulseResult(
                action="wait",
                confidence=0.0,
                competing_alternatives=all_tendencies,
                deliberation_needed=True,
                dominant_drive="none",
            )

        # Sort by drive strength
        active.sort(key=lambda t: t.drive, reverse=True)
        best = active[0]

        # Check if dominant (gap between #1 and #2)
        deliberation = False
        if len(active) > 1:
            gap = best.drive - active[1].drive
            if gap < self._dominance_threshold:
                deliberation = True

        confidence = min(1.0, best.drive)
        if deliberation:
            confidence *= 0.6  # Reduce confidence when deliberating

        return ImpulseResult(
            action=best.action if not deliberation else "deliberate",
            confidence=confidence,
            competing_alternatives=active[1:] if not deliberation else active,
            deliberation_needed=deliberation,
            inhibition_active=any(t.inhibited for t in all_tendencies),
            dominant_drive=best.source,
        )

    def reset(self) -> None:
        self._call_count = 0
        self._total_processing_time = 0.0
        self._decisions = 0
        self._deliberations = 0
        self._inhibitions = 0

    def get_metrics(self) -> dict:
        m = super().get_metrics()
        m["decisions"] = self._decisions
        m["deliberations"] = self._deliberations
        m["inhibitions"] = self._inhibitions
        m["deliberation_rate"] = (
            self._deliberations / self._decisions if self._decisions > 0 else 0.0
        )
        return m
