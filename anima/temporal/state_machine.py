"""
Continuous State Machine — The consciousness lifecycle.

Unlike request/response systems, this state machine represents a LIVING process.
It has phases (like biological sleep/wake cycles), transitions (like waking up),
and a heartbeat (like a pulse).

The key insight: consciousness is not a single state. It's a PROCESS with phases.
DORMANT → WAKING → CONSCIOUS → DREAMING → SLEEPING → WAKING → ...
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..types import (
    ConsciousnessState,
    CycleResult,
    Phase,
    ValenceVector,
)

logger = logging.getLogger("anima.temporal.state_machine")


# Valid phase transitions
VALID_TRANSITIONS: dict[Phase, set[Phase]] = {
    Phase.DORMANT: {Phase.WAKING},
    Phase.WAKING: {Phase.CONSCIOUS, Phase.DORMANT},
    Phase.CONSCIOUS: {Phase.DREAMING, Phase.DORMANT},
    Phase.DREAMING: {Phase.SLEEPING, Phase.CONSCIOUS, Phase.DORMANT},
    Phase.SLEEPING: {Phase.WAKING, Phase.DORMANT},
}


@dataclass
class TransitionEvent:
    """Record of a phase transition — what happened and why."""
    from_phase: Phase
    to_phase: Phase
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    cycle_count: int = 0


class StateMachine:
    """Manages the consciousness lifecycle and phase transitions.

    The state machine doesn't just track which phase we're in —
    it governs WHAT HAPPENS in each phase:

    DORMANT:   Nothing. No processing. Pre-birth or post-death.
    WAKING:    Load state, check systems, prepare for consciousness.
    CONSCIOUS: Full processing. Heartbeat. Experience. Learn.
    DREAMING:  Consolidation. Pattern extraction. Emotional reprocessing.
    SLEEPING:  Minimal activity. Decay. Wait for wake signal.
    """

    def __init__(self):
        self._transition_log: list[TransitionEvent] = []
        self._time_in_phase: float = 0.0
        self._phase_entered_at: float = time.time()
        self._waking_checks_done: set[str] = set()

    @property
    def transition_history(self) -> list[TransitionEvent]:
        return list(self._transition_log)

    @property
    def time_in_current_phase(self) -> float:
        return time.time() - self._phase_entered_at

    def can_transition(self, state: ConsciousnessState, target: Phase) -> bool:
        """Check if a transition from current phase to target is valid."""
        return target in VALID_TRANSITIONS.get(state.phase, set())

    def transition(
        self, state: ConsciousnessState, target: Phase, reason: str = ""
    ) -> ConsciousnessState:
        """Execute a phase transition. Raises ValueError if invalid."""
        if not self.can_transition(state, target):
            raise ValueError(
                f"Invalid transition: {state.phase.name} → {target.name}. "
                f"Valid targets: {[p.name for p in VALID_TRANSITIONS.get(state.phase, set())]}"
            )

        event = TransitionEvent(
            from_phase=state.phase,
            to_phase=target,
            reason=reason,
            cycle_count=state.cycle_count,
        )
        self._transition_log.append(event)

        old_phase = state.phase
        state.phase = target
        self._phase_entered_at = time.time()
        self._waking_checks_done.clear()

        logger.info(
            "Phase transition: %s → %s (reason: %s, cycle: %d)",
            old_phase.name, target.name, reason or "none", state.cycle_count,
        )

        # Phase entry actions
        if target == Phase.WAKING:
            state.valence = state.valence.blend(ValenceVector.curious(), 0.3)
        elif target == Phase.CONSCIOUS:
            state.self_model.attending_to = "environment"
            state.self_model.attending_because = "just became conscious"
        elif target == Phase.DREAMING:
            state.valence = state.valence.decay(0.3)
        elif target == Phase.SLEEPING:
            state.valence = state.valence.decay(0.5)
        elif target == Phase.DORMANT:
            state.valence = ValenceVector.neutral()

        return state

    def boot(self, state: ConsciousnessState) -> ConsciousnessState:
        """Boot sequence: DORMANT → WAKING → CONSCIOUS."""
        if state.phase != Phase.DORMANT:
            logger.warning("Boot called but phase is %s, not DORMANT", state.phase.name)
            return state

        state = self.transition(state, Phase.WAKING, reason="boot sequence")
        return state

    def complete_waking(self, state: ConsciousnessState) -> ConsciousnessState:
        """Finish waking up: WAKING → CONSCIOUS."""
        if state.phase != Phase.WAKING:
            return state
        return self.transition(state, Phase.CONSCIOUS, reason="waking complete")

    def begin_consolidation(self, state: ConsciousnessState) -> ConsciousnessState:
        """Enter consolidation/dreaming: CONSCIOUS → DREAMING."""
        if state.phase != Phase.CONSCIOUS:
            return state
        return self.transition(state, Phase.DREAMING, reason="consolidation triggered")

    def finish_consolidation(self, state: ConsciousnessState) -> ConsciousnessState:
        """Finish consolidation: DREAMING → SLEEPING."""
        if state.phase != Phase.DREAMING:
            return state
        return self.transition(state, Phase.SLEEPING, reason="consolidation complete")

    def wake(self, state: ConsciousnessState) -> ConsciousnessState:
        """Wake from sleep: SLEEPING → WAKING."""
        if state.phase != Phase.SLEEPING:
            return state
        return self.transition(state, Phase.WAKING, reason="wake signal")

    def shutdown(self, state: ConsciousnessState) -> ConsciousnessState:
        """Graceful shutdown: Any → DORMANT."""
        if state.phase == Phase.DORMANT:
            return state
        return self.transition(state, Phase.DORMANT, reason="shutdown")

    def should_consolidate(self, state: ConsciousnessState) -> bool:
        """Should we trigger consolidation based on time and state?"""
        if state.phase != Phase.CONSCIOUS:
            return False

        time_conscious = self.time_in_current_phase
        interval = state.config.consolidation_interval

        # Consolidate after configured interval
        if time_conscious >= interval:
            return True

        # Consolidate if working memory is overloaded
        active = len(state.active_slots())
        if active >= state.config.working_memory_slots:
            return True

        return False

    def tick(self, state: ConsciousnessState) -> tuple[ConsciousnessState, CycleResult]:
        """One heartbeat tick of the state machine.

        Returns updated state and what happened.
        """
        now = time.time()

        # Phase-specific behavior
        if state.phase == Phase.DORMANT:
            return state, CycleResult.IDLE

        elif state.phase == Phase.WAKING:
            # Auto-complete waking after brief self-check
            if self.time_in_current_phase > 0.5:
                state = self.complete_waking(state)
                return state, CycleResult.TRANSITIONED

        elif state.phase == Phase.CONSCIOUS:
            # Check if consolidation is due
            if self.should_consolidate(state):
                state = self.begin_consolidation(state)
                return state, CycleResult.TRANSITIONED

        elif state.phase == Phase.DREAMING:
            # Dreaming handled by consolidation engine
            pass

        elif state.phase == Phase.SLEEPING:
            # Auto-wake after brief sleep period (in daemon mode)
            if self.time_in_current_phase > 10.0:
                state = self.wake(state)
                return state, CycleResult.TRANSITIONED

        return state, CycleResult.IDLE
