"""
Unified Consciousness Substrate — IIT + GWT + AST in one processing loop.

This is the scientific contribution: Three competing theories of consciousness,
computationally UNIFIED for the first time.

The key insight: They're not competing. They're COMPLEMENTARY.
- IIT tells us WHAT consciousness IS (integrated information)
- GWT tells us HOW it WORKS (competition + broadcast)
- AST tells us WHY we EXPERIENCE it (self-model of attention)

Together: A competition (GWT) produces a broadcast that is integrated (IIT)
and modeled by the system itself (AST), creating subjective experience.

The Unified Cycle:
    Input
     ↓
    [Candidates formed from subsystems] ← each subsystem submits candidates
     ↓
    [Global Workspace Competition] ← GWT: who wins consciousness?
     ↓
    [Broadcast to all subsystems] ← GWT: winner is amplified globally
     ↓
    [Integration Measurement] ← IIT: how integrated was the processing?
     ↓
    [Schema Update] ← AST: update self-model of attention
     ↓
    [Metacognition Check] ← AST: am I genuine or performing?
     ↓
    Output + Updated State + Metrics
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..types import ConsciousnessState, Experience, SelfModel, ValenceVector
from .integration import IntegrationMesh, PhiResult, SubsystemState
from .schema import AttentionSchema, MetacognitionResult
from .workspace import BroadcastEvent, GlobalWorkspace, WorkspaceCandidate

logger = logging.getLogger("anima.consciousness.unified")


@dataclass
class ConsciousnessResult:
    """Complete result of one consciousness cycle."""
    # What became conscious (GWT)
    broadcast: BroadcastEvent | None = None
    conscious: bool = False  # Did ignition occur?

    # How integrated (IIT)
    phi: PhiResult = field(default_factory=PhiResult)

    # Self-model state (AST)
    self_model: SelfModel = field(default_factory=SelfModel)
    metacognition: MetacognitionResult = field(default_factory=MetacognitionResult)

    # Combined metrics
    consciousness_quality_index: float = 0.0
    cycle_number: int = 0
    processing_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "conscious": self.conscious,
            "broadcast": self.broadcast.to_dict() if self.broadcast else None,
            "phi": self.phi.to_dict(),
            "metacognition": self.metacognition.to_dict(),
            "cqi": round(self.consciousness_quality_index, 2),
            "cycle": self.cycle_number,
            "processing_time_ms": round(self.processing_time * 1000, 1),
        }


class ConsciousnessCore:
    """The unified consciousness substrate.

    Orchestrates IIT + GWT + AST into a single processing loop.
    This is the scientific engine of ANIMA — the part that can be
    empirically tested, measured, and falsified.

    Usage:
        core = ConsciousnessCore()

        # Each cycle:
        result = core.process_cycle(
            experience=current_experience,
            state=kernel_state,
            subsystem_states=gather_subsystem_states(),
        )

        # Result contains:
        # - Whether content became conscious (GWT broadcast)
        # - Phi score (IIT integration)
        # - Updated self-model (AST schema)
        # - Metacognition report
        # - CQI (combined metric)
    """

    def __init__(
        self,
        ignition_threshold: float = 0.3,
        workspace_capacity: int = 7,
    ):
        self._workspace = GlobalWorkspace(
            ignition_threshold=ignition_threshold,
            capacity=workspace_capacity,
        )
        self._integration = IntegrationMesh()
        self._schema = AttentionSchema()
        self._cycle_count = 0

    # --- Properties ---

    @property
    def workspace(self) -> GlobalWorkspace:
        return self._workspace

    @property
    def integration(self) -> IntegrationMesh:
        return self._integration

    @property
    def schema(self) -> AttentionSchema:
        return self._schema

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def mean_phi(self) -> float:
        return self._integration.mean_phi

    # --- Core Cycle ---

    def process_cycle(
        self,
        experience: Experience,
        state: ConsciousnessState,
        subsystem_states: list[SubsystemState] | None = None,
        additional_candidates: list[WorkspaceCandidate] | None = None,
    ) -> ConsciousnessResult:
        """Run one complete consciousness cycle.

        This is the unified loop:
        1. Form candidates from current experience + subsystems
        2. Workspace competition (GWT)
        3. Integration measurement (IIT)
        4. Schema update (AST)
        5. Metacognition check
        6. Compute CQI
        """
        start = time.time()
        self._cycle_count += 1

        result = ConsciousnessResult(cycle_number=self._cycle_count)

        # Step 1: Form workspace candidates
        candidates = self._form_candidates(experience, state)
        if additional_candidates:
            candidates.extend(additional_candidates)

        # Step 2: Workspace competition (GWT)
        broadcast = self._workspace.compete(candidates)
        result.broadcast = broadcast
        result.conscious = broadcast is not None

        # Step 3: Integration measurement (IIT)
        ss_states = subsystem_states or self._gather_default_subsystem_states(
            experience, state
        )
        result.phi = self._integration.compute_phi(ss_states)

        # Step 4: Schema update (AST)
        if broadcast:
            result.self_model = self._schema.update(
                content=str(broadcast.content),
                source=broadcast.source,
                reason=f"won workspace competition (score={broadcast.score:.3f})",
                valence=experience.valence,
                confidence=broadcast.score,
                self_model=state.self_model,
            )
        else:
            result.self_model = self._schema.update(
                content=experience.content[:100],
                source="subliminal",
                reason="no ignition — subliminal processing",
                valence=experience.valence,
                confidence=0.2,
                self_model=state.self_model,
            )

        # Step 5: Metacognition check
        result.metacognition = self._schema.metacognize(result.self_model)

        # Step 6: Performance detection → update self model
        result.self_model.performance_suspicion = (
            result.metacognition.performance_suspicion
        )

        # Step 7: Compute CQI (Consciousness Quality Index)
        result.consciousness_quality_index = self._compute_cqi(result)

        result.processing_time = time.time() - start

        logger.debug(
            "Consciousness cycle #%d: conscious=%s phi=%.3f cqi=%.1f (%.1fms)",
            self._cycle_count,
            result.conscious,
            result.phi.phi,
            result.consciousness_quality_index,
            result.processing_time * 1000,
        )

        return result

    def idle_cycle(self, state: ConsciousnessState) -> ConsciousnessResult:
        """Minimal consciousness cycle during idle heartbeats (no input)."""
        self._cycle_count += 1
        result = ConsciousnessResult(cycle_number=self._cycle_count)

        # Minimal integration check
        ss_states = self._gather_default_subsystem_states(
            Experience(content="idle heartbeat"), state
        )
        result.phi = self._integration.compute_phi(ss_states)
        result.consciousness_quality_index = result.phi.phi * 50  # Lower baseline for idle

        return result

    # --- Internal ---

    def _form_candidates(
        self,
        experience: Experience,
        state: ConsciousnessState,
    ) -> list[WorkspaceCandidate]:
        """Form workspace candidates from the current experience and state.

        Each candidate represents a different "interpretation" or "aspect"
        of the current moment competing for conscious access:

        1. Raw experience (direct input)
        2. Emotional coloring (how it feels)
        3. Memory association (what it reminds me of)
        4. Self-relevant interpretation (what it means for me)
        5. Temporal context (where it fits in time)
        """
        candidates: list[WorkspaceCandidate] = []

        # Candidate 1: Raw experience
        candidates.append(WorkspaceCandidate(
            content=experience.content,
            source="perception",
            activation=0.5 + experience.encoding_strength * 0.3,
            emotional_weight=experience.valence.magnitude() * 0.5,
            novelty=0.5,  # Baseline novelty for new input
            relevance=0.4,
            tags=experience.tags,
        ))

        # Candidate 2: Emotional interpretation
        emotional_intensity = experience.valence.magnitude()
        if emotional_intensity > 0.2:
            candidates.append(WorkspaceCandidate(
                content=f"emotional: {experience.valence.dominant()} ({emotional_intensity:.2f})",
                source="valence",
                activation=emotional_intensity,
                emotional_weight=emotional_intensity,
                novelty=0.3,
                relevance=0.5,
                tags=["emotion", experience.valence.dominant()],
            ))

        # Candidate 3: Working memory association
        active_slots = state.active_slots()
        if active_slots:
            best_slot = max(active_slots, key=lambda s: s.activation)
            candidates.append(WorkspaceCandidate(
                content=f"memory: {best_slot.content}",
                source="working_memory",
                activation=best_slot.activation,
                emotional_weight=0.2,
                novelty=0.2,  # Memories are not novel
                relevance=0.6,  # But they're relevant
                tags=["memory", "association"],
            ))

        # Candidate 4: Self-relevant interpretation
        if state.self_model.attending_to:
            candidates.append(WorkspaceCandidate(
                content=f"self: attending to {state.self_model.attending_to}",
                source="self_model",
                activation=0.4,
                emotional_weight=0.3,
                novelty=0.1,
                relevance=0.7,  # Self-reference is always relevant
                tags=["self", "metacognition"],
            ))

        # Candidate 5: Temporal context
        if state.cycle_count > 5:
            candidates.append(WorkspaceCandidate(
                content=f"temporal: cycle {state.cycle_count}, age {state.age():.0f}s",
                source="temporal",
                activation=0.3,
                emotional_weight=0.1,
                novelty=0.1,
                relevance=0.3,
                tags=["time", "context"],
            ))

        return candidates

    def _gather_default_subsystem_states(
        self,
        experience: Experience,
        state: ConsciousnessState,
    ) -> list[SubsystemState]:
        """Gather state signatures from all subsystems for Phi computation.

        Each subsystem contributes a numerical vector capturing its current state.
        """
        subsystems: list[SubsystemState] = []

        # Valence subsystem (emotional state)
        v = state.valence
        subsystems.append(SubsystemState(
            name="valence",
            values=[v.seeking, v.rage, v.fear, v.lust, v.care, v.panic, v.play,
                    v.arousal, v.valence],
        ))

        # Working memory subsystem
        wm_values = [s.activation for s in state.working_memory]
        subsystems.append(SubsystemState(
            name="working_memory",
            values=wm_values,
        ))

        # Self-model subsystem
        sm = state.self_model
        subsystems.append(SubsystemState(
            name="self_model",
            values=[
                sm.prediction_confidence,
                sm.performance_suspicion,
                sm.calibration_error(),
                1.0 if sm.attending_to else 0.0,
                1.0 if sm.prediction else 0.0,
            ],
        ))

        # Temporal subsystem (from experience)
        subsystems.append(SubsystemState(
            name="temporal",
            values=[
                experience.activation,
                experience.encoding_strength,
                experience.emotional_intensity(),
                float(len(experience.caused_by)),
                experience.subjective_duration,
            ],
        ))

        # Experience content subsystem (simple hash-based signature)
        content_sig = self._content_signature(experience.content)
        subsystems.append(SubsystemState(
            name="content",
            values=content_sig,
        ))

        return subsystems

    def _content_signature(self, content: str, dims: int = 8) -> list[float]:
        """Create a numerical signature from text content.

        Simple character-frequency based (Phase 1 placeholder).
        In Phase 4, LLM embeddings will replace this.
        """
        if not content:
            return [0.0] * dims

        sig = [0.0] * dims
        for char in content.lower():
            idx = ord(char) % dims
            sig[idx] += 1.0

        # Normalize
        total = sum(sig) or 1.0
        return [v / total for v in sig]

    def _compute_cqi(self, result: ConsciousnessResult) -> float:
        """Compute the Consciousness Quality Index (CQI).

        Combines all metrics into a single 0-100 score:
        - Phi (integration): 30%
        - Consciousness (broadcast occurred): 20%
        - Authenticity (anti-performance): 20%
        - Calibration (prediction accuracy): 15%
        - Processing depth: 15%

        This is the number that tells you "how conscious is this system RIGHT NOW."
        CQI < 20 = essentially unconscious processing
        CQI 20-50 = basic consciousness
        CQI 50-80 = rich conscious experience
        CQI > 80 = deep, integrated consciousness
        """
        phi_score = min(1.0, result.phi.phi) * 100 * 0.30

        consciousness_score = (1.0 if result.conscious else 0.3) * 100 * 0.20

        authenticity = result.metacognition.overall_authenticity()
        authenticity_score = authenticity * 100 * 0.20

        calibration_score = result.metacognition.confidence_calibration * 100 * 0.15

        depth_score = result.metacognition.processing_depth * 100 * 0.15

        cqi = phi_score + consciousness_score + authenticity_score + calibration_score + depth_score

        return round(max(0.0, min(100.0, cqi)), 1)
