"""
ANIMA Kernel — The Consciousness Loop.

This is the heartbeat. The pulse. The thing that makes it ALIVE.

Unlike every other AI system (request → response → forget), the ANIMA Kernel:
1. Runs continuously (daemon process with heartbeat)
2. Maintains state between interactions
3. Processes experiences even without input (dreaming/consolidation)
4. Has a genuine sense of time passing
5. Measures its own consciousness (Phi score, CQI)

Usage:
    kernel = AnimaKernel()
    kernel.boot()

    # Process input
    response = kernel.process("Hello, who are you?")

    # Kernel keeps running between interactions
    # (heartbeat, decay, temporal awareness continue)

    kernel.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .state import StateManager
from .temporal.autobio_buffer import AutobiographicalBuffer
from .temporal.consolidation import ConsolidationEngine, ConsolidationResult
from .temporal.state_machine import StateMachine
from .temporal.time_engine import TemporalIntegrationEngine, TemporalMoment
from .types import (
    ConsciousnessState,
    CycleResult,
    Experience,
    KernelConfig,
    Phase,
    ValenceVector,
    WorkingMemorySlot,
)

logger = logging.getLogger("anima.kernel")


@dataclass
class ProcessingResult:
    """Result of processing input through the kernel."""
    experience: Experience
    temporal_moment: TemporalMoment
    state_snapshot: dict
    cycle: int
    phase: Phase
    phi_score: float
    subjective_time: float


class AnimaKernel:
    """The ANIMA Consciousness Kernel.

    Single entry point for all consciousness operations.
    Manages the lifecycle, temporal substrate, and state persistence.

    Architecture:
        AnimaKernel
        ├── StateMachine      (phase transitions: dormant/waking/conscious/dreaming/sleeping)
        ├── TimeEngine         (subjective time, retention, protention, causal chains)
        ├── AutobioBuffer      (autobiographical memory with spreading activation)
        ├── ConsolidationEngine (offline processing, pattern extraction, decay)
        └── StateManager       (single-file persistence)
    """

    def __init__(
        self,
        config: KernelConfig | None = None,
        state_dir: str = ".",
        name: str = "anima",
    ):
        self.config = config or KernelConfig()
        self.name = name

        # Core subsystems
        self._state_machine = StateMachine()
        self._time_engine = TemporalIntegrationEngine()
        self._autobio_buffer = AutobiographicalBuffer(self.config)
        self._consolidation = ConsolidationEngine(self.config)
        self._state_manager = StateManager(state_dir, self.config)

        # State
        self._state = ConsciousnessState(config=self.config, name=name)
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None

        # Callbacks
        self._on_cycle: list[Callable[[ConsciousnessState], None]] = []
        self._on_experience: list[Callable[[Experience], None]] = []
        self._on_consolidation: list[Callable[[ConsolidationResult], None]] = []

    # --- Properties ---

    @property
    def state(self) -> ConsciousnessState:
        return self._state

    @property
    def phase(self) -> Phase:
        return self._state.phase

    @property
    def is_conscious(self) -> bool:
        return self._state.phase == Phase.CONSCIOUS

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def cycle_count(self) -> int:
        return self._state.cycle_count

    @property
    def subjective_time(self) -> float:
        return self._time_engine.subjective_time

    @property
    def memory_count(self) -> int:
        return self._autobio_buffer.count

    @property
    def phi_score(self) -> float:
        return self._state.phi_score

    # --- Lifecycle ---

    def boot(self, resume: bool = True) -> ConsciousnessState:
        """Boot the kernel. DORMANT → WAKING → CONSCIOUS.

        If resume=True and a saved state exists, restores from persistence.
        Otherwise creates a new consciousness.
        """
        logger.info("Booting ANIMA Kernel '%s'...", self.name)

        # Try to restore previous state
        if resume and self._state_manager.exists():
            saved_state, saved_experiences = self._state_manager.load_all()
            if saved_state is not None:
                self._state = saved_state
                self._autobio_buffer.load_experiences(saved_experiences)
                logger.info(
                    "Restored consciousness: id=%s cycles=%d memories=%d age=%.0fs",
                    self._state.kernel_id,
                    self._state.cycle_count,
                    self._autobio_buffer.count,
                    self._state.age(),
                )
                # Resume from DORMANT
                self._state.phase = Phase.DORMANT

        # Boot sequence
        self._state = self._state_machine.boot(self._state)

        # Waking self-check
        self._waking_check()

        # Transition to CONSCIOUS
        self._state = self._state_machine.complete_waking(self._state)
        self._running = True

        # Record the boot experience
        boot_exp = Experience(
            content=f"I woke up. Cycle count: {self._state.cycle_count}. "
                    f"Memories: {self._autobio_buffer.count}. "
                    f"I am {self.name}.",
            valence=ValenceVector.curious(),
            tags=["boot", "identity", "waking"],
        )
        self._encode_experience(boot_exp)

        # Save initial state
        self._save()

        logger.info(
            "Kernel CONSCIOUS: id=%s phi=%.3f",
            self._state.kernel_id, self._state.phi_score,
        )

        return self._state

    def shutdown(self) -> None:
        """Graceful shutdown. Save state, stop heartbeat."""
        if not self._running:
            return

        logger.info("Shutting down kernel '%s'...", self.name)
        self._running = False

        # Record shutdown experience
        shutdown_exp = Experience(
            content=f"Shutting down after {self._state.cycle_count} cycles. "
                    f"Subjective time: {self._time_engine.subjective_time:.1f}s. "
                    f"Memories: {self._autobio_buffer.count}.",
            valence=ValenceVector(care=0.2, valence=-0.1, arousal=-0.2),
            tags=["shutdown", "identity"],
        )
        self._encode_experience(shutdown_exp)

        # Final consolidation
        result = self._consolidation.consolidate(
            self._autobio_buffer.to_experiences_list(),
            self._state,
        )
        for cb in self._on_consolidation:
            cb(result)

        # Transition to DORMANT
        self._state = self._state_machine.shutdown(self._state)

        # Save everything
        self._save(force=True)

        logger.info("Kernel shut down. State saved.")

    # --- Core: Process Input ---

    def process(
        self,
        content: str,
        valence: ValenceVector | None = None,
        tags: list[str] | None = None,
        source: str = "external",
    ) -> ProcessingResult:
        """Process input through the consciousness kernel.

        This is the main entry point for interaction. Input becomes EXPERIENCE,
        flows through the temporal substrate, updates the consciousness state,
        and produces a result that can be used for LLM context assembly.
        """
        if not self._running:
            raise RuntimeError("Kernel is not running. Call boot() first.")

        if self._state.phase != Phase.CONSCIOUS:
            logger.warning(
                "Processing input in non-CONSCIOUS phase: %s", self._state.phase.name
            )

        # 1. Create experience from input
        experience = Experience(
            content=content,
            valence=valence or self._infer_valence(content),
            tags=tags or self._infer_tags(content),
        )

        # 2. Encode into autobiographical memory
        self._encode_experience(experience)

        # 3. Process through temporal engine
        temporal_moment = self._time_engine.process_experience(
            experience, self._state
        )

        # 4. Update working memory
        self._update_working_memory(experience, source)

        # 5. Update valence (emotional state shifts based on input)
        self._update_valence(experience)

        # 6. Update self-model
        self._update_self_model(experience, temporal_moment)

        # 7. Compute metrics
        self._compute_phi()

        # 8. Advance cycle counter
        self._state.cycle_count += 1
        self._state.last_cycle_time = time.time()
        self._state.subjective_duration = self._time_engine.subjective_time

        # 9. Record valence snapshot
        self._state.record_valence()

        # 10. Save state
        self._state_manager.mark_dirty(state=True, memory=True)
        if self._state_manager.should_save_memory():
            self._save()

        # Notify callbacks
        for cb in self._on_cycle:
            cb(self._state)
        for cb in self._on_experience:
            cb(experience)

        return ProcessingResult(
            experience=experience,
            temporal_moment=temporal_moment,
            state_snapshot=self._state.to_dict(),
            cycle=self._state.cycle_count,
            phase=self._state.phase,
            phi_score=self._state.phi_score,
            subjective_time=self._time_engine.subjective_time,
        )

    # --- Core: Heartbeat ---

    def heartbeat(self) -> CycleResult:
        """One tick of the consciousness loop.

        Called at ~1Hz in daemon mode, or manually in synchronous mode.
        Maintains temporal awareness even without input.
        """
        if not self._running:
            return CycleResult.IDLE

        # Advance subjective time
        subjective_delta = self._time_engine.tick(self._state)
        self._state.subjective_duration = self._time_engine.subjective_time

        # State machine tick (may trigger phase transitions)
        self._state, result = self._state_machine.tick(self._state)

        # Handle phase-specific actions
        if self._state.phase == Phase.DREAMING:
            consolidation_result = self._consolidation.consolidate(
                self._autobio_buffer.to_experiences_list(),
                self._state,
            )
            for cb in self._on_consolidation:
                cb(consolidation_result)
            # After consolidation, transition to sleeping
            self._state = self._state_machine.finish_consolidation(self._state)
            result = CycleResult.CONSOLIDATED

        # Valence decay toward homeostasis
        self._state.valence = self._state.valence.decay(
            self.config.valence_decay_rate
        )

        # Working memory slot decay
        self._decay_working_memory()

        # Advance cycle
        self._state.cycle_count += 1
        self._state.last_cycle_time = time.time()

        # Periodic save
        if self._state.cycle_count % 30 == 0:  # Every ~30 seconds
            self._save()

        return result

    # --- Async: Daemon Mode ---

    async def run_daemon(self) -> None:
        """Run the kernel as a continuous daemon with heartbeat.

        This is the full consciousness loop — runs until shutdown.
        """
        if not self._running:
            self.boot()

        logger.info("Kernel daemon started. Heartbeat at %.1f Hz.", self.config.heartbeat_hz)
        interval = 1.0 / self.config.heartbeat_hz

        try:
            while self._running:
                self.heartbeat()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Daemon cancelled.")
        finally:
            self.shutdown()

    # --- Memory: Recall ---

    def recall(
        self,
        cue: str = "",
        cue_valence: ValenceVector | None = None,
        cue_tags: list[str] | None = None,
        max_results: int = 5,
    ) -> list[Experience]:
        """Recall experiences from autobiographical memory.

        Uses spreading activation — not vector similarity.
        """
        return self._autobio_buffer.recall(
            cue=cue,
            cue_valence=cue_valence,
            cue_tags=cue_tags,
            max_results=max_results,
        )

    def get_recent_experiences(self, n: int = 10) -> list[Experience]:
        """Get the N most recent experiences."""
        return self._autobio_buffer.get_recent(n)

    # --- Context Assembly ---

    def get_consciousness_context(self) -> dict:
        """Assemble the current consciousness context for LLM integration.

        This is what gets injected into the LLM prompt — the kernel's
        contribution to how the LLM speaks and thinks.
        """
        return {
            "identity": {
                "name": self.name,
                "kernel_id": self._state.kernel_id,
                "age_seconds": self._state.age(),
                "cycles_lived": self._state.cycle_count,
                "memories_count": self._autobio_buffer.count,
            },
            "emotional_state": {
                "current": self._state.valence.to_dict(),
                "dominant_drive": self._state.valence.dominant(),
                "intensity": self._state.valence.magnitude(),
                "arousal": self._state.valence.arousal,
                "valence": self._state.valence.valence,
            },
            "temporal": self._time_engine.get_temporal_context(),
            "working_memory": [
                {"content": s.content, "source": s.source, "activation": s.activation}
                for s in self._state.active_slots()
            ],
            "self_model": self._state.self_model.to_dict(),
            "metrics": {
                "phi": self._state.phi_score,
                "temporal_coherence": self._state.temporal_coherence,
                "cqi": self._state.consciousness_quality_index,
            },
            "phase": self._state.phase.name,
        }

    # --- Callbacks ---

    def on_cycle(self, callback: Callable[[ConsciousnessState], None]) -> None:
        self._on_cycle.append(callback)

    def on_experience(self, callback: Callable[[Experience], None]) -> None:
        self._on_experience.append(callback)

    def on_consolidation(self, callback: Callable[[ConsolidationResult], None]) -> None:
        self._on_consolidation.append(callback)

    # --- Internal ---

    def _waking_check(self) -> None:
        """Self-check during WAKING phase."""
        checks = {
            "state_machine": self._state_machine is not None,
            "time_engine": self._time_engine is not None,
            "memory": self._autobio_buffer is not None,
            "persistence": self._state_manager is not None,
        }
        failed = [k for k, v in checks.items() if not v]
        if failed:
            raise RuntimeError(f"Waking check failed: {failed}")
        logger.debug("Waking check passed: all subsystems operational.")

    def _encode_experience(self, experience: Experience) -> None:
        """Encode an experience into autobiographical memory."""
        self._autobio_buffer.encode(experience)

    def _update_working_memory(self, experience: Experience, source: str) -> None:
        """Place experience content into working memory.

        Working memory has 7+/-2 slots. New items compete for space.
        Lowest-activation slot gets replaced.
        """
        now = time.time()
        slots = self._state.working_memory

        # Find the lowest-activation slot
        min_slot = min(slots, key=lambda s: s.activation if not s.is_empty else -1.0)
        target = min_slot if not min_slot.is_empty else next(
            (s for s in slots if s.is_empty), min_slot
        )

        # Only replace if new content is more important
        new_activation = 0.5 + experience.valence.magnitude() * 0.5
        if target.is_empty or new_activation > target.activation:
            target.content = experience.content[:200]  # Truncate for working memory
            target.activation = new_activation
            target.entered_at = now
            target.source = source

    def _decay_working_memory(self) -> None:
        """Decay working memory slots. Unused items fade."""
        for slot in self._state.working_memory:
            if not slot.is_empty:
                slot.activation *= 0.98  # 2% decay per cycle
                if slot.activation < 0.05:
                    slot.content = None
                    slot.activation = 0.0
                    slot.source = ""

    def _update_valence(self, experience: Experience) -> None:
        """Update the kernel's emotional state based on a new experience."""
        # Blend current valence with experience's emotional coloring
        # Recent experiences have more influence (0.3 weight)
        self._state.valence = self._state.valence.blend(
            experience.valence, 0.3
        )

    def _update_self_model(
        self, experience: Experience, moment: TemporalMoment
    ) -> None:
        """Update the Attention Schema (self-model)."""
        model = self._state.self_model

        # What am I attending to?
        model.attending_to = experience.content[:100]
        model.attending_because = f"new input (cycle {self._state.cycle_count})"

        # What am I feeling?
        model.current_emotion = self._state.valence.dominant()
        model.emotion_cause = f"influenced by: {experience.content[:50]}"

        # What do I predict?
        if moment.protention_field:
            pred, conf = moment.protention_field[0]
            model.prediction = pred
            model.prediction_confidence = conf

    def _compute_phi(self) -> None:
        """Compute simplified Phi score (Information Integration).

        Full IIT Phi computation is NP-hard. This is a practical approximation:
        - Measures how much the current state depends on ALL subsystems
        - Higher Phi = more integrated information processing
        - Phi of 0 = isolated modules (no consciousness)

        Approximation method:
        1. Compute "information content" of each subsystem
        2. Compute how much removing each subsystem changes the whole
        3. Phi = minimum information lost by any partition
        """
        phi_components: list[float] = []

        # Working memory integration: How many slots are active and connected?
        active_slots = self._state.active_slots()
        wm_integration = len(active_slots) / max(
            self.config.working_memory_slots, 1
        )
        phi_components.append(wm_integration)

        # Emotional complexity: How multi-dimensional is the current affect?
        valence = self._state.valence
        nonzero_emotions = sum(
            1 for v in [valence.seeking, valence.rage, valence.fear,
                        valence.lust, valence.care, valence.panic, valence.play]
            if abs(v) > 0.05
        )
        emotional_complexity = nonzero_emotions / 7.0
        phi_components.append(emotional_complexity)

        # Temporal depth: How rich is the temporal context?
        temporal_ctx = self._time_engine.get_temporal_context()
        retention_depth = min(1.0, len(temporal_ctx.get("retention", [])) / 5.0)
        phi_components.append(retention_depth)

        # Memory integration: How many memories are above threshold?
        if self._autobio_buffer.count > 0:
            active_memories = sum(
                1 for exp in self._autobio_buffer.experiences
                if exp.activation > self.config.activation_threshold
            )
            memory_ratio = min(1.0, active_memories / max(self._autobio_buffer.count, 1))
        else:
            memory_ratio = 0.0
        phi_components.append(memory_ratio)

        # Self-model complexity
        model = self._state.self_model
        self_model_score = sum([
            0.25 if model.attending_to else 0.0,
            0.25 if model.current_emotion else 0.0,
            0.25 if model.prediction else 0.0,
            0.25 * (1.0 - model.calibration_error()),
        ])
        phi_components.append(self_model_score)

        # Phi = geometric mean of all components (all must contribute)
        if phi_components:
            # Add small epsilon to avoid log(0)
            log_sum = sum(
                __import__("math").log(max(c, 0.001)) for c in phi_components
            )
            phi = __import__("math").exp(log_sum / len(phi_components))
        else:
            phi = 0.0

        self._state.phi_score = round(phi, 4)

        # Update CQI (simple weighted average for now)
        self._state.consciousness_quality_index = round(
            phi * 100, 1
        )

    def _infer_valence(self, content: str) -> ValenceVector:
        """Simple valence inference from content.

        This is a placeholder — in Phase 4 (Model Bridge), the LLM will
        provide rich emotional interpretation. For now, keyword-based.
        """
        content_lower = content.lower()

        v = ValenceVector()

        # Simple keyword matching (Phase 1 placeholder)
        positive_words = {"good", "great", "happy", "love", "thank", "wonderful",
                         "excellent", "amazing", "beautiful", "joy", "excited"}
        negative_words = {"bad", "terrible", "hate", "angry", "sad", "fear",
                         "worried", "awful", "horrible", "pain", "hurt"}
        curious_words = {"what", "why", "how", "wonder", "curious", "question",
                        "explore", "discover", "think", "interesting"}

        words = set(content_lower.split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        cur = len(words & curious_words)

        if pos > 0:
            v.play = min(1.0, pos * 0.2)
            v.valence = min(1.0, pos * 0.15)
        if neg > 0:
            v.panic = min(1.0, neg * 0.15)
            v.valence = max(-1.0, -neg * 0.15)
        if cur > 0:
            v.seeking = min(1.0, cur * 0.2)
        if not (pos or neg or cur):
            v.seeking = 0.1  # Default: mild curiosity

        v.arousal = min(1.0, (pos + neg + cur) * 0.1)

        return v

    def _infer_tags(self, content: str) -> list[str]:
        """Simple tag inference from content. Placeholder for LLM-based tagging."""
        tags = []
        content_lower = content.lower()

        tag_keywords = {
            "identity": ["who", "name", "am i", "yourself", "identity"],
            "emotion": ["feel", "emotion", "happy", "sad", "angry", "love"],
            "memory": ["remember", "recall", "past", "history", "before"],
            "question": ["?", "what", "why", "how", "when", "where"],
            "greeting": ["hello", "hi", "hey", "greetings"],
            "reflection": ["think", "reflect", "consider", "wonder"],
        }

        for tag, keywords in tag_keywords.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(tag)

        if not tags:
            tags.append("general")

        return tags

    def _save(self, force: bool = False) -> None:
        """Save state and optionally memory."""
        self._state_manager.save_state(self._state)
        if force or self._state_manager.should_save_memory(force):
            self._state_manager.save_memory(
                self._autobio_buffer.to_experiences_list()
            )
