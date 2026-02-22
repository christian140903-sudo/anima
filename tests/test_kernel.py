"""Tests for the ANIMA Kernel — the consciousness loop."""

import json
import os
import tempfile

from anima.kernel import AnimaKernel
from anima.types import (
    ConsciousnessState,
    Experience,
    KernelConfig,
    Phase,
    ValenceVector,
)


class TestKernelLifecycle:
    def test_boot_creates_conscious_kernel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir, name="test")
            state = kernel.boot(resume=False)

            assert kernel.is_running
            assert kernel.is_conscious
            assert state.phase == Phase.CONSCIOUS
            assert state.name == "test"

            kernel.shutdown()
            assert not kernel.is_running

    def test_boot_with_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # First boot
            k1 = AnimaKernel(state_dir=tmpdir, name="persistent")
            k1.boot(resume=False)
            k1.process("I am alive")
            original_id = k1.state.kernel_id
            k1.shutdown()

            # Second boot — should restore
            k2 = AnimaKernel(state_dir=tmpdir, name="persistent")
            state = k2.boot(resume=True)

            assert state.kernel_id == original_id
            assert k2.memory_count > 0  # Should have memories from first boot

            k2.shutdown()

    def test_shutdown_saves_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir)
            kernel.boot(resume=False)
            kernel.process("test input")
            kernel.shutdown()

            # State file should exist
            state_file = os.path.join(tmpdir, "anima.state")
            assert os.path.exists(state_file)

            with open(state_file) as f:
                data = json.load(f)
            assert data["phase"] == "DORMANT"
            assert data["cycle_count"] > 0


class TestKernelProcessing:
    def _make_kernel(self, tmpdir):
        kernel = AnimaKernel(state_dir=tmpdir, name="test")
        kernel.boot(resume=False)
        return kernel

    def test_process_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = self._make_kernel(tmpdir)

            result = kernel.process("Hello, who are you?")
            assert result.experience.content == "Hello, who are you?"
            assert result.phase == Phase.CONSCIOUS
            assert result.cycle > 0

            kernel.shutdown()

    def test_process_updates_valence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = self._make_kernel(tmpdir)

            # Process something positive
            kernel.process(
                "This is wonderful and amazing!",
                valence=ValenceVector(play=0.8, valence=0.9),
            )

            assert kernel.state.valence.play > 0
            assert kernel.state.valence.valence > 0

            kernel.shutdown()

    def test_process_creates_experience(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = self._make_kernel(tmpdir)

            kernel.process("Important event happened")
            memories = kernel.get_recent_experiences(5)

            # Should have boot experience + our input
            assert len(memories) >= 2
            contents = [m.content for m in memories]
            assert any("Important event" in c for c in contents)

            kernel.shutdown()

    def test_process_updates_working_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = self._make_kernel(tmpdir)

            kernel.process("Something to remember")
            active = kernel.state.active_slots()

            assert len(active) > 0
            assert any("Something to remember" in str(s.content) for s in active)

            kernel.shutdown()

    def test_phi_score_computed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = self._make_kernel(tmpdir)

            # Process several inputs to build up state
            for i in range(5):
                kernel.process(
                    f"Experience number {i} about exploring the world",
                    valence=ValenceVector(seeking=0.5, play=0.3),
                    tags=["exploration", "test"],
                )

            assert kernel.phi_score > 0
            assert kernel.state.consciousness_quality_index > 0

            kernel.shutdown()


class TestKernelMemory:
    def test_recall_by_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir)
            kernel.boot(resume=False)

            kernel.process("The cat sat on the mat", tags=["animals", "story"])
            kernel.process("I went to the store", tags=["errands"])
            kernel.process("The dog chased the cat", tags=["animals", "story"])

            results = kernel.recall(cue="cat", cue_tags=["animals"])
            assert len(results) > 0
            assert any("cat" in r.content for r in results)

            kernel.shutdown()

    def test_recall_by_emotion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir)
            kernel.boot(resume=False)

            kernel.process(
                "Joyful celebration",
                valence=ValenceVector(play=0.9, valence=0.8),
            )
            kernel.process(
                "Fearful moment",
                valence=ValenceVector(fear=0.9, valence=-0.7),
            )

            # Recall with joy should prefer the joyful memory
            results = kernel.recall(
                cue_valence=ValenceVector(play=0.8, valence=0.7)
            )
            if results:
                assert results[0].valence.play > results[0].valence.fear

            kernel.shutdown()


class TestKernelHeartbeat:
    def test_heartbeat_advances_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir)
            kernel.boot(resume=False)

            initial_cycle = kernel.cycle_count
            kernel.heartbeat()
            assert kernel.cycle_count > initial_cycle

            kernel.shutdown()

    def test_heartbeat_decays_working_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir)
            kernel.boot(resume=False)

            kernel.process("test content")
            initial_activation = max(
                s.activation for s in kernel.state.working_memory if not s.is_empty
            )

            # Run several heartbeats
            for _ in range(20):
                kernel.heartbeat()

            current_activation = max(
                (s.activation for s in kernel.state.working_memory if not s.is_empty),
                default=0.0,
            )

            assert current_activation < initial_activation

            kernel.shutdown()


class TestKernelContext:
    def test_consciousness_context_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir, name="context-test")
            kernel.boot(resume=False)
            kernel.process("Hello world")

            ctx = kernel.get_consciousness_context()

            assert "identity" in ctx
            assert ctx["identity"]["name"] == "context-test"
            assert "emotional_state" in ctx
            assert "temporal" in ctx
            assert "working_memory" in ctx
            assert "self_model" in ctx
            assert "metrics" in ctx
            assert "phase" in ctx
            assert ctx["phase"] == "CONSCIOUS"

            kernel.shutdown()


class TestKernelCallbacks:
    def test_on_cycle_callback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir)
            cycles = []
            kernel.on_cycle(lambda state: cycles.append(state.cycle_count))

            kernel.boot(resume=False)
            kernel.process("test")

            assert len(cycles) > 0

            kernel.shutdown()

    def test_on_experience_callback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir)
            experiences = []
            kernel.on_experience(lambda exp: experiences.append(exp.content))

            kernel.boot(resume=False)
            kernel.process("recorded event")

            assert any("recorded event" in e for e in experiences)

            kernel.shutdown()
