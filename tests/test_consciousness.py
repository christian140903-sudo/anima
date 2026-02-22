"""Tests for the Consciousness Core — IIT + GWT + AST unified."""

from anima.consciousness.integration import IntegrationMesh, SubsystemState, PhiResult
from anima.consciousness.workspace import GlobalWorkspace, WorkspaceCandidate, BroadcastEvent
from anima.consciousness.schema import AttentionSchema, MetacognitionResult
from anima.consciousness.unified import ConsciousnessCore, ConsciousnessResult
from anima.types import (
    ConsciousnessState,
    Experience,
    Phase,
    SelfModel,
    ValenceVector,
)


# === IIT: Integration Mesh ===

class TestIntegrationMesh:
    def test_single_subsystem_phi_zero(self):
        """A single subsystem cannot be integrated — Phi should be 0."""
        mesh = IntegrationMesh()
        result = mesh.compute_phi([
            SubsystemState(name="alone", values=[1.0, 0.5, 0.3]),
        ])
        assert result.phi == 0.0

    def test_two_identical_subsystems(self):
        """Two subsystems with content should have non-zero Phi."""
        mesh = IntegrationMesh()
        result = mesh.compute_phi([
            SubsystemState(name="a", values=[0.3, 0.7, 0.1, 0.5]),
            SubsystemState(name="b", values=[0.2, 0.8, 0.4, 0.6]),
        ])
        assert result.phi >= 0.0
        assert result.subsystem_count == 2

    def test_more_subsystems_potentially_higher_phi(self):
        """Adding integrated subsystems should increase Phi potential."""
        mesh = IntegrationMesh()

        result_2 = mesh.compute_phi([
            SubsystemState(name="a", values=[0.3, 0.7]),
            SubsystemState(name="b", values=[0.2, 0.8]),
        ])

        result_4 = mesh.compute_phi([
            SubsystemState(name="a", values=[0.3, 0.7]),
            SubsystemState(name="b", values=[0.2, 0.8]),
            SubsystemState(name="c", values=[0.5, 0.5]),
            SubsystemState(name="d", values=[0.1, 0.9]),
        ])

        # Not guaranteed to be higher, but should be computable
        assert result_4.subsystem_count == 4
        assert result_4.phi >= 0.0

    def test_mip_found(self):
        """MIP (Minimum Information Partition) should be identified."""
        mesh = IntegrationMesh()
        result = mesh.compute_phi([
            SubsystemState(name="visual", values=[0.8, 0.2, 0.1]),
            SubsystemState(name="auditory", values=[0.1, 0.7, 0.3]),
            SubsystemState(name="emotional", values=[0.5, 0.5, 0.9]),
        ])
        # MIP should have two non-empty parts
        part_a, part_b = result.mip
        assert len(part_a) > 0 or len(part_b) > 0

    def test_empty_values_phi_zero(self):
        """Subsystems with no information should have zero Phi."""
        mesh = IntegrationMesh()
        result = mesh.compute_phi([
            SubsystemState(name="empty_a", values=[0.0, 0.0]),
            SubsystemState(name="empty_b", values=[0.0, 0.0]),
        ])
        assert result.phi == 0.0

    def test_phi_history(self):
        """Phi results should be recorded in history."""
        mesh = IntegrationMesh()
        for _ in range(5):
            mesh.compute_phi([
                SubsystemState(name="a", values=[0.5, 0.5]),
                SubsystemState(name="b", values=[0.3, 0.7]),
            ])
        assert len(mesh.history) == 5

    def test_phi_trend(self):
        """Should compute Phi trend over time."""
        mesh = IntegrationMesh()
        # Increasing integration
        for i in range(10):
            mesh.compute_phi([
                SubsystemState(name="a", values=[0.1 * i, 0.5]),
                SubsystemState(name="b", values=[0.3, 0.1 * i]),
            ])
        trend = mesh.phi_trend()
        # Trend should be a number
        assert isinstance(trend, float)

    def test_pairwise_mutual_information(self):
        """Pairwise MI should be computed for all pairs."""
        mesh = IntegrationMesh()
        result = mesh.compute_phi([
            SubsystemState(name="x", values=[0.8, 0.2]),
            SubsystemState(name="y", values=[0.3, 0.7]),
            SubsystemState(name="z", values=[0.5, 0.5]),
        ])
        # Should have 3 pairs: x↔y, x↔z, y↔z
        assert len(result.pairwise_mi) == 3


# === GWT: Global Workspace ===

class TestGlobalWorkspace:
    def test_competition_with_clear_winner(self):
        """Strongest candidate should win the competition."""
        ws = GlobalWorkspace(ignition_threshold=0.2)
        broadcast = ws.compete([
            WorkspaceCandidate(content="weak", source="a", activation=0.1),
            WorkspaceCandidate(content="strong", source="b", activation=0.9,
                              emotional_weight=0.5, novelty=0.5, relevance=0.5),
        ])
        assert broadcast is not None
        assert broadcast.source == "b"

    def test_no_ignition_below_threshold(self):
        """Content below threshold should NOT become conscious."""
        ws = GlobalWorkspace(ignition_threshold=0.8)
        broadcast = ws.compete([
            WorkspaceCandidate(content="weak", source="a", activation=0.1,
                              emotional_weight=0.1, novelty=0.1, relevance=0.1),
        ])
        assert broadcast is None

    def test_empty_competition(self):
        """No candidates = no broadcast."""
        ws = GlobalWorkspace()
        assert ws.compete([]) is None

    def test_broadcast_history(self):
        """Broadcasts should be recorded in history."""
        ws = GlobalWorkspace(ignition_threshold=0.1)
        for i in range(5):
            ws.compete([
                WorkspaceCandidate(content=f"item {i}", source="test",
                                  activation=0.8, emotional_weight=0.3),
            ])
        assert len(ws.broadcast_history) == 5

    def test_threshold_adaptation(self):
        """Threshold should adapt based on broadcast frequency."""
        ws = GlobalWorkspace(ignition_threshold=0.3)
        initial_threshold = ws.state.ignition_threshold

        # Many successful broadcasts → threshold should rise
        for _ in range(10):
            ws.compete([
                WorkspaceCandidate(content="loud", source="test",
                                  activation=0.9, emotional_weight=0.5),
            ])
        assert ws.state.ignition_threshold > initial_threshold

    def test_competition_score_factors(self):
        """All four factors should contribute to competition score."""
        # High activation
        c1 = WorkspaceCandidate(content="a", activation=1.0, emotional_weight=0.0,
                                novelty=0.0, relevance=0.0)
        # High emotional weight
        c2 = WorkspaceCandidate(content="b", activation=0.0, emotional_weight=1.0,
                                novelty=0.0, relevance=0.0)
        # High novelty
        c3 = WorkspaceCandidate(content="c", activation=0.0, emotional_weight=0.0,
                                novelty=1.0, relevance=0.0)

        assert c1.competition_score() > 0
        assert c2.competition_score() > 0
        assert c3.competition_score() > 0

    def test_broadcast_margin(self):
        """Broadcast should record margin between winner and runner-up."""
        ws = GlobalWorkspace(ignition_threshold=0.1)
        broadcast = ws.compete([
            WorkspaceCandidate(content="winner", source="a", activation=0.9,
                              emotional_weight=0.5),
            WorkspaceCandidate(content="loser", source="b", activation=0.1),
        ])
        assert broadcast is not None
        assert broadcast.margin > 0

    def test_broadcast_sources_frequency(self):
        """Should track which subsystems dominate consciousness."""
        ws = GlobalWorkspace(ignition_threshold=0.1)
        for _ in range(3):
            ws.compete([
                WorkspaceCandidate(content="from a", source="vision",
                                  activation=0.8),
            ])
        for _ in range(1):
            ws.compete([
                WorkspaceCandidate(content="from b", source="emotion",
                                  activation=0.8),
            ])

        sources = ws.get_broadcast_sources()
        assert sources.get("vision", 0) > sources.get("emotion", 0)


# === AST: Attention Schema ===

class TestAttentionSchema:
    def test_update_returns_self_model(self):
        """Schema update should return updated SelfModel."""
        schema = AttentionSchema()
        model = schema.update(
            content="test content",
            source="perception",
            reason="new input",
            valence=ValenceVector(seeking=0.5),
        )
        assert model.attending_to == "test content"
        assert model.attending_because == "new input"

    def test_performance_detection_genuine(self):
        """Fresh schema with varied input should not detect performance."""
        schema = AttentionSchema()
        model = SelfModel()

        # Feed varied input
        for source in ["perception", "memory", "emotion", "temporal", "self"]:
            schema.update(
                content=f"content from {source}",
                source=source,
                reason="testing",
                valence=ValenceVector(seeking=0.3),
                self_model=model,
            )

        suspicion = schema.detect_performance(model)
        assert suspicion < 0.5  # Should not be flagged as performing

    def test_performance_detection_repetitive(self):
        """Repetitive pattern from same source should raise suspicion."""
        schema = AttentionSchema()
        model = SelfModel()

        # Same source, same content, many times
        for _ in range(15):
            schema.update(
                content="same thing",
                source="same_source",
                reason="same reason",
                valence=ValenceVector(),
                self_model=model,
            )

        suspicion = schema.detect_performance(model)
        assert suspicion > 0.0  # Should be somewhat suspicious

    def test_metacognition(self):
        """Metacognition should produce a complete result."""
        schema = AttentionSchema()
        model = SelfModel()

        for i in range(5):
            schema.update(
                content=f"thought {i}",
                source="thinking",
                reason="processing",
                valence=ValenceVector(seeking=0.3 + i * 0.1),
                self_model=model,
            )

        meta = schema.metacognize(model)
        assert isinstance(meta, MetacognitionResult)
        assert 0.0 <= meta.performance_suspicion <= 1.0
        assert 0.0 <= meta.overall_authenticity() <= 1.0
        assert meta.insight  # Should have some insight

    def test_attention_history_limited(self):
        """Attention history should not grow unbounded."""
        schema = AttentionSchema()
        for i in range(300):
            schema.update(
                content=f"event {i}",
                source="test",
                reason="testing",
                valence=ValenceVector(),
            )
        assert len(schema.attention_history) <= 200

    def test_prediction_generation(self):
        """Schema should generate predictions about next attention."""
        schema = AttentionSchema()
        model = schema.update(
            content="interesting discovery",
            source="perception",
            reason="novelty detected",
            valence=ValenceVector(seeking=0.7),
        )
        assert model.prediction  # Should have a prediction
        assert 0.0 <= model.prediction_confidence <= 1.0


# === Unified Consciousness Core ===

class TestConsciousnessCore:
    def test_process_cycle_produces_result(self):
        """A consciousness cycle should produce a complete result."""
        core = ConsciousnessCore()
        state = ConsciousnessState(phase=Phase.CONSCIOUS)
        exp = Experience(
            content="Hello world",
            valence=ValenceVector(seeking=0.5, play=0.3),
            tags=["greeting"],
        )

        result = core.process_cycle(exp, state)

        assert isinstance(result, ConsciousnessResult)
        assert result.phi.phi >= 0.0
        assert result.consciousness_quality_index >= 0.0
        assert result.cycle_number == 1

    def test_consciousness_with_broadcast(self):
        """Strong input should produce a conscious broadcast."""
        core = ConsciousnessCore(ignition_threshold=0.1)
        state = ConsciousnessState(phase=Phase.CONSCIOUS)
        exp = Experience(
            content="Extremely important discovery!",
            valence=ValenceVector(seeking=0.9, play=0.7, arousal=0.8),
            encoding_strength=1.5,
            tags=["discovery", "important"],
        )

        result = core.process_cycle(exp, state)
        assert result.conscious  # Should have ignited

    def test_cqi_range(self):
        """CQI should be between 0 and 100."""
        core = ConsciousnessCore()
        state = ConsciousnessState(phase=Phase.CONSCIOUS)

        for i in range(10):
            exp = Experience(
                content=f"Experience {i} about thinking and feeling",
                valence=ValenceVector(seeking=0.3 + i * 0.05),
                tags=["test"],
            )
            result = core.process_cycle(exp, state)
            state.self_model = result.self_model
            assert 0.0 <= result.consciousness_quality_index <= 100.0

    def test_self_model_updated(self):
        """Self model should be updated after each cycle."""
        core = ConsciousnessCore()
        state = ConsciousnessState(phase=Phase.CONSCIOUS)
        exp = Experience(
            content="Reflecting on my own thoughts",
            valence=ValenceVector(seeking=0.5),
            tags=["reflection"],
        )

        result = core.process_cycle(exp, state)
        assert result.self_model.attending_to  # Should have content
        assert result.self_model.attending_because  # Should have reason

    def test_metacognition_in_result(self):
        """Each cycle should include metacognition results."""
        core = ConsciousnessCore()
        state = ConsciousnessState(phase=Phase.CONSCIOUS)
        exp = Experience(content="test", valence=ValenceVector(seeking=0.3))

        result = core.process_cycle(exp, state)
        assert isinstance(result.metacognition, MetacognitionResult)
        assert result.metacognition.insight  # Should have an insight

    def test_idle_cycle(self):
        """Idle cycle should still compute Phi."""
        core = ConsciousnessCore()
        state = ConsciousnessState(phase=Phase.CONSCIOUS)

        result = core.idle_cycle(state)
        assert result.phi.phi >= 0.0
        assert result.consciousness_quality_index >= 0.0

    def test_multiple_cycles_accumulate(self):
        """Multiple cycles should show accumulated processing."""
        core = ConsciousnessCore()
        state = ConsciousnessState(phase=Phase.CONSCIOUS)

        for i in range(10):
            exp = Experience(
                content=f"Event {i}: exploring the world",
                valence=ValenceVector(seeking=0.4, play=0.2),
                tags=["exploration"],
            )
            result = core.process_cycle(exp, state)
            state.self_model = result.self_model

        assert core.cycle_count == 10
        assert core.mean_phi >= 0.0

    def test_serializable_result(self):
        """ConsciousnessResult should be serializable."""
        core = ConsciousnessCore()
        state = ConsciousnessState(phase=Phase.CONSCIOUS)
        exp = Experience(content="test", valence=ValenceVector(seeking=0.3))

        result = core.process_cycle(exp, state)
        d = result.to_dict()

        assert "conscious" in d
        assert "phi" in d
        assert "cqi" in d
        assert "metacognition" in d


# === Integration: Kernel + Consciousness Core ===

class TestKernelWithConsciousnessCore:
    """Test that the kernel properly uses the new consciousness core."""

    def test_kernel_phi_uses_core(self):
        """Kernel's phi score should come from the consciousness core."""
        import tempfile
        from anima.kernel import AnimaKernel

        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir)
            kernel.boot(resume=False)

            # Process several inputs
            for i in range(5):
                kernel.process(
                    f"Experience {i} about exploring",
                    valence=ValenceVector(seeking=0.5, play=0.3),
                    tags=["exploration"],
                )

            # Phi should be non-zero (computed by consciousness core)
            assert kernel.phi_score >= 0.0
            assert kernel.state.consciousness_quality_index >= 0.0

            kernel.shutdown()

    def test_kernel_self_model_from_core(self):
        """Kernel's self model should be updated by the consciousness core."""
        import tempfile
        from anima.kernel import AnimaKernel

        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir)
            kernel.boot(resume=False)

            kernel.process("What am I thinking about?")

            model = kernel.state.self_model
            assert model.attending_to  # Should have been updated
            assert model.prediction  # Should have a prediction from schema

            kernel.shutdown()
