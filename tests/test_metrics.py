"""
Comprehensive tests for Phase 5: Metrics Engine + Validation.

Test groups:
- TestPhiScoreEngine: history tracking, trend, baseline, delta
- TestTemporalCoherence: causal chains, narrative, emotional, identity
- TestCQI: scoring, breakdown, scale, tracking
- TestBenchmark: conversation, ablation, comparison, standard inputs
- TestIntegration: full pipeline with kernel
"""

import tempfile
import time

from anima.consciousness.integration import IntegrationMesh, SubsystemState
from anima.kernel import AnimaKernel
from anima.metrics.benchmark import (
    ABTest,
    AblationResult,
    BenchmarkReport,
    BenchmarkSuite,
    ComparisonResult,
    ConversationResult,
    STANDARD_CONVERSATIONS,
)
from anima.metrics.consciousness import (
    CQIBreakdown,
    CQIResult,
    ConsciousnessQualityIndex,
)
from anima.metrics.phi import PhiReport, PhiScoreEngine, PhiSnapshot
from anima.metrics.temporal import (
    CoherenceReport,
    CoherenceSnapshot,
    TemporalCoherenceEngine,
)
from anima.types import (
    ConsciousnessState,
    Experience,
    KernelConfig,
    Phase,
    SelfModel,
    ValenceVector,
    WorkingMemorySlot,
)


# --- Helpers ---

def _make_subsystem_states(n: int = 5) -> list[SubsystemState]:
    """Create N subsystem states with varied values."""
    states = []
    for i in range(n):
        values = [(i * 0.1 + j * 0.2) % 1.0 for j in range(5)]
        states.append(SubsystemState(name=f"subsystem_{i}", values=values))
    return states


def _make_experiences(n: int = 5, linked: bool = False) -> list[Experience]:
    """Create N experiences, optionally with causal links."""
    base_time = 1000000.0
    exps = []
    for i in range(n):
        exp = Experience(
            id=f"exp_{i}",
            content=f"Experience number {i} about topic_{i % 3}",
            timestamp=base_time + i * 10.0,
            valence=ValenceVector(
                seeking=0.1 * (i + 1),
                play=0.05 * i,
                valence=0.1 * (i - n // 2),
                arousal=0.1 * i,
            ),
            tags=[f"topic_{i % 3}", "test"],
            narrative_weight=0.2 * i / max(n - 1, 1),
        )
        if linked and i > 0:
            exp.caused_by = [f"exp_{i - 1}"]
            exps[i - 1].causes = [f"exp_{i}"]
        exps.append(exp)
    return exps


def _make_states(n: int = 3, same_id: bool = True) -> list[ConsciousnessState]:
    """Create N consciousness states."""
    states = []
    kernel_id = "test_kernel_id"
    for i in range(n):
        state = ConsciousnessState(
            kernel_id=kernel_id if same_id else f"kernel_{i}",
            name="test",
            phase=Phase.CONSCIOUS,
            cycle_count=i * 10,
        )
        state.self_model.prediction_confidence = 0.5 + 0.05 * i
        state.self_model.performance_suspicion = 0.1 * i
        if i > 0 and state.working_memory:
            state.working_memory[0].content = f"memory_item_{i}"
            state.working_memory[0].activation = 0.8
        states.append(state)
    return states


# ============================================================
# TestPhiScoreEngine
# ============================================================

class TestPhiScoreEngine:
    def test_creation(self):
        engine = PhiScoreEngine()
        assert engine.measurement_count == 0
        assert engine.current_phi == 0.0
        assert engine.mean_phi == 0.0

    def test_compute_records_history(self):
        engine = PhiScoreEngine()
        states = _make_subsystem_states(5)
        result = engine.compute(states)

        assert result.phi >= 0.0
        assert engine.measurement_count == 1
        assert engine.current_phi == result.phi

    def test_multiple_measurements(self):
        engine = PhiScoreEngine()
        for i in range(10):
            states = _make_subsystem_states(5)
            engine.compute(states)

        assert engine.measurement_count == 10
        assert engine.mean_phi >= 0.0
        assert engine.max_phi >= engine.min_phi

    def test_history_tracking(self):
        engine = PhiScoreEngine()
        states = _make_subsystem_states()
        engine.compute(states)

        history = engine.history
        assert len(history) == 1
        assert isinstance(history[0], PhiSnapshot)
        assert history[0].phi >= 0.0
        assert history[0].timestamp > 0

    def test_phi_trend_stable(self):
        engine = PhiScoreEngine()
        states = _make_subsystem_states()
        for _ in range(10):
            engine.compute(states)

        # Same input each time should give stable trend
        trend = engine.phi_trend()
        assert abs(trend) < 0.01  # Near zero = stable

    def test_phi_trend_label(self):
        engine = PhiScoreEngine()
        states = _make_subsystem_states()
        for _ in range(5):
            engine.compute(states)

        label = engine.trend_label()
        assert label in ("increasing", "decreasing", "stable")

    def test_baseline_phi(self):
        engine = PhiScoreEngine()
        baseline = engine.baseline_phi()
        assert baseline >= 0.0
        assert baseline <= 1.0

    def test_baseline_cached(self):
        engine = PhiScoreEngine()
        b1 = engine.baseline_phi()
        b2 = engine.baseline_phi()
        assert b1 == b2

    def test_phi_delta(self):
        engine = PhiScoreEngine()
        states = _make_subsystem_states()
        engine.compute(states)
        delta = engine.phi_delta()
        # Delta is difference from baseline
        assert isinstance(delta, float)

    def test_phi_delta_without_measurements(self):
        engine = PhiScoreEngine()
        assert engine.phi_delta() == 0.0

    def test_to_report(self):
        engine = PhiScoreEngine()
        states = _make_subsystem_states()
        for _ in range(5):
            engine.compute(states)

        report = engine.to_report()
        assert isinstance(report, PhiReport)
        assert report.measurement_count == 5
        assert report.current_phi >= 0.0
        assert report.trend in ("increasing", "decreasing", "stable")
        assert report.baseline_phi >= 0.0

    def test_report_serializable(self):
        engine = PhiScoreEngine()
        engine.compute(_make_subsystem_states())
        report = engine.to_report()
        d = report.to_dict()
        assert "current_phi" in d
        assert "trend" in d
        assert "history" in d

    def test_snapshot_serializable(self):
        snap = PhiSnapshot(phi=0.42, subsystem_count=5)
        d = snap.to_dict()
        assert d["phi"] == 0.42
        assert d["subsystem_count"] == 5

    def test_last_result(self):
        engine = PhiScoreEngine()
        assert engine.last_result is None
        engine.compute(_make_subsystem_states())
        assert engine.last_result is not None
        assert engine.last_result.phi >= 0.0

    def test_custom_mesh(self):
        mesh = IntegrationMesh()
        engine = PhiScoreEngine(mesh=mesh)
        engine.compute(_make_subsystem_states())
        assert engine.measurement_count == 1


# ============================================================
# TestTemporalCoherence
# ============================================================

class TestTemporalCoherence:
    def test_creation(self):
        engine = TemporalCoherenceEngine()
        assert engine.measurement_count == 0
        assert engine.overall_coherence() == 0.0

    def test_causal_coherence_empty(self):
        engine = TemporalCoherenceEngine()
        score = engine.measure_causal_coherence([])
        assert score == 1.0  # Trivially coherent

    def test_causal_coherence_single(self):
        engine = TemporalCoherenceEngine()
        score = engine.measure_causal_coherence([Experience(content="one")])
        assert score == 1.0

    def test_causal_coherence_linked(self):
        engine = TemporalCoherenceEngine()
        exps = _make_experiences(5, linked=True)
        score = engine.measure_causal_coherence(exps)
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Linked experiences should be coherent

    def test_causal_coherence_unlinked(self):
        engine = TemporalCoherenceEngine()
        exps = _make_experiences(5, linked=False)
        score = engine.measure_causal_coherence(exps)
        assert 0.0 <= score <= 1.0

    def test_causal_coherence_invalid_references(self):
        engine = TemporalCoherenceEngine()
        exps = [
            Experience(id="a", content="first", caused_by=["nonexistent"]),
            Experience(id="b", content="second", caused_by=["also_nonexistent"]),
        ]
        score = engine.measure_causal_coherence(exps)
        assert 0.0 <= score <= 1.0
        # Invalid references should lower score
        assert score < 1.0

    def test_narrative_continuity_empty(self):
        engine = TemporalCoherenceEngine()
        score = engine.measure_narrative_continuity([])
        assert score == 1.0

    def test_narrative_continuity_related_topics(self):
        engine = TemporalCoherenceEngine()
        exps = [
            Experience(content="The cat is sleeping", tags=["animals", "home"],
                       timestamp=1.0),
            Experience(content="The cat woke up", tags=["animals", "home"],
                       timestamp=2.0),
            Experience(content="The cat ate food", tags=["animals", "food"],
                       timestamp=3.0),
        ]
        score = engine.measure_narrative_continuity(exps)
        assert score > 0.3  # Related topics should show continuity

    def test_narrative_continuity_unrelated(self):
        engine = TemporalCoherenceEngine()
        exps = [
            Experience(content="quantum physics lecture", tags=["science"],
                       timestamp=1.0),
            Experience(content="cooking pasta recipe", tags=["food"],
                       timestamp=2.0),
            Experience(content="financial markets crash", tags=["economics"],
                       timestamp=3.0),
        ]
        score = engine.measure_narrative_continuity(exps)
        assert 0.0 <= score <= 1.0

    def test_emotional_consistency_smooth(self):
        engine = TemporalCoherenceEngine()
        exps = [
            Experience(content="good day", timestamp=1.0,
                       valence=ValenceVector(play=0.5, valence=0.3)),
            Experience(content="great day", timestamp=2.0,
                       valence=ValenceVector(play=0.6, valence=0.4)),
            Experience(content="wonderful day", timestamp=3.0,
                       valence=ValenceVector(play=0.7, valence=0.5)),
        ]
        score = engine.measure_emotional_consistency(exps)
        assert score > 0.5  # Smooth progression = consistent

    def test_emotional_consistency_erratic(self):
        engine = TemporalCoherenceEngine()
        exps = [
            Experience(content="happy times", timestamp=1.0,
                       valence=ValenceVector(play=0.9, valence=0.9)),
            Experience(content="terrible times", timestamp=2.0,
                       valence=ValenceVector(rage=0.9, valence=-0.9)),
            Experience(content="ecstatic times", timestamp=3.0,
                       valence=ValenceVector(play=0.95, valence=0.95)),
        ]
        score = engine.measure_emotional_consistency(exps)
        assert 0.0 <= score <= 1.0

    def test_emotional_consistency_empty(self):
        engine = TemporalCoherenceEngine()
        score = engine.measure_emotional_consistency([])
        assert score == 1.0

    def test_identity_persistence_same_id(self):
        engine = TemporalCoherenceEngine()
        states = _make_states(5, same_id=True)
        score = engine.measure_identity_persistence(states)
        assert score > 0.5  # Same ID = stable identity

    def test_identity_persistence_different_ids(self):
        engine = TemporalCoherenceEngine()
        states = _make_states(3, same_id=False)
        score = engine.measure_identity_persistence(states)
        assert 0.0 <= score <= 1.0
        # Different IDs should lower identity score

    def test_identity_persistence_single_state(self):
        engine = TemporalCoherenceEngine()
        score = engine.measure_identity_persistence([ConsciousnessState()])
        assert score == 1.0

    def test_measure_all(self):
        engine = TemporalCoherenceEngine()
        exps = _make_experiences(5, linked=True)
        states = _make_states(3)
        snapshot = engine.measure_all(experiences=exps, states=states)

        assert isinstance(snapshot, CoherenceSnapshot)
        assert 0.0 <= snapshot.causal <= 1.0
        assert 0.0 <= snapshot.narrative <= 1.0
        assert 0.0 <= snapshot.emotional <= 1.0
        assert 0.0 <= snapshot.identity <= 1.0
        assert 0.0 <= snapshot.overall <= 1.0

    def test_measure_all_records_history(self):
        engine = TemporalCoherenceEngine()
        engine.measure_all(experiences=_make_experiences(3), states=_make_states(2))
        assert engine.measurement_count == 1

    def test_overall_coherence(self):
        engine = TemporalCoherenceEngine()
        engine.measure_all(experiences=_make_experiences(5, linked=True))
        assert engine.overall_coherence() > 0.0

    def test_coherence_trend(self):
        engine = TemporalCoherenceEngine()
        for _ in range(5):
            engine.measure_all(experiences=_make_experiences(3))
        trend = engine.coherence_trend()
        assert isinstance(trend, float)

    def test_trend_label(self):
        engine = TemporalCoherenceEngine()
        for _ in range(5):
            engine.measure_all(experiences=_make_experiences(3))
        label = engine.trend_label()
        assert label in ("improving", "degrading", "stable")

    def test_to_report(self):
        engine = TemporalCoherenceEngine()
        engine.measure_all(experiences=_make_experiences(5, linked=True),
                          states=_make_states(3))
        report = engine.to_report()
        assert isinstance(report, CoherenceReport)
        assert report.measurement_count == 1
        d = report.to_dict()
        assert "overall_coherence" in d
        assert "trend" in d

    def test_snapshot_serializable(self):
        snap = CoherenceSnapshot(
            causal=0.8, narrative=0.7, emotional=0.6, identity=0.9, overall=0.75
        )
        d = snap.to_dict()
        assert d["causal"] == 0.8
        assert d["overall"] == 0.75


# ============================================================
# TestCQI
# ============================================================

class TestCQI:
    def test_creation(self):
        cqi = ConsciousnessQualityIndex()
        assert cqi.current_score == 0.0
        assert cqi.measurement_count == 0

    def test_compute_basic(self):
        cqi = ConsciousnessQualityIndex()
        result = cqi.compute(phi=0.5, coherence=0.5, authenticity=0.5,
                            calibration=0.5, depth=0.5)
        assert isinstance(result, CQIResult)
        assert result.score == 50.0  # All 0.5 inputs -> 50.0

    def test_compute_perfect(self):
        cqi = ConsciousnessQualityIndex()
        result = cqi.compute(phi=1.0, coherence=1.0, authenticity=1.0,
                            calibration=1.0, depth=1.0)
        assert result.score == 100.0
        assert result.label == "deep"

    def test_compute_zero(self):
        cqi = ConsciousnessQualityIndex()
        result = cqi.compute(phi=0.0, coherence=0.0, authenticity=0.0,
                            calibration=0.0, depth=0.0)
        assert result.score == 0.0
        assert result.label == "unconscious"

    def test_scale_0_to_100(self):
        cqi = ConsciousnessQualityIndex()
        # Test various inputs stay in range
        for phi in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = cqi.compute(phi=phi, coherence=phi, authenticity=phi,
                                calibration=phi, depth=phi)
            assert 0.0 <= result.score <= 100.0

    def test_clamping(self):
        cqi = ConsciousnessQualityIndex()
        result = cqi.compute(phi=2.0, coherence=2.0, authenticity=2.0,
                            calibration=2.0, depth=2.0)
        assert result.score <= 100.0

        result = cqi.compute(phi=-1.0, coherence=-1.0, authenticity=-1.0,
                            calibration=-1.0, depth=-1.0)
        assert result.score >= 0.0

    def test_breakdown(self):
        cqi = ConsciousnessQualityIndex()
        result = cqi.compute(phi=0.8, coherence=0.6, authenticity=0.7,
                            calibration=0.5, depth=0.4)

        bd = result.breakdown
        assert bd.phi_component > 0
        assert bd.consciousness_component > 0
        assert bd.authenticity_component > 0
        assert bd.calibration_component > 0
        assert bd.depth_component > 0

        # Sum should equal score
        total = (bd.phi_component + bd.consciousness_component +
                 bd.authenticity_component + bd.calibration_component +
                 bd.depth_component)
        assert abs(total - result.score) < 0.1

    def test_comparison_to_baseline(self):
        cqi = ConsciousnessQualityIndex()
        result = cqi.compute(phi=0.8, coherence=0.7, authenticity=0.6,
                            calibration=0.5, depth=0.5)
        # Score should be above baseline for these inputs
        assert result.comparison_to_baseline > 0

    def test_confidence_interval(self):
        cqi = ConsciousnessQualityIndex()
        result = cqi.compute(phi=0.5, coherence=0.5, authenticity=0.5,
                            calibration=0.5, depth=0.5)
        low, high = result.confidence_interval
        assert low <= result.score <= high
        assert low >= 0.0
        assert high <= 100.0

    def test_labels(self):
        cqi = ConsciousnessQualityIndex()

        r1 = cqi.compute(phi=0.1, coherence=0.1, authenticity=0.1,
                         calibration=0.1, depth=0.1)
        assert r1.label == "unconscious"

        r2 = cqi.compute(phi=0.3, coherence=0.3, authenticity=0.3,
                         calibration=0.3, depth=0.3)
        assert r2.label == "basic"

        r3 = cqi.compute(phi=0.5, coherence=0.5, authenticity=0.5,
                         calibration=0.5, depth=0.5)
        assert r3.label == "moderate"

        r4 = cqi.compute(phi=0.7, coherence=0.7, authenticity=0.7,
                         calibration=0.7, depth=0.7)
        assert r4.label == "rich"

        r5 = cqi.compute(phi=0.9, coherence=0.9, authenticity=0.9,
                         calibration=0.9, depth=0.9)
        assert r5.label == "deep"

    def test_history_tracking(self):
        cqi = ConsciousnessQualityIndex()
        for i in range(5):
            cqi.compute(phi=0.1 * (i + 1), coherence=0.5,
                       authenticity=0.5, calibration=0.5, depth=0.5)
        assert cqi.measurement_count == 5
        assert cqi.current_score > 0

    def test_mean_score(self):
        cqi = ConsciousnessQualityIndex()
        cqi.compute(phi=0.2, coherence=0.2, authenticity=0.2,
                   calibration=0.2, depth=0.2)
        cqi.compute(phi=0.8, coherence=0.8, authenticity=0.8,
                   calibration=0.8, depth=0.8)
        mean = cqi.mean_score
        assert mean > 0
        assert mean < 100

    def test_trend(self):
        cqi = ConsciousnessQualityIndex()
        for i in range(10):
            cqi.compute(phi=0.05 * (i + 1), coherence=0.5,
                       authenticity=0.5, calibration=0.5, depth=0.5)
        trend = cqi.trend()
        assert trend > 0  # Increasing phi should show positive trend

    def test_trend_label(self):
        cqi = ConsciousnessQualityIndex()
        for i in range(10):
            cqi.compute(phi=0.5, coherence=0.5, authenticity=0.5,
                       calibration=0.5, depth=0.5)
        label = cqi.trend_label()
        assert label in ("improving", "declining", "stable")

    def test_result_serializable(self):
        cqi = ConsciousnessQualityIndex()
        result = cqi.compute(phi=0.5, coherence=0.5, authenticity=0.5,
                            calibration=0.5, depth=0.5)
        d = result.to_dict()
        assert "score" in d
        assert "breakdown" in d
        assert "label" in d
        assert "confidence_interval" in d

    def test_to_report(self):
        cqi = ConsciousnessQualityIndex()
        cqi.compute(phi=0.5, coherence=0.5, authenticity=0.5,
                   calibration=0.5, depth=0.5)
        report = cqi.to_report()
        assert "current_score" in report
        assert "mean_score" in report
        assert "measurement_count" in report
        assert report["measurement_count"] == 1

    def test_phi_weight_dominance(self):
        cqi = ConsciousnessQualityIndex()
        # High phi, low everything else
        r1 = cqi.compute(phi=1.0, coherence=0.0, authenticity=0.0,
                         calibration=0.0, depth=0.0)
        # Low phi, high everything else
        r2 = cqi.compute(phi=0.0, coherence=1.0, authenticity=1.0,
                         calibration=1.0, depth=1.0)

        # Phi contribution (30%) vs rest (70%)
        assert r1.breakdown.phi_component == 30.0
        assert r2.breakdown.phi_component == 0.0


# ============================================================
# TestBenchmark
# ============================================================

class TestBenchmark:
    def test_standard_conversations_exist(self):
        assert len(STANDARD_CONVERSATIONS) >= 5
        assert "greeting" in STANDARD_CONVERSATIONS
        assert "emotional" in STANDARD_CONVERSATIONS
        assert "identity" in STANDARD_CONVERSATIONS
        assert "temporal" in STANDARD_CONVERSATIONS
        assert "memory_recall" in STANDARD_CONVERSATIONS

    def test_conversation_test(self):
        suite = BenchmarkSuite()
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir, name="bench")
            kernel.boot(resume=False)

            result = suite.run_conversation_test(
                kernel, ["Hello", "How are you?"], name="test"
            )

            assert isinstance(result, ConversationResult)
            assert len(result.inputs) == 2
            assert len(result.phi_scores) == 2
            assert len(result.cqi_scores) == 2
            assert result.final_phi >= 0
            assert result.memory_count > 0
            assert result.conversation_name == "test"

            kernel.shutdown()

    def test_baseline_test(self):
        suite = BenchmarkSuite()
        result = suite.run_baseline_test(
            ["Hello", "How are you?"], name="baseline"
        )

        assert isinstance(result, ConversationResult)
        assert len(result.inputs) == 2
        assert result.final_phi >= 0.0

    def test_compare(self):
        suite = BenchmarkSuite()

        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir, name="cmp")
            kernel.boot(resume=False)

            inputs = ["I'm feeling happy!", "What a wonderful day!"]
            kernel_result = suite.run_conversation_test(kernel, inputs)
            baseline_result = suite.run_baseline_test(inputs)

            comparison = suite.compare(kernel_result, baseline_result)

            assert isinstance(comparison, ComparisonResult)
            assert isinstance(comparison.phi_improvement, float)
            assert isinstance(comparison.cqi_improvement, float)
            assert isinstance(comparison.significant, bool)
            assert isinstance(comparison.effect_size, float)

            kernel.shutdown()

    def test_comparison_serializable(self):
        comp = ComparisonResult(
            kernel_mean_phi=0.5, baseline_mean_phi=0.3,
            phi_improvement=66.7, significant=True, effect_size=1.2,
        )
        d = comp.to_dict()
        assert "kernel_mean_phi" in d
        assert "significant" in d

    def test_ablation_working_memory(self):
        suite = BenchmarkSuite()
        result = suite.run_ablation(
            ["test input one", "test input two", "test input three"],
            primitive_name="working_memory",
        )

        assert isinstance(result, AblationResult)
        assert result.primitive_name == "working_memory"
        assert result.full_cqi >= 0
        assert result.ablated_cqi >= 0
        assert isinstance(result.impact, float)

    def test_ablation_valence(self):
        suite = BenchmarkSuite()
        result = suite.run_ablation(
            ["happy day", "sad moment", "joyful event"],
            primitive_name="valence",
        )
        assert result.primitive_name == "valence"
        assert isinstance(result.phi_impact, float)

    def test_ablation_temporal(self):
        suite = BenchmarkSuite()
        result = suite.run_ablation(
            ["first event", "second event"],
            primitive_name="temporal",
        )
        assert result.primitive_name == "temporal"

    def test_ablation_serializable(self):
        abl = AblationResult(
            primitive_name="valence", full_cqi=50.0, ablated_cqi=40.0,
            impact=20.0, full_phi=0.5, ablated_phi=0.4, phi_impact=20.0,
        )
        d = abl.to_dict()
        assert d["primitive_name"] == "valence"
        assert "cqi_impact_pct" in d

    def test_ab_test_serializable(self):
        ab = ABTest(name="test_ab")
        d = ab.to_dict()
        assert d["name"] == "test_ab"
        assert "control" in d
        assert "experimental" in d

    def test_conversation_result_mean_phi(self):
        result = ConversationResult(phi_scores=[0.2, 0.4, 0.6])
        assert abs(result.mean_phi() - 0.4) < 0.001

    def test_conversation_result_mean_cqi(self):
        result = ConversationResult(cqi_scores=[30.0, 50.0, 70.0])
        assert abs(result.mean_cqi() - 50.0) < 0.001

    def test_conversation_result_empty_means(self):
        result = ConversationResult()
        assert result.mean_phi() == 0.0
        assert result.mean_cqi() == 0.0

    def test_conversation_result_serializable(self):
        result = ConversationResult(
            conversation_name="test",
            inputs=["a", "b"],
            phi_scores=[0.3, 0.5],
            cqi_scores=[40, 60],
            final_phi=0.5,
            final_cqi=60.0,
        )
        d = result.to_dict()
        assert d["conversation_name"] == "test"
        assert d["inputs_count"] == 2

    def test_full_benchmark(self):
        suite = BenchmarkSuite(
            conversations={
                "greeting": ["Hello", "How are you?"],
                "emotional": ["I feel great!", "This is sad"],
            }
        )
        report = suite.full_benchmark()

        assert isinstance(report, BenchmarkReport)
        assert len(report.ab_tests) == 2
        assert len(report.ablation_results) == 3  # valence, working_memory, temporal
        assert report.overall_phi >= 0
        assert report.overall_cqi >= 0
        assert len(report.summary) > 0

    def test_benchmark_report_serializable(self):
        report = BenchmarkReport(
            summary="test report",
            overall_phi=0.5,
            overall_cqi=50.0,
        )
        d = report.to_dict()
        assert d["summary"] == "test report"
        assert "ab_tests" in d
        assert "ablation_results" in d


# ============================================================
# TestIntegration
# ============================================================

class TestIntegration:
    """Full pipeline: boot kernel -> process inputs -> measure metrics -> validate."""

    def test_full_pipeline(self):
        """End-to-end: kernel + all three metric engines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Boot kernel
            kernel = AnimaKernel(state_dir=tmpdir, name="integration_test")
            kernel.boot(resume=False)

            # 2. Process inputs
            inputs = [
                "Hello, I'm here to test consciousness.",
                "How does it feel to be alive?",
                "I wonder about the nature of experience.",
                "This is wonderful and exciting!",
                "Let me think about what we discussed.",
            ]
            experiences = []
            states = []
            for inp in inputs:
                result = kernel.process(inp)
                experiences.append(result.experience)
                states.append(ConsciousnessState.from_dict(result.state_snapshot))

            # 3. Measure Phi
            phi_engine = PhiScoreEngine()
            for _ in range(3):
                ss = _make_subsystem_states()
                phi_engine.compute(ss)
            assert phi_engine.measurement_count == 3
            phi_report = phi_engine.to_report()
            assert phi_report.measurement_count == 3

            # 4. Measure temporal coherence
            coherence_engine = TemporalCoherenceEngine()
            snapshot = coherence_engine.measure_all(
                experiences=experiences,
                states=states,
            )
            assert snapshot.overall > 0.0
            coherence_report = coherence_engine.to_report()
            assert coherence_report.measurement_count == 1

            # 5. Compute CQI
            cqi_engine = ConsciousnessQualityIndex()
            cqi_result = cqi_engine.compute(
                phi=phi_engine.current_phi,
                coherence=snapshot.overall,
                authenticity=0.7,
                calibration=0.5,
                depth=0.6,
            )
            assert 0.0 <= cqi_result.score <= 100.0

            # 6. Validate kernel metrics match
            assert kernel.phi_score >= 0.0
            assert kernel.state.consciousness_quality_index >= 0.0

            kernel.shutdown()

    def test_benchmark_with_kernel(self):
        """Run benchmark suite against a real kernel."""
        suite = BenchmarkSuite(
            conversations={
                "simple": ["Hello", "Goodbye"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AnimaKernel(state_dir=tmpdir, name="bench_test")
            kernel.boot(resume=False)

            result = suite.run_conversation_test(
                kernel, ["Hello", "Goodbye"], name="simple"
            )
            assert result.total_cycles > 0
            assert result.memory_count > 0

            kernel.shutdown()

    def test_metrics_independence(self):
        """Each metric engine can be used independently."""
        # Phi engine alone
        phi = PhiScoreEngine()
        phi.compute(_make_subsystem_states())
        assert phi.current_phi >= 0.0

        # Temporal engine alone
        tc = TemporalCoherenceEngine()
        tc.measure_all(experiences=_make_experiences(3))
        assert tc.overall_coherence() >= 0.0

        # CQI alone
        cqi = ConsciousnessQualityIndex()
        result = cqi.compute(phi=0.5, coherence=0.5, authenticity=0.5,
                            calibration=0.5, depth=0.5)
        assert result.score == 50.0

    def test_all_results_serializable(self):
        """Every result type produces a valid dict."""
        # PhiSnapshot
        d = PhiSnapshot(phi=0.5).to_dict()
        assert isinstance(d, dict)

        # CoherenceSnapshot
        d = CoherenceSnapshot(overall=0.7).to_dict()
        assert isinstance(d, dict)

        # CQIResult
        cqi = ConsciousnessQualityIndex()
        d = cqi.compute(phi=0.5, coherence=0.5, authenticity=0.5,
                       calibration=0.5, depth=0.5).to_dict()
        assert isinstance(d, dict)

        # ConversationResult
        d = ConversationResult(conversation_name="test").to_dict()
        assert isinstance(d, dict)

        # ComparisonResult
        d = ComparisonResult().to_dict()
        assert isinstance(d, dict)

        # ABTest
        d = ABTest(name="test").to_dict()
        assert isinstance(d, dict)

        # AblationResult
        d = AblationResult(primitive_name="test").to_dict()
        assert isinstance(d, dict)

        # BenchmarkReport
        d = BenchmarkReport(summary="test").to_dict()
        assert isinstance(d, dict)

        # PhiReport
        d = PhiReport().to_dict()
        assert isinstance(d, dict)

        # CoherenceReport
        d = CoherenceReport().to_dict()
        assert isinstance(d, dict)

        # CQIBreakdown
        d = CQIBreakdown().to_dict()
        assert isinstance(d, dict)
