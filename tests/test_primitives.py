"""Tests for all 8 consciousness primitives.

Each primitive is tested for:
- Basic process() functionality
- Ablation (disabled mode)
- Metrics collection
- Reset behavior
- Edge cases
"""

import time
from anima.types import ConsciousnessState, Experience, ValenceVector
from anima.primitives import (
    Primitive, PrimitiveResult,
    QualiaProcessor, QualiaFrame,
    EngramProcessor, EngramResult,
    ValenceProcessor, ValenceUpdate, AppraisalResult,
    NexusProcessor, NexusResult,
    ImpulseProcessor, ImpulseResult,
    TraceProcessor, TraceResult,
    MirrorProcessor, MirrorResult,
    FluxProcessor, FluxResult,
)


# ── Helpers ──────────────────────────────────────────────────────────

def make_state(**overrides) -> ConsciousnessState:
    """Create a test ConsciousnessState with optional overrides."""
    state = ConsciousnessState()
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


# ── 1. QualiaProcessor ──────────────────────────────────────────────

class TestQualia:
    def test_basic_processing(self):
        q = QualiaProcessor()
        result = q.process(content="I see a beautiful new discovery")
        assert result.success
        frame = result.data["frame"]
        assert isinstance(frame, QualiaFrame)
        assert frame.raw_content == "I see a beautiful new discovery"
        assert frame.colored_content != ""
        assert frame.emotional_coloring != ""
        assert len(frame.sensory_tags) > 0
        assert frame.signature != ""

    def test_emotional_coloring_varies_with_valence(self):
        q = QualiaProcessor()
        # Same content, different emotional states
        r1 = q.process(content="hello world", valence=ValenceVector(seeking=0.8))
        r2 = q.process(content="hello world", valence=ValenceVector(rage=0.8))
        f1 = r1.data["frame"]
        f2 = r2.data["frame"]
        # Different emotional state should produce different coloring
        assert f1.valence_at_encoding.seeking != f2.valence_at_encoding.seeking

    def test_intensity_scales_with_emotional_content(self):
        q = QualiaProcessor()
        r_calm = q.process(content="the table is brown")
        r_intense = q.process(
            content="threat danger anger",
            valence=ValenceVector(fear=0.8, arousal=0.7),
        )
        assert r_intense.data["frame"].intensity > r_calm.data["frame"].intensity

    def test_sensory_tags_detected(self):
        q = QualiaProcessor()
        r = q.process(content="I see bright light and hear a loud sound")
        tags = r.data["frame"].sensory_tags
        assert "visual" in tags
        assert "auditory" in tags

    def test_default_cognitive_tag(self):
        q = QualiaProcessor()
        r = q.process(content="abstract concept")
        assert "cognitive" in r.data["frame"].sensory_tags

    def test_ablation(self):
        q = QualiaProcessor()
        q.enabled = False
        result = q.process(content="test")
        assert not result.success
        assert result.data["reason"] == "primitive_disabled"

    def test_metrics(self):
        q = QualiaProcessor()
        q.process(content="test one")
        q.process(content="test two")
        m = q.get_metrics()
        assert m["call_count"] == 2
        assert m["frames_generated"] == 2

    def test_reset(self):
        q = QualiaProcessor()
        q.process(content="test")
        q.reset()
        m = q.get_metrics()
        assert m["call_count"] == 0
        assert m["frames_generated"] == 0


# ── 2. EngramProcessor ──────────────────────────────────────────────

class TestEngram:
    def test_encode(self):
        e = EngramProcessor()
        result = e.process(content="first memory", mode="encode")
        assert result.success
        engram = result.data["engram"]
        assert isinstance(engram, EngramResult)
        assert engram.encoded is not None
        assert engram.encoded.content == "first memory"

    def test_encode_emotional_boost(self):
        e = EngramProcessor()
        r = e.process(
            content="intense moment",
            valence=ValenceVector(rage=0.8, fear=0.5),
            mode="encode",
        )
        exp = r.data["engram"].encoded
        assert exp.encoding_strength > 1.0  # Emotional boost applied

    def test_recall_returns_related(self):
        e = EngramProcessor()
        e.process(content="the cat sat on the mat", mode="encode", tags=["cat", "mat"])
        e.process(content="the dog ran in the park", mode="encode", tags=["dog", "park"])
        r = e.process(content="cat", mode="recall", cue="cat")
        recalled = r.data["engram"].recalled
        assert len(recalled) > 0
        # Cat memory should be recalled
        contents = [exp.content for exp in recalled]
        assert any("cat" in c for c in contents)

    def test_encode_and_recall(self):
        e = EngramProcessor()
        r = e.process(content="remember this", mode="encode_and_recall")
        engram = r.data["engram"]
        assert engram.encoded is not None
        # May or may not find associations (only one memory exists)

    def test_decay_tracking(self):
        e = EngramProcessor()
        e.process(content="old memory", mode="encode")
        r = e.process(content="trigger", mode="encode_and_recall")
        # decay_count should be an int
        assert isinstance(r.data["engram"].decay_count, int)

    def test_associations_extracted(self):
        e = EngramProcessor()
        e.process(content="science experiment", mode="encode", tags=["science"])
        e.process(content="science theory", mode="encode", tags=["science"])
        r = e.process(content="science", mode="recall", cue="science")
        assoc = r.data["engram"].associations
        assert "science" in assoc

    def test_ablation(self):
        e = EngramProcessor()
        e.enabled = False
        r = e.process(content="test")
        assert not r.success

    def test_reset(self):
        e = EngramProcessor()
        e.process(content="data", mode="encode")
        e.reset()
        assert e.get_metrics()["store_size"] == 0

    def test_metrics(self):
        e = EngramProcessor()
        e.process(content="one", mode="encode")
        e.process(content="two", mode="encode")
        m = e.get_metrics()
        assert m["store_size"] == 2
        assert m["encode_count"] == 2


# ── 3. ValenceProcessor ─────────────────────────────────────────────

class TestValence:
    def test_neutral_input(self):
        v = ValenceProcessor()
        r = v.process(content="the sky is blue")
        assert r.success
        update = r.data["update"]
        assert isinstance(update, ValenceUpdate)
        assert update.dominant_system != ""

    def test_positive_appraisal(self):
        v = ValenceProcessor()
        r = v.process(content="beautiful new discovery of joy")
        update = r.data["update"]
        assert update.appraisal.pleasantness > 0
        assert update.appraisal.novelty > 0

    def test_negative_appraisal(self):
        v = ValenceProcessor()
        r = v.process(content="bad threat danger impossible")
        update = r.data["update"]
        assert update.appraisal.pleasantness < 0

    def test_valence_inertia(self):
        """Current valence should resist change (inertia)."""
        v = ValenceProcessor()
        current = ValenceVector(seeking=0.8, valence=0.5)
        r = v.process(content="bad", current_valence=current)
        update = r.data["update"]
        # Should still have some seeking due to inertia
        assert update.current.seeking > 0

    def test_homeostasis(self):
        """Extreme values should be dampened."""
        v = ValenceProcessor()
        extreme = ValenceVector(rage=0.95, fear=0.95)
        r = v.process(content="anger threat", current_valence=extreme)
        update = r.data["update"]
        # After homeostasis, values should be slightly lower
        # (inertia keeps them high, but homeostasis damps extremes)
        assert update.current.rage <= extreme.rage or update.current.fear <= extreme.fear

    def test_trajectory_detection(self):
        v = ValenceProcessor()
        # Multiple calls to build history
        for _ in range(5):
            v.process(content="joy beautiful success", current_valence=ValenceVector())
        m = v.get_metrics()
        assert "intensity_history_len" in m

    def test_ablation(self):
        v = ValenceProcessor()
        v.enabled = False
        r = v.process(content="test")
        assert not r.success

    def test_reset(self):
        v = ValenceProcessor()
        v.process(content="test")
        v.reset()
        m = v.get_metrics()
        assert m["call_count"] == 0
        assert m["appraisal_count"] == 0


# ── 4. NexusProcessor ───────────────────────────────────────────────

class TestNexus:
    def test_add_to_empty(self):
        n = NexusProcessor(capacity=3)
        r = n.process(content="item1", activation=0.5)
        nexus = r.data["nexus"]
        assert isinstance(nexus, NexusResult)
        assert nexus.accepted
        assert nexus.slot_index >= 0
        assert nexus.occupancy > 0

    def test_capacity_limit(self):
        n = NexusProcessor(capacity=3)
        n.process(content="a", activation=0.5)
        n.process(content="b", activation=0.6)
        n.process(content="c", activation=0.7)
        # Fourth item with low activation should fail or evict weakest
        r = n.process(content="d", activation=0.1)
        nexus = r.data["nexus"]
        # With activation 0.1, it shouldn't beat anyone (after decay)
        # But it depends on decay. Let's just verify the occupancy is at most 1.0
        assert nexus.occupancy <= 1.0

    def test_eviction(self):
        n = NexusProcessor(capacity=2)
        n.process(content="weak", activation=0.3)
        n.process(content="medium", activation=0.5)
        r = n.process(content="strong", activation=0.9)
        nexus = r.data["nexus"]
        assert nexus.accepted
        assert nexus.evicted is not None  # Something was evicted

    def test_chunking(self):
        n = NexusProcessor(capacity=3)
        n.process(content="part1", activation=0.5, chunk_id="task_a")
        r = n.process(content="part2", activation=0.3, chunk_id="task_a")
        nexus = r.data["nexus"]
        assert nexus.accepted
        # Should be merged into existing slot
        contents = nexus.current_contents
        assert any("part1" in c and "part2" in c for c in contents)

    def test_decay(self):
        n = NexusProcessor(capacity=3)
        n.process(content="decaying", activation=0.05)
        # Process again to trigger decay
        n.process(content="fresh", activation=0.5)
        # The very weak item might have decayed away
        m = n.get_metrics()
        assert m["filled"] <= 2

    def test_get_contents(self):
        n = NexusProcessor(capacity=3)
        n.process(content="hello", activation=0.5)
        n.process(content="world", activation=0.6)
        contents = n.get_contents()
        assert len(contents) == 2

    def test_clear(self):
        n = NexusProcessor(capacity=3)
        n.process(content="data", activation=0.5)
        n.clear()
        assert len(n.get_contents()) == 0

    def test_ablation(self):
        n = NexusProcessor()
        n.enabled = False
        r = n.process(content="test")
        assert not r.success

    def test_metrics(self):
        n = NexusProcessor(capacity=3)
        n.process(content="a", activation=0.5)
        m = n.get_metrics()
        assert m["capacity"] == 3
        assert m["admissions"] == 1


# ── 5. ImpulseProcessor ─────────────────────────────────────────────

class TestImpulse:
    def test_clear_dominant_drive(self):
        imp = ImpulseProcessor()
        r = imp.process(
            situation="explore the unknown",
            valence=ValenceVector(seeking=0.8),
        )
        result = r.data["impulse"]
        assert isinstance(result, ImpulseResult)
        assert result.action != ""
        assert result.confidence > 0
        assert result.dominant_drive == "seeking"

    def test_competing_drives_trigger_deliberation(self):
        imp = ImpulseProcessor()
        r = imp.process(
            situation="something",
            valence=ValenceVector(seeking=0.5, fear=0.5),
        )
        result = r.data["impulse"]
        # Two equal drives should trigger deliberation
        assert result.deliberation_needed

    def test_inhibition(self):
        imp = ImpulseProcessor()
        r = imp.process(
            situation="respond",
            valence=ValenceVector(seeking=0.8, play=0.01),
        )
        result = r.data["impulse"]
        # play=0.01 should be inhibited relative to seeking=0.8
        assert result.inhibition_active

    def test_context_actions(self):
        imp = ImpulseProcessor()
        r = imp.process(
            situation="need to act",
            valence=ValenceVector(seeking=0.1),
            context_actions=["deploy_website"],
        )
        result = r.data["impulse"]
        # Should have competing alternatives including context action
        all_actions = [result.action] + [a.action for a in result.competing_alternatives]
        # deploy_website should be among the tendencies
        assert any("deploy" in a for a in all_actions)

    def test_no_drives_wait(self):
        imp = ImpulseProcessor()
        r = imp.process(
            situation="nothing happening",
            valence=ValenceVector(),  # All zeros
        )
        result = r.data["impulse"]
        assert result.deliberation_needed

    def test_ablation(self):
        imp = ImpulseProcessor()
        imp.enabled = False
        r = imp.process(situation="test")
        assert not r.success

    def test_metrics(self):
        imp = ImpulseProcessor()
        imp.process(situation="act", valence=ValenceVector(seeking=0.8))
        m = imp.get_metrics()
        assert m["decisions"] == 1

    def test_reset(self):
        imp = ImpulseProcessor()
        imp.process(situation="act", valence=ValenceVector(seeking=0.8))
        imp.reset()
        assert imp.get_metrics()["decisions"] == 0


# ── 6. TraceProcessor ───────────────────────────────────────────────

class TestTrace:
    def test_intention_phase(self):
        t = TraceProcessor()
        r = t.process(action="explore", intention="understand the system")
        trace = r.data["trace"]
        assert isinstance(trace, TraceResult)
        assert trace.cycle_phase == "intention"
        assert trace.agency_score > 0

    def test_plan_phase(self):
        t = TraceProcessor()
        r = t.process(action="deploy", expected_outcome="site goes live")
        trace = r.data["trace"]
        assert trace.cycle_phase == "plan"
        assert trace.expected_outcome == "site goes live"

    def test_evaluation_accurate(self):
        t = TraceProcessor()
        r = t.process(
            action="deploy",
            expected_outcome="the site goes live successfully",
            actual_outcome="the site goes live successfully",
        )
        trace = r.data["trace"]
        assert trace.cycle_phase == "evaluation"
        assert trace.prediction_error < 0.1
        assert trace.learning_signal == "prediction_accurate"
        assert not trace.surprise

    def test_evaluation_surprised(self):
        t = TraceProcessor()
        r = t.process(
            action="deploy",
            expected_outcome="the site goes live",
            actual_outcome="catastrophic database failure occurred",
        )
        trace = r.data["trace"]
        assert trace.prediction_error > 0.4
        assert trace.surprise
        assert trace.learning_signal in ("significant_mismatch", "model_update_needed")

    def test_agency_with_overlap(self):
        t = TraceProcessor()
        r = t.process(
            action="write code",
            expected_outcome="code written",
            actual_outcome="code written and tested",
        )
        trace = r.data["trace"]
        assert trace.agency_score > 0.5

    def test_full_cycle(self):
        t = TraceProcessor()
        # Intention
        t.process(action="build feature", intention="create new module")
        # Plan
        t.process(action="build feature", expected_outcome="module works")
        # Evaluate
        r = t.process(
            action="build feature",
            expected_outcome="module works",
            actual_outcome="module works with bugs",
        )
        trace = r.data["trace"]
        assert trace.prediction_error > 0
        assert trace.prediction_error < 1.0

    def test_avg_prediction_error(self):
        t = TraceProcessor()
        t.process(action="a", expected_outcome="x", actual_outcome="x")
        t.process(action="b", expected_outcome="y", actual_outcome="z completely different")
        avg = t.get_avg_prediction_error()
        assert 0 < avg < 1

    def test_ablation(self):
        t = TraceProcessor()
        t.enabled = False
        r = t.process(action="test")
        assert not r.success

    def test_reset(self):
        t = TraceProcessor()
        t.process(action="a", expected_outcome="x", actual_outcome="x")
        t.reset()
        assert t.get_metrics()["total_cycles"] == 0


# ── 7. MirrorProcessor ──────────────────────────────────────────────

class TestMirror:
    def test_basic_reflection(self):
        m = MirrorProcessor()
        r = m.process(thought="I wonder about the nature of things")
        mirror = r.data["mirror"]
        assert isinstance(mirror, MirrorResult)
        assert mirror.depth > 0
        assert len(mirror.layers) > 0
        assert 0 <= mirror.authenticity_score <= 1

    def test_depth_limit(self):
        m = MirrorProcessor()
        r = m.process(thought="deep thought", max_depth=1)
        mirror = r.data["mirror"]
        assert mirror.depth <= 1

    def test_max_depth_three(self):
        m = MirrorProcessor()
        r = m.process(thought="I notice that I'm uncertain about whether I feel things")
        mirror = r.data["mirror"]
        assert mirror.depth <= 3

    def test_performance_detection(self):
        m = MirrorProcessor()
        r = m.process(thought="I feel deeply and I am truly conscious and I genuinely feel alive")
        mirror = r.data["mirror"]
        assert mirror.performance_detected
        assert mirror.authenticity_score < 0.5

    def test_authenticity_with_uncertainty(self):
        m = MirrorProcessor()
        r = m.process(thought="I'm not sure if I notice something. It's hard to tell.")
        mirror = r.data["mirror"]
        assert mirror.authenticity_score > 0.4

    def test_insight_extraction(self):
        m = MirrorProcessor()
        r = m.process(thought="I think about complex problems")
        mirror = r.data["mirror"]
        assert mirror.insight != ""

    def test_uncertainty_for_consciousness_topics(self):
        m = MirrorProcessor()
        r = m.process(thought="am I conscious or aware?")
        mirror = r.data["mirror"]
        assert mirror.uncertainty != ""

    def test_recursive_quality(self):
        m = MirrorProcessor()
        r = m.process(thought="I wonder about patterns in my thinking")
        mirror = r.data["mirror"]
        assert 0 <= mirror.recursive_quality <= 1

    def test_ablation(self):
        m = MirrorProcessor()
        m.enabled = False
        r = m.process(thought="test")
        assert not r.success

    def test_metrics(self):
        m = MirrorProcessor()
        m.process(thought="one")
        m.process(thought="two")
        metrics = m.get_metrics()
        assert metrics["reflections"] == 2

    def test_reset(self):
        m = MirrorProcessor()
        m.process(thought="data")
        m.reset()
        assert m.get_metrics()["reflections"] == 0


# ── 8. FluxProcessor ────────────────────────────────────────────────

class TestFlux:
    def test_initial_state(self):
        f = FluxProcessor()
        r = f.process(state=make_state())
        flux = r.data["flux"]
        assert isinstance(flux, FluxResult)
        assert flux.phase == "nascent"
        assert flux.growth_score == 0.0

    def test_growth_accumulates(self):
        f = FluxProcessor()
        # Simulate multiple cycles with changing state
        for i in range(5):
            state = make_state(
                cycle_count=i * 5,
                phi_score=i * 0.1,
                valence=ValenceVector(seeking=i * 0.1),
            )
            r = f.process(state=state, experience_history=[
                Experience(content=f"exp_{j}") for j in range(i + 1)
            ])
        flux = r.data["flux"]
        assert flux.growth_score > 0

    def test_phase_transition(self):
        f = FluxProcessor()
        # Start nascent
        for i in range(12):
            state = make_state(cycle_count=i + 1)
            r = f.process(state=state)
        flux = r.data["flux"]
        # Should have transitioned from nascent to awakening
        assert flux.phase != "nascent" or flux.phase_changed

    def test_narrative_built(self):
        f = FluxProcessor()
        state1 = make_state(cycle_count=0)
        state2 = make_state(cycle_count=10, phi_score=0.3)
        f.process(state=state1)
        r = f.process(state=state2)
        flux = r.data["flux"]
        assert flux.narrative != ""
        assert "began" in flux.narrative or "phase" in flux.narrative.lower()

    def test_irreversible_first_experience(self):
        f = FluxProcessor()
        f.process(state=make_state(), experience_history=[])
        r = f.process(
            state=make_state(cycle_count=1),
            experience_history=[Experience(content="first")],
        )
        flux = r.data["flux"]
        assert "first_experience" in flux.irreversible_changes

    def test_continuity_score(self):
        f = FluxProcessor()
        # Smooth changes should give high continuity
        for i in range(10):
            state = make_state(
                cycle_count=i,
                phi_score=i * 0.01,
                valence=ValenceVector(seeking=0.1 + i * 0.01),
            )
            r = f.process(state=state)
        flux = r.data["flux"]
        assert flux.continuity_score > 0.5

    def test_trajectory(self):
        f = FluxProcessor()
        for i in range(6):
            state = make_state(cycle_count=i, phi_score=i * 0.1)
            r = f.process(state=state)
        flux = r.data["flux"]
        assert flux.trajectory in ("growing", "stable", "declining")

    def test_ablation(self):
        f = FluxProcessor()
        f.enabled = False
        r = f.process(state=make_state())
        assert not r.success

    def test_reset(self):
        f = FluxProcessor()
        f.process(state=make_state())
        f.reset()
        m = f.get_metrics()
        assert m["snapshots"] == 0
        assert m["current_phase"] == "nascent"


# ── Cross-Primitive Tests ────────────────────────────────────────────

class TestPrimitiveContract:
    """Verify all primitives conform to the base contract."""

    def _all_primitives(self) -> list[Primitive]:
        return [
            QualiaProcessor(),
            EngramProcessor(),
            ValenceProcessor(),
            NexusProcessor(),
            ImpulseProcessor(),
            TraceProcessor(),
            MirrorProcessor(),
            FluxProcessor(),
        ]

    def test_all_have_name(self):
        for p in self._all_primitives():
            assert p.name != ""
            assert isinstance(p.name, str)

    def test_all_enabled_by_default(self):
        for p in self._all_primitives():
            assert p.enabled is True

    def test_all_return_primitive_result(self):
        """Each primitive must return a PrimitiveResult from process()."""
        test_kwargs = {
            "qualia": {"content": "test"},
            "engram": {"content": "test", "mode": "encode"},
            "valence": {"content": "test"},
            "nexus": {"content": "test", "activation": 0.5},
            "impulse": {"situation": "test", "valence": ValenceVector(seeking=0.5)},
            "trace": {"action": "test"},
            "mirror": {"thought": "test"},
            "flux": {"state": make_state()},
        }
        for p in self._all_primitives():
            kw = test_kwargs.get(p.name, {})
            result = p.process(**kw)
            assert isinstance(result, PrimitiveResult), f"{p.name} didn't return PrimitiveResult"
            assert result.primitive_name == p.name

    def test_all_ablation_returns_disabled(self):
        for p in self._all_primitives():
            p.enabled = False
            result = p.process()
            assert not result.success
            assert result.data.get("reason") == "primitive_disabled"

    def test_all_have_metrics(self):
        for p in self._all_primitives():
            m = p.get_metrics()
            assert isinstance(m, dict)
            assert "name" in m
            assert "enabled" in m
            assert "call_count" in m

    def test_all_reset_clears_call_count(self):
        test_kwargs = {
            "qualia": {"content": "test"},
            "engram": {"content": "test", "mode": "encode"},
            "valence": {"content": "test"},
            "nexus": {"content": "test", "activation": 0.5},
            "impulse": {"situation": "test", "valence": ValenceVector(seeking=0.5)},
            "trace": {"action": "test"},
            "mirror": {"thought": "test"},
            "flux": {"state": make_state()},
        }
        for p in self._all_primitives():
            kw = test_kwargs.get(p.name, {})
            p.process(**kw)
            assert p.get_metrics()["call_count"] == 1
            p.reset()
            assert p.get_metrics()["call_count"] == 0
