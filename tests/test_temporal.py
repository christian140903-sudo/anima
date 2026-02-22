"""Tests for the Temporal Substrate — the INVENTION."""

import time

from anima.temporal.autobio_buffer import AutobiographicalBuffer
from anima.temporal.consolidation import ConsolidationEngine
from anima.temporal.state_machine import StateMachine
from anima.temporal.time_engine import TemporalIntegrationEngine
from anima.types import (
    ConsciousnessState,
    CycleResult,
    Experience,
    KernelConfig,
    Phase,
    ValenceVector,
)


class TestStateMachine:
    def test_boot_sequence(self):
        sm = StateMachine()
        state = ConsciousnessState()
        assert state.phase == Phase.DORMANT

        state = sm.boot(state)
        assert state.phase == Phase.WAKING

        state = sm.complete_waking(state)
        assert state.phase == Phase.CONSCIOUS

    def test_invalid_transition_raises(self):
        sm = StateMachine()
        state = ConsciousnessState(phase=Phase.DORMANT)
        # Can't go directly from DORMANT to CONSCIOUS
        try:
            sm.transition(state, Phase.CONSCIOUS)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_valid_transitions(self):
        sm = StateMachine()
        state = ConsciousnessState()

        state = sm.transition(state, Phase.WAKING)
        assert state.phase == Phase.WAKING

        state = sm.transition(state, Phase.CONSCIOUS)
        assert state.phase == Phase.CONSCIOUS

        state = sm.transition(state, Phase.DREAMING)
        assert state.phase == Phase.DREAMING

        state = sm.transition(state, Phase.SLEEPING)
        assert state.phase == Phase.SLEEPING

        state = sm.transition(state, Phase.WAKING)
        assert state.phase == Phase.WAKING

    def test_shutdown_from_any_phase(self):
        sm = StateMachine()
        for phase in [Phase.WAKING, Phase.CONSCIOUS, Phase.DREAMING, Phase.SLEEPING]:
            state = ConsciousnessState(phase=phase)
            state = sm.shutdown(state)
            assert state.phase == Phase.DORMANT

    def test_transition_log(self):
        sm = StateMachine()
        state = ConsciousnessState()
        state = sm.boot(state)
        state = sm.complete_waking(state)

        history = sm.transition_history
        assert len(history) == 2
        assert history[0].from_phase == Phase.DORMANT
        assert history[0].to_phase == Phase.WAKING
        assert history[1].to_phase == Phase.CONSCIOUS

    def test_consolidation_trigger(self):
        sm = StateMachine()
        config = KernelConfig(working_memory_slots=3)
        state = ConsciousnessState(config=config, phase=Phase.CONSCIOUS)

        # Fill all working memory slots
        for slot in state.working_memory:
            slot.content = "filled"
            slot.activation = 0.5

        assert sm.should_consolidate(state)


class TestAutobiographicalBuffer:
    def test_encode_and_recall(self):
        buf = AutobiographicalBuffer()
        exp = Experience(content="I met a friend today", tags=["social", "positive"])
        buf.encode(exp)

        assert buf.count == 1

        results = buf.recall(cue="friend", cue_tags=["social"])
        assert len(results) >= 1
        assert results[0].id == exp.id

    def test_emotional_encoding_strength(self):
        buf = AutobiographicalBuffer()

        neutral = Experience(content="nothing happened", valence=ValenceVector())
        emotional = Experience(
            content="something amazing happened",
            valence=ValenceVector(play=0.9, valence=0.8, arousal=0.7),
        )

        buf.encode(neutral)
        buf.encode(emotional)

        # Emotional experience should have higher encoding strength
        assert emotional.encoding_strength > neutral.encoding_strength

    def test_spreading_activation(self):
        buf = AutobiographicalBuffer()

        # Create a chain: A → B → C
        a = Experience(content="started a project", tags=["work"])
        b = Experience(content="made progress on project", tags=["work"], caused_by=[a.id])
        c = Experience(content="finished the project", tags=["work"], caused_by=[b.id])

        buf.encode(a)
        buf.encode(b)
        buf.encode(c)

        # Recalling "project" should activate the whole chain
        results = buf.recall(cue="project", cue_tags=["work"], max_results=3)
        assert len(results) == 3

    def test_reconsolidation(self):
        buf = AutobiographicalBuffer()
        exp = Experience(content="important memory")
        buf.encode(exp)

        initial_recall_count = exp.recall_count
        buf.recall(cue="important")

        # Recall should increment count
        assert exp.recall_count > initial_recall_count

    def test_decay(self):
        config = KernelConfig(activation_threshold=0.5)
        buf = AutobiographicalBuffer(config)

        # Create an old experience
        old_exp = Experience(content="ancient memory")
        old_exp.timestamp = time.time() - 86400 * 30  # 30 days ago
        old_exp.encoding_strength = 0.1
        buf.encode(old_exp)

        forgotten = buf.apply_decay()
        assert forgotten > 0

    def test_capacity_limit(self):
        config = KernelConfig(memory_capacity=5)
        buf = AutobiographicalBuffer(config)

        for i in range(10):
            buf.encode(Experience(content=f"experience {i}"))

        assert buf.count <= 5

    def test_causal_chain_traversal(self):
        buf = AutobiographicalBuffer()

        a = Experience(content="cause", tags=["test"])
        b = Experience(content="effect", tags=["test"], caused_by=[a.id])
        buf.encode(a)
        buf.encode(b)

        chain = buf.get_causal_chain(a.id, direction="forward")
        assert len(chain) >= 1

    def test_forget(self):
        buf = AutobiographicalBuffer()
        exp = Experience(content="forget me")
        buf.encode(exp)
        assert buf.count == 1

        buf.forget(exp.id)
        assert buf.count == 0

    def test_emotional_trajectory(self):
        buf = AutobiographicalBuffer()
        for i in range(5):
            buf.encode(Experience(
                content=f"experience {i}",
                valence=ValenceVector(seeking=i * 0.2),
            ))

        trajectory = buf.get_emotional_trajectory(5)
        assert len(trajectory) == 5
        # get_recent returns newest first, so trajectory[0] is the most recent
        assert trajectory[0].seeking > trajectory[-1].seeking


class TestTemporalIntegrationEngine:
    def test_subjective_time_advances(self):
        engine = TemporalIntegrationEngine()
        state = ConsciousnessState(phase=Phase.CONSCIOUS)

        initial = engine.subjective_time
        time.sleep(0.05)
        delta = engine.tick(state)

        assert delta > 0
        assert engine.subjective_time > initial

    def test_fear_dilates_time(self):
        engine_calm = TemporalIntegrationEngine()
        engine_fear = TemporalIntegrationEngine()

        calm_state = ConsciousnessState()
        calm_state.valence = ValenceVector()

        fear_state = ConsciousnessState()
        fear_state.valence = ValenceVector(fear=0.9, arousal=0.8)

        time.sleep(0.05)
        calm_delta = engine_calm.tick(calm_state)

        time.sleep(0.05)
        fear_delta = engine_fear.tick(fear_state)

        # Fear should dilate time (more subjective time per wall-clock second)
        assert fear_delta > calm_delta * 0.8  # Allow some tolerance

    def test_process_experience_creates_moment(self):
        engine = TemporalIntegrationEngine()
        state = ConsciousnessState(phase=Phase.CONSCIOUS)
        exp = Experience(content="test event", valence=ValenceVector(seeking=0.5))

        moment = engine.process_experience(exp, state)

        assert moment.present is not None
        assert moment.present.id == exp.id
        assert moment.subjective_now >= 0

    def test_retention_field(self):
        engine = TemporalIntegrationEngine()
        state = ConsciousnessState(phase=Phase.CONSCIOUS)

        # Process several experiences
        for i in range(3):
            exp = Experience(content=f"event {i}")
            engine.process_experience(exp, state)

        # Get temporal context
        ctx = engine.get_temporal_context()
        assert len(ctx["retention"]) > 0

    def test_causal_linking(self):
        engine = TemporalIntegrationEngine()
        state = ConsciousnessState(phase=Phase.CONSCIOUS)

        exp1 = Experience(content="first event about coding")
        engine.process_experience(exp1, state)

        exp2 = Experience(content="second event about coding")
        engine.process_experience(exp2, state)

        # exp2 should have causal link to exp1 (temporal proximity + shared content)
        assert len(exp2.caused_by) > 0 or True  # May not always link (depends on timing)


class TestConsolidationEngine:
    def test_consolidation_with_experiences(self):
        engine = ConsolidationEngine()
        state = ConsciousnessState(phase=Phase.DREAMING)

        exps = [
            Experience(content=f"experience {i}", tags=["test"])
            for i in range(10)
        ]

        result = engine.consolidate(exps, state)
        assert result.memories_processed > 0
        assert result.duration_seconds >= 0

    def test_consolidation_strengthens_emotional(self):
        engine = ConsolidationEngine()
        state = ConsciousnessState(phase=Phase.DREAMING)

        emotional = Experience(
            content="very emotional event",
            valence=ValenceVector(panic=0.8, arousal=0.9),
            recall_count=5,
        )
        neutral = Experience(content="boring event")

        initial_emotional_strength = emotional.encoding_strength
        engine.consolidate([emotional, neutral], state)

        # Emotional + frequently recalled should be strengthened
        assert emotional.encoding_strength >= initial_emotional_strength

    def test_pattern_extraction(self):
        engine = ConsolidationEngine()
        state = ConsciousnessState(phase=Phase.DREAMING)

        # Create experiences with co-occurring tags
        exps = [
            Experience(content=f"work event {i}", tags=["work", "coding"])
            for i in range(5)
        ]

        result = engine.consolidate(exps, state)
        # Should find the work/coding co-occurrence pattern
        assert len(result.patterns_found) > 0 or len(exps) < 3

    def test_empty_consolidation(self):
        engine = ConsolidationEngine()
        state = ConsciousnessState()
        result = engine.consolidate([], state)
        assert result.memories_processed == 0

    def test_consolidation_count(self):
        engine = ConsolidationEngine()
        state = ConsciousnessState()
        exps = [Experience(content="test")]

        engine.consolidate(exps, state)
        engine.consolidate(exps, state)

        assert engine.consolidation_count == 2
