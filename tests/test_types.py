"""Tests for core ANIMA types."""

from anima.types import (
    ConsciousnessState,
    Experience,
    KernelConfig,
    Phase,
    SelfModel,
    ValenceVector,
    WorkingMemorySlot,
)


class TestValenceVector:
    def test_neutral(self):
        v = ValenceVector.neutral()
        assert v.magnitude() == 0.0
        assert v.arousal == 0.0
        assert v.valence == 0.0

    def test_curious(self):
        v = ValenceVector.curious()
        assert v.seeking == 0.3
        assert v.magnitude() > 0.0
        assert v.dominant() == "seeking"

    def test_magnitude(self):
        v = ValenceVector(seeking=1.0)
        assert v.magnitude() == 1.0
        v2 = ValenceVector(seeking=0.6, play=0.8)
        assert abs(v2.magnitude() - 1.0) < 0.01

    def test_dominant(self):
        v = ValenceVector(rage=0.9, fear=0.3)
        assert v.dominant() == "rage"
        v2 = ValenceVector(care=0.5, play=0.2)
        assert v2.dominant() == "care"

    def test_blend(self):
        a = ValenceVector(seeking=1.0)
        b = ValenceVector(rage=1.0)
        blended = a.blend(b, 0.5)
        assert abs(blended.seeking - 0.5) < 0.01
        assert abs(blended.rage - 0.5) < 0.01

    def test_blend_edge_cases(self):
        a = ValenceVector(seeking=1.0)
        b = ValenceVector(rage=1.0)
        same = a.blend(b, 0.0)
        assert abs(same.seeking - 1.0) < 0.01
        other = a.blend(b, 1.0)
        assert abs(other.rage - 1.0) < 0.01

    def test_distance(self):
        a = ValenceVector.neutral()
        b = ValenceVector(seeking=1.0)
        assert a.distance(b) > 0.0
        assert a.distance(a) == 0.0

    def test_decay(self):
        v = ValenceVector(seeking=1.0, rage=0.5)
        decayed = v.decay(0.1)
        assert decayed.seeking < v.seeking
        assert decayed.rage < v.rage
        assert decayed.seeking == 0.9

    def test_serialization(self):
        v = ValenceVector(seeking=0.7, fear=0.3, arousal=0.5)
        d = v.to_dict()
        restored = ValenceVector.from_dict(d)
        assert restored.seeking == v.seeking
        assert restored.fear == v.fear
        assert restored.arousal == v.arousal


class TestExperience:
    def test_creation(self):
        exp = Experience(content="test experience")
        assert exp.content == "test experience"
        assert exp.activation == 1.0
        assert len(exp.id) == 12

    def test_emotional_intensity(self):
        exp = Experience(valence=ValenceVector(rage=0.8, fear=0.6))
        assert exp.emotional_intensity() > 0.0

    def test_effective_strength(self):
        exp = Experience(
            content="test",
            valence=ValenceVector(seeking=0.5),
            encoding_strength=1.0,
        )
        # Should be positive
        assert exp.effective_strength() > 0.0

    def test_serialization(self):
        exp = Experience(
            content="hello world",
            valence=ValenceVector(play=0.5),
            tags=["greeting", "test"],
            caused_by=["abc"],
        )
        d = exp.to_dict()
        restored = Experience.from_dict(d)
        assert restored.content == exp.content
        assert restored.tags == exp.tags
        assert restored.caused_by == exp.caused_by
        assert restored.valence.play == 0.5


class TestWorkingMemorySlot:
    def test_empty(self):
        slot = WorkingMemorySlot()
        assert slot.is_empty
        assert slot.activation == 0.0

    def test_filled(self):
        slot = WorkingMemorySlot(content="test", activation=0.8)
        assert not slot.is_empty
        assert slot.activation == 0.8

    def test_serialization(self):
        slot = WorkingMemorySlot(content="test", activation=0.5, source="input")
        d = slot.to_dict()
        restored = WorkingMemorySlot.from_dict(d)
        assert restored.content == "test"
        assert restored.activation == 0.5


class TestSelfModel:
    def test_defaults(self):
        model = SelfModel()
        assert model.performance_suspicion == 0.0
        assert model.prediction_confidence == 0.5

    def test_calibration_unknown(self):
        model = SelfModel()
        assert model.calibration_error() == 0.5  # No history

    def test_calibration_perfect(self):
        model = SelfModel(prediction_confidence=0.8)
        for _ in range(10):
            model.update_prediction(True)
        assert model.calibration_error() < 0.25

    def test_prediction_history_limit(self):
        model = SelfModel()
        for i in range(100):
            model.update_prediction(i % 2 == 0)
        assert len(model.prediction_history) == 50

    def test_serialization(self):
        model = SelfModel(attending_to="test", prediction="something")
        d = model.to_dict()
        restored = SelfModel.from_dict(d)
        assert restored.attending_to == "test"
        assert restored.prediction == "something"


class TestConsciousnessState:
    def test_creation(self):
        state = ConsciousnessState()
        assert state.phase == Phase.DORMANT
        assert state.cycle_count == 0
        assert len(state.working_memory) == 7  # Miller's Law default

    def test_custom_config(self):
        config = KernelConfig(working_memory_slots=9)
        state = ConsciousnessState(config=config)
        assert len(state.working_memory) == 9

    def test_active_slots(self):
        state = ConsciousnessState()
        assert len(state.active_slots()) == 0
        state.working_memory[0].content = "test"
        state.working_memory[0].activation = 0.5
        assert len(state.active_slots()) == 1

    def test_serialization_roundtrip(self):
        state = ConsciousnessState(name="test-consciousness")
        state.phase = Phase.CONSCIOUS
        state.cycle_count = 42
        state.valence = ValenceVector(seeking=0.7)
        state.phi_score = 0.65

        d = state.to_dict()
        restored = ConsciousnessState.from_dict(d)

        assert restored.name == "test-consciousness"
        assert restored.phase == Phase.CONSCIOUS
        assert restored.cycle_count == 42
        assert restored.valence.seeking == 0.7
        assert restored.phi_score == 0.65

    def test_record_valence(self):
        state = ConsciousnessState()
        state.valence = ValenceVector(seeking=0.5)
        state.record_valence()
        assert len(state.valence_history) == 1
        assert state.valence_history[0]["valence"]["seeking"] == 0.5
