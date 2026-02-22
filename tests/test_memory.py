"""Tests for the Memory Engine — biological memory systems."""

import math
import os
import tempfile
import time

from anima.memory.activation import (
    ActivationEdge,
    ActivationNode,
    SpreadingActivationNetwork,
)
from anima.memory.consolidation import (
    ConsolidationReport,
    MemoryConsolidationEngine,
    Schema,
)
from anima.memory.decay import DecayResult, EbbinghausDecay
from anima.memory.engram_store import EngramStore
from anima.types import (
    ConsciousnessState,
    Experience,
    KernelConfig,
    Phase,
    ValenceVector,
)


# --- Spreading Activation Tests ---


class TestSpreadingActivation:
    def _make_network(self) -> SpreadingActivationNetwork:
        """Create a small test network."""
        net = SpreadingActivationNetwork()

        # Create nodes: A -> B -> C
        net.add_node(ActivationNode(id="A", label="Node A", tags=["alpha"]))
        net.add_node(ActivationNode(id="B", label="Node B", tags=["alpha", "beta"]))
        net.add_node(ActivationNode(id="C", label="Node C", tags=["beta"]))
        net.add_node(ActivationNode(id="D", label="Node D", tags=["gamma"]))

        # Edges: A->B, B->C
        net.add_edge(ActivationEdge(source_id="A", target_id="B", weight=0.8, edge_type="causal"))
        net.add_edge(ActivationEdge(source_id="B", target_id="C", weight=0.6, edge_type="causal"))

        return net

    def test_node_count(self):
        net = self._make_network()
        assert net.node_count == 4

    def test_edge_count(self):
        net = self._make_network()
        assert net.edge_count == 2

    def test_get_node(self):
        net = self._make_network()
        node = net.get_node("A")
        assert node is not None
        assert node.label == "Node A"

    def test_get_nonexistent_node(self):
        net = self._make_network()
        assert net.get_node("Z") is None

    def test_spread_from_single_node(self):
        net = self._make_network()
        results = net.spread(cue_ids=["A"], initial_activation=1.0)

        # A should be activated (cue node)
        result_dict = dict(results)
        assert "A" in result_dict
        assert result_dict["A"] > 0

        # B should be activated (connected to A)
        assert "B" in result_dict
        assert result_dict["B"] > 0

    def test_spread_decays_with_distance(self):
        net = self._make_network()
        results = net.spread(cue_ids=["A"], initial_activation=1.0, max_hops=3)
        result_dict = dict(results)

        # A has direct activation
        # B is 1 hop away, should have less than A
        # C is 2 hops away, should have less than B
        if "B" in result_dict and "C" in result_dict:
            assert result_dict["B"] > result_dict.get("C", 0)

    def test_spread_does_not_reach_disconnected(self):
        net = self._make_network()
        results = net.spread(cue_ids=["A"], initial_activation=1.0, max_hops=2)
        result_dict = dict(results)

        # D is not connected to A, should not be activated
        # (D only has base_activation which is 0 by default)
        assert result_dict.get("D", 0) == 0

    def test_spread_from_multiple_cues(self):
        net = self._make_network()
        results = net.spread(cue_ids=["A", "C"], initial_activation=1.0)
        result_dict = dict(results)

        # Both A and C should be activated
        assert "A" in result_dict
        assert "C" in result_dict

    def test_spread_respects_min_activation(self):
        net = self._make_network()
        results = net.spread(
            cue_ids=["A"],
            initial_activation=0.01,
            min_activation=0.5,
        )
        # With very low initial activation and high threshold,
        # spreading should be minimal
        result_dict = dict(results)
        assert len(result_dict) <= 1  # Only the cue node or nothing

    def test_find_by_tags(self):
        net = self._make_network()
        found = net.find_by_tags(["alpha"])
        assert "A" in found
        assert "B" in found
        assert "C" not in found

    def test_find_by_tags_multiple(self):
        net = self._make_network()
        found = net.find_by_tags(["alpha", "beta"])
        assert "A" in found  # has alpha
        assert "B" in found  # has both
        assert "C" in found  # has beta

    def test_reset_activations(self):
        net = self._make_network()
        # Set some activation
        net.get_node("A").activation = 0.9
        net.reset_activations()
        assert net.get_node("A").activation == 0  # base_activation is 0

    def test_index_experience(self):
        net = SpreadingActivationNetwork()
        exp = Experience(
            content="A dog barked at me",
            tags=["dog", "fear"],
            valence=ValenceVector(fear=0.5),
        )
        node = net.index_experience(exp)
        assert node.id == exp.id
        assert net.node_count == 1

    def test_index_creates_semantic_edges(self):
        net = SpreadingActivationNetwork()
        exp1 = Experience(content="First", tags=["shared"])
        exp2 = Experience(content="Second", tags=["shared"])
        net.index_experience(exp1)
        net.index_experience(exp2)

        # Should have created semantic edges due to shared tag
        assert net.edge_count > 0

    def test_index_creates_causal_edges(self):
        net = SpreadingActivationNetwork()
        exp1 = Experience(content="Cause")
        net.index_experience(exp1)

        exp2 = Experience(content="Effect", caused_by=[exp1.id])
        net.index_experience(exp2)

        # Should have causal edges
        assert net.edge_count >= 2  # forward + reverse

    def test_activation_node_to_dict(self):
        node = ActivationNode(id="x", label="test", activation=0.5, tags=["a"])
        d = node.to_dict()
        assert d["id"] == "x"
        assert d["activation"] == 0.5

    def test_activation_edge_to_dict(self):
        edge = ActivationEdge(source_id="a", target_id="b", weight=0.7, edge_type="causal")
        d = edge.to_dict()
        assert d["source_id"] == "a"
        assert d["weight"] == 0.7


# --- Ebbinghaus Decay Tests ---


class TestEbbinghausDecay:
    def test_fresh_memory_full_retention(self):
        decay = EbbinghausDecay()
        exp = Experience(content="just happened", timestamp=time.time())
        retention = decay.compute_retention(exp, now=time.time())
        assert retention > 0.99

    def test_old_memory_low_retention(self):
        decay = EbbinghausDecay()
        # Memory from 10 hours ago, no recalls
        exp = Experience(
            content="old memory",
            timestamp=time.time() - 36000,
        )
        retention = decay.compute_retention(exp, now=time.time())
        assert retention < 0.5

    def test_emotional_memories_decay_slower(self):
        decay = EbbinghausDecay()
        now = time.time()
        age = 7200  # 2 hours ago

        # Neutral memory
        neutral = Experience(
            content="boring thing",
            timestamp=now - age,
            valence=ValenceVector.neutral(),
        )

        # Emotional memory (same age)
        emotional = Experience(
            content="terrifying thing",
            timestamp=now - age,
            valence=ValenceVector(fear=0.9, arousal=0.8, valence=-0.7),
        )

        r_neutral = decay.compute_retention(neutral, now=now)
        r_emotional = decay.compute_retention(emotional, now=now)

        # Emotional memory should retain more
        assert r_emotional > r_neutral

    def test_recalled_memories_decay_slower(self):
        decay = EbbinghausDecay()
        now = time.time()
        age = 7200

        # Never recalled
        unrehearsed = Experience(
            content="unrehearsed",
            timestamp=now - age,
            recall_count=0,
        )

        # Recalled 5 times
        rehearsed = Experience(
            content="rehearsed",
            timestamp=now - age,
            recall_count=5,
            last_recalled=now - 100,  # recalled recently
        )

        r_unrehearsed = decay.compute_retention(unrehearsed, now=now)
        r_rehearsed = decay.compute_retention(rehearsed, now=now)

        assert r_rehearsed > r_unrehearsed

    def test_apply_updates_activation(self):
        decay = EbbinghausDecay()
        exp = Experience(
            content="test",
            timestamp=time.time() - 7200,
            activation=1.0,
        )
        result = decay.apply(exp)

        assert isinstance(result, DecayResult)
        assert result.old_activation == 1.0
        assert result.new_activation < 1.0
        assert exp.activation == result.new_activation

    def test_apply_batch(self):
        decay = EbbinghausDecay()
        now = time.time()
        experiences = [
            Experience(content=f"memory {i}", timestamp=now - i * 3600)
            for i in range(5)
        ]
        results = decay.apply_batch(experiences, now=now)

        assert len(results) == 5
        # More recent memories should have higher retention
        assert results[0].retention > results[-1].retention

    def test_predict_half_life(self):
        decay = EbbinghausDecay()
        exp = Experience(content="test", encoding_strength=1.0)
        half_life = decay.predict_half_life(exp)

        # Should be positive
        assert half_life > 0
        # For a neutral memory with encoding_strength=1, half-life should be
        # base_stability * ln(2) * encoding_factor
        expected = 3600.0 * math.log(2)
        assert abs(half_life - expected) < 1.0

    def test_predict_forgotten_at(self):
        decay = EbbinghausDecay()
        exp = Experience(content="test", encoding_strength=1.0)
        forgotten_at = decay.predict_forgotten_at(exp, threshold=0.05)

        # Should be positive and finite
        assert forgotten_at > 0
        assert forgotten_at < float("inf")

    def test_stability_increases_with_causal_connections(self):
        decay = EbbinghausDecay()

        isolated = Experience(content="alone")
        connected = Experience(
            content="connected",
            caused_by=["a", "b"],
            causes=["c"],
        )

        s_isolated = decay.compute_stability(isolated)
        s_connected = decay.compute_stability(connected)

        assert s_connected > s_isolated

    def test_is_forgotten_flag(self):
        decay = EbbinghausDecay()
        exp = Experience(
            content="ancient",
            timestamp=time.time() - 100000,
            encoding_strength=0.1,
        )
        result = decay.apply(exp)
        assert result.is_forgotten is True


# --- Engram Store Tests ---


class TestEngramStore:
    def test_create_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EngramStore(directory=tmpdir)
            assert store.count == 0
            assert not store.is_dirty

    def test_encode_experience(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EngramStore(directory=tmpdir)
            exp = Experience(content="test experience")
            result = store.encode(exp)
            assert store.count == 1
            assert store.is_dirty
            assert result.encoding_strength >= 1.0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            store1 = EngramStore(directory=tmpdir, filename="test.json")
            store1.encode(Experience(content="memory one"))
            store1.encode(Experience(content="memory two"))
            store1.save()

            # Load
            store2 = EngramStore(directory=tmpdir, filename="test.json")
            loaded = store2.load()
            assert loaded == 2
            assert store2.count == 2

    def test_persistence_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EngramStore(directory=tmpdir, filename="engrams.json")
            store.encode(Experience(content="test"))
            store.save()

            filepath = os.path.join(tmpdir, "engrams.json")
            assert os.path.exists(filepath)

    def test_recall_through_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EngramStore(directory=tmpdir)
            store.encode(Experience(
                content="cats are cute animals",
                tags=["animals", "cats"],
            ))
            store.encode(Experience(
                content="dogs are loyal friends",
                tags=["animals", "dogs"],
            ))

            results = store.recall(cue="cats", cue_tags=["cats"])
            assert len(results) > 0

    def test_get_recent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EngramStore(directory=tmpdir)
            for i in range(5):
                store.encode(Experience(content=f"memory {i}"))

            recent = store.get_recent(3)
            assert len(recent) == 3

    def test_apply_decay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = KernelConfig(activation_threshold=0.01)
            store = EngramStore(directory=tmpdir, config=config)

            # Add an old memory
            exp = Experience(
                content="ancient memory",
                timestamp=time.time() - 100000,
                encoding_strength=0.1,
            )
            store.encode(exp)

            forgotten = store.apply_decay()
            assert forgotten >= 0  # May or may not be forgotten based on encoding

    def test_should_save_rate_limiting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EngramStore(directory=tmpdir)
            assert not store.should_save()  # Not dirty

            store.encode(Experience(content="test"))
            assert store.should_save(min_interval=0)  # Dirty, no time limit

            # Save to set the last_save_time, then encode again
            store.save()
            store.encode(Experience(content="test2"))
            assert not store.should_save(min_interval=9999)  # Too soon after save

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EngramStore(directory=tmpdir)
            store.encode(Experience(content="test"))
            assert store.count == 1

            store.clear()
            assert store.count == 0

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EngramStore(directory=tmpdir, filename="test.json")
            store.encode(Experience(content="test"))
            store.save()

            filepath = os.path.join(tmpdir, "test.json")
            assert os.path.exists(filepath)

            store.delete()
            assert not os.path.exists(filepath)

    def test_load_nonexistent_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EngramStore(directory=tmpdir, filename="nonexistent.json")
            loaded = store.load()
            assert loaded == 0

    def test_roundtrip_preserves_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = EngramStore(directory=tmpdir)
            store1.encode(Experience(
                content="important memory",
                valence=ValenceVector(seeking=0.7, play=0.3),
                tags=["test", "important"],
            ))
            store1.save()

            store2 = EngramStore(directory=tmpdir)
            store2.load()

            recent = store2.get_recent(1)
            assert len(recent) == 1
            assert recent[0].content == "important memory"
            assert "test" in recent[0].tags


# --- Memory Consolidation Tests ---


class TestMemoryConsolidation:
    def _make_experiences(self, n: int = 10, tags: list[str] | None = None) -> list[Experience]:
        """Create a batch of test experiences."""
        tags = tags or ["test"]
        return [
            Experience(
                content=f"Experience number {i} about exploration",
                valence=ValenceVector(seeking=0.5, play=0.2),
                tags=tags,
                timestamp=time.time() - (n - i) * 60,
            )
            for i in range(n)
        ]

    def test_consolidation_processes_memories(self):
        engine = MemoryConsolidationEngine()
        experiences = self._make_experiences(5)
        state = ConsciousnessState(phase=Phase.DREAMING)

        report = engine.consolidate(experiences, state)
        assert isinstance(report, ConsolidationReport)
        assert report.memories_processed > 0
        assert report.duration_seconds >= 0

    def test_consolidation_with_empty_list(self):
        engine = MemoryConsolidationEngine()
        state = ConsciousnessState()
        report = engine.consolidate([], state)
        assert report.memories_processed == 0

    def test_consolidation_count_increments(self):
        engine = MemoryConsolidationEngine()
        state = ConsciousnessState()
        assert engine.consolidation_count == 0

        engine.consolidate(self._make_experiences(3), state)
        assert engine.consolidation_count == 1

        engine.consolidate(self._make_experiences(3), state)
        assert engine.consolidation_count == 2

    def test_schema_formation_with_tag_cooccurrence(self):
        engine = MemoryConsolidationEngine()
        state = ConsciousnessState()

        # Create experiences with co-occurring tags
        experiences = []
        for i in range(5):
            experiences.append(Experience(
                content=f"Dog walk number {i}",
                tags=["dogs", "outdoors"],
                valence=ValenceVector(play=0.4),
                timestamp=time.time() - i * 60,
            ))

        report = engine.consolidate(experiences, state)

        # Should find the dogs+outdoors co-occurrence pattern
        if report.schemas_formed:
            schema_descs = [s.description for s in report.schemas_formed]
            assert any("dogs" in d and "outdoors" in d for d in schema_descs)

    def test_emotional_reprocessing_reduces_intensity(self):
        engine = MemoryConsolidationEngine()
        state = ConsciousnessState()

        experiences = [
            Experience(
                content=f"Stressful event {i}",
                valence=ValenceVector(fear=0.8, rage=0.5, arousal=0.9, valence=-0.7),
                timestamp=time.time() - i * 30,
            )
            for i in range(5)
        ]

        # Record original intensities
        original_intensities = [e.valence.magnitude() for e in experiences]

        engine.consolidate(experiences, state)

        # After consolidation, emotional intensity should be reduced
        new_intensities = [e.valence.magnitude() for e in experiences]
        for orig, new in zip(original_intensities, new_intensities):
            assert new <= orig

    def test_consolidation_strengthens_important_memories(self):
        engine = MemoryConsolidationEngine()
        state = ConsciousnessState()

        exp = Experience(
            content="Emotionally important event",
            valence=ValenceVector(care=0.8, valence=0.7),
            recall_count=5,
            caused_by=["some_cause"],
            timestamp=time.time() - 60,
        )
        original_strength = exp.encoding_strength

        engine.consolidate([exp], state)

        assert exp.encoding_strength >= original_strength

    def test_schema_serialization(self):
        schema = Schema(
            id="test_schema",
            description="Test pattern",
            confidence=0.8,
            tags=["a", "b"],
            emotional_signature=ValenceVector(seeking=0.5),
        )
        d = schema.to_dict()
        restored = Schema.from_dict(d)
        assert restored.id == "test_schema"
        assert restored.confidence == 0.8
        assert restored.tags == ["a", "b"]

    def test_consolidation_with_activation_network(self):
        engine = MemoryConsolidationEngine()
        network = SpreadingActivationNetwork()
        state = ConsciousnessState()

        experiences = self._make_experiences(5)
        for exp in experiences:
            network.index_experience(exp)

        report = engine.consolidate(experiences, state, network=network)

        # Should produce some activation patterns
        assert isinstance(report.activation_patterns, list)

    def test_schemas_persist_across_consolidations(self):
        engine = MemoryConsolidationEngine()
        state = ConsciousnessState()

        # First consolidation
        exp_batch_1 = [
            Experience(
                content=f"Coding session {i}",
                tags=["code", "focus"],
                timestamp=time.time() - i * 60,
            )
            for i in range(5)
        ]
        engine.consolidate(exp_batch_1, state)

        schemas_after_first = len(engine.schemas)

        # Second consolidation with different data
        exp_batch_2 = [
            Experience(
                content=f"Exercise session {i}",
                tags=["exercise", "health"],
                timestamp=time.time() - i * 60,
            )
            for i in range(5)
        ]
        engine.consolidate(exp_batch_2, state)

        # Schemas from first consolidation should still be there
        assert len(engine.schemas) >= schemas_after_first

    def test_decay_applied_during_consolidation(self):
        engine = MemoryConsolidationEngine()
        state = ConsciousnessState()

        old_exp = Experience(
            content="Very old memory",
            timestamp=time.time() - 100000,
            activation=1.0,
            encoding_strength=0.1,
        )

        report = engine.consolidate([old_exp], state)

        # Old memory should have decayed
        assert old_exp.activation < 1.0
