"""Tests for state persistence."""

import json
import os
import tempfile

from anima.state import StateManager
from anima.types import (
    ConsciousnessState,
    Experience,
    KernelConfig,
    Phase,
    ValenceVector,
)


class TestStateManager:
    def test_save_and_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            state = ConsciousnessState(name="test", phase=Phase.CONSCIOUS)
            state.cycle_count = 42
            state.valence = ValenceVector(seeking=0.7)

            manager.save_state(state)
            loaded = manager.load_state()

            assert loaded is not None
            assert loaded.name == "test"
            assert loaded.phase == Phase.CONSCIOUS
            assert loaded.cycle_count == 42
            assert loaded.valence.seeking == 0.7

    def test_save_and_load_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            experiences = [
                Experience(content=f"experience {i}", tags=["test"])
                for i in range(5)
            ]

            manager.save_memory(experiences)
            loaded = manager.load_memory()

            assert len(loaded) == 5
            assert loaded[0].content == "experience 0"
            assert loaded[0].tags == ["test"]

    def test_save_all_load_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            state = ConsciousnessState(name="full-test")
            experiences = [Experience(content="test exp")]

            manager.save_all(state, experiences)
            loaded_state, loaded_exps = manager.load_all()

            assert loaded_state is not None
            assert loaded_state.name == "full-test"
            assert len(loaded_exps) == 1

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            assert not manager.exists()

            manager.save_state(ConsciousnessState())
            assert manager.exists()

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            manager.save_state(ConsciousnessState())
            manager.save_memory([Experience(content="test")])
            assert manager.exists()

            manager.delete()
            assert not manager.exists()
            assert not manager.memory_path.exists()

    def test_atomic_write_creates_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            state = ConsciousnessState()
            manager.save_state(state)

            # Should be valid JSON
            with open(manager.state_path) as f:
                data = json.load(f)
            assert "_version" in data
            assert "_saved_at" in data

    def test_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            assert manager.load_state() is None
            assert manager.load_memory() == []

    def test_load_corrupted_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            # Write corrupted JSON
            with open(manager.state_path, "w") as f:
                f.write("not valid json{{{")

            assert manager.load_state() is None

    def test_dirty_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(tmpdir)
            assert not manager.should_save_memory()

            manager.mark_dirty(memory=True)
            # Won't save immediately due to rate limiting
            # But force=True overrides
            assert manager.should_save_memory(force=True)
