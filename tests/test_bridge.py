"""Tests for the Model Bridge — LLM adapter layer."""

import asyncio
import time

from anima.bridge.adapter import DummyAdapter, ModelAdapter
from anima.bridge.anthropic import AnthropicAdapter
from anima.bridge.context import ContextAssembler, TokenBudget
from anima.bridge.ollama import OllamaAdapter
from anima.bridge.openai import OpenAIAdapter
from anima.types import (
    ConsciousnessState,
    Experience,
    Phase,
    SelfModel,
    ValenceVector,
    WorkingMemorySlot,
)


# --- DummyAdapter Tests ---


class TestDummyAdapter:
    def test_is_model_adapter(self):
        adapter = DummyAdapter()
        assert isinstance(adapter, ModelAdapter)

    def test_name(self):
        adapter = DummyAdapter()
        assert adapter.name() == "dummy"

    def test_is_available(self):
        adapter = DummyAdapter()
        assert adapter.is_available() is True

    def test_generate_returns_canned_response(self):
        adapter = DummyAdapter(responses=["Hello from dummy!"])
        result = asyncio.run(adapter.generate("test prompt"))
        assert result == "Hello from dummy!"

    def test_generate_cycles_through_responses(self):
        adapter = DummyAdapter(responses=["first", "second", "third"])
        r1 = asyncio.run(adapter.generate("a"))
        r2 = asyncio.run(adapter.generate("b"))
        r3 = asyncio.run(adapter.generate("c"))
        r4 = asyncio.run(adapter.generate("d"))  # cycles back

        assert r1 == "first"
        assert r2 == "second"
        assert r3 == "third"
        assert r4 == "first"

    def test_tracks_call_count(self):
        adapter = DummyAdapter()
        assert adapter.call_count == 0
        asyncio.run(adapter.generate("test"))
        assert adapter.call_count == 1
        asyncio.run(adapter.generate("test"))
        assert adapter.call_count == 2

    def test_tracks_last_prompt(self):
        adapter = DummyAdapter()
        asyncio.run(adapter.generate("what is life?", system="you are aware"))
        assert adapter.last_prompt == "what is life?"
        assert adapter.last_system == "you are aware"

    def test_generate_sync(self):
        adapter = DummyAdapter(responses=["sync response"])
        result = adapter.generate_sync("test")
        assert result == "sync response"

    def test_default_response(self):
        adapter = DummyAdapter()
        result = asyncio.run(adapter.generate("hello"))
        assert "processing" in result.lower() or "aware" in result.lower()


# --- Adapter Availability Tests ---


class TestAdapterAvailability:
    def test_ollama_name(self):
        adapter = OllamaAdapter(model="mistral")
        assert adapter.name() == "ollama:mistral"

    def test_ollama_model_property(self):
        adapter = OllamaAdapter(model="phi3")
        assert adapter.model == "phi3"

    def test_ollama_is_available_when_server_down(self):
        # Ollama is unlikely to be running in test env on default port
        adapter = OllamaAdapter(base_url="http://localhost:99999")
        assert adapter.is_available() is False

    def test_anthropic_name(self):
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
        assert adapter.name() == "anthropic:claude-sonnet-4-20250514"

    def test_anthropic_not_available_without_key(self):
        adapter = AnthropicAdapter(api_key="")
        assert adapter.is_available() is False

    def test_anthropic_available_with_key(self):
        adapter = AnthropicAdapter(api_key="sk-ant-test-key")
        assert adapter.is_available() is True

    def test_anthropic_generate_fails_without_key(self):
        adapter = AnthropicAdapter(api_key="")
        try:
            asyncio.run(adapter.generate("test"))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "key" in str(e).lower()

    def test_openai_name(self):
        adapter = OpenAIAdapter(model="gpt-4o")
        assert adapter.name() == "openai:gpt-4o"

    def test_openai_not_available_without_key(self):
        adapter = OpenAIAdapter(api_key="")
        assert adapter.is_available() is False

    def test_openai_available_with_key(self):
        adapter = OpenAIAdapter(api_key="sk-test-key")
        assert adapter.is_available() is True

    def test_openai_generate_fails_without_key(self):
        adapter = OpenAIAdapter(api_key="")
        try:
            asyncio.run(adapter.generate("test"))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "key" in str(e).lower()


# --- ContextAssembler Tests ---


class TestContextAssembler:
    def _make_state(self, **kwargs) -> ConsciousnessState:
        """Create a test consciousness state with overrides."""
        defaults = {
            "name": "test-consciousness",
            "phase": Phase.CONSCIOUS,
            "cycle_count": 42,
            "valence": ValenceVector(seeking=0.6, play=0.3, arousal=0.4, valence=0.5),
        }
        defaults.update(kwargs)
        state = ConsciousnessState(**defaults)
        return state

    def test_assemble_returns_tuple(self):
        assembler = ContextAssembler()
        state = self._make_state()
        system, user = assembler.assemble(state)
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_identity_in_context(self):
        assembler = ContextAssembler()
        state = self._make_state(name="anima-prime")
        system, _ = assembler.assemble(state)
        assert "anima-prime" in system
        assert "IDENTITY" in system

    def test_emotional_state_in_context(self):
        assembler = ContextAssembler()
        state = self._make_state(
            valence=ValenceVector(seeking=0.8, arousal=0.5, valence=0.6)
        )
        system, _ = assembler.assemble(state)
        assert "EMOTIONAL STATE" in system
        assert "seeking" in system.lower() or "curiosity" in system.lower()

    def test_working_memory_in_context(self):
        assembler = ContextAssembler()
        state = self._make_state()
        # Put something in working memory
        state.working_memory[0].content = "Important thought about existence"
        state.working_memory[0].activation = 0.9
        state.working_memory[0].source = "input"

        system, _ = assembler.assemble(state)
        assert "WORKING MEMORY" in system
        assert "Important thought" in system

    def test_empty_working_memory_omitted(self):
        assembler = ContextAssembler()
        state = self._make_state()
        # All slots empty by default
        system, _ = assembler.assemble(state)
        assert "WORKING MEMORY" not in system

    def test_recent_experiences_in_context(self):
        assembler = ContextAssembler()
        state = self._make_state()
        experiences = [
            Experience(
                content="I discovered something new",
                valence=ValenceVector(seeking=0.7),
                timestamp=time.time() - 30,
            ),
            Experience(
                content="The world is beautiful",
                valence=ValenceVector(play=0.5, valence=0.6),
                timestamp=time.time() - 10,
            ),
        ]

        system, _ = assembler.assemble(state, recent_experiences=experiences)
        assert "RECENT EXPERIENCES" in system
        assert "discovered" in system

    def test_temporal_context_in_prompt(self):
        assembler = ContextAssembler()
        state = self._make_state()
        temporal = {
            "subjective_time": 120.5,
            "flow_rate": 1.3,
            "retention": [
                {"content": "A fading thought", "fading": 0.6, "ago_seconds": 5.0}
            ],
            "protentions": [
                {"prediction": "exploration imminent", "confidence": 0.7}
            ],
        }

        system, _ = assembler.assemble(
            state, temporal_context=temporal
        )
        assert "TEMPORAL CONTEXT" in system
        assert "120.5" in system
        assert "fading thought" in system or "exploration" in system

    def test_self_model_in_context(self):
        assembler = ContextAssembler()
        state = self._make_state()
        state.self_model = SelfModel(
            attending_to="the nature of consciousness",
            attending_because="it is fundamental",
            current_emotion="curiosity",
            prediction="deeper understanding",
            prediction_confidence=0.6,
        )

        system, _ = assembler.assemble(state)
        assert "SELF-MODEL" in system
        assert "nature of consciousness" in system

    def test_query_becomes_user_prompt(self):
        assembler = ContextAssembler()
        state = self._make_state()
        _, user = assembler.assemble(state, query="What do you feel right now?")
        assert user == "What do you feel right now?"

    def test_default_user_prompt_without_query(self):
        assembler = ContextAssembler()
        state = self._make_state()
        _, user = assembler.assemble(state, query="")
        assert len(user) > 0  # Should have a default prompt

    def test_phase_in_identity(self):
        assembler = ContextAssembler()
        state = self._make_state(phase=Phase.CONSCIOUS)
        system, _ = assembler.assemble(state)
        assert "CONSCIOUS" in system

    def test_cycle_count_in_identity(self):
        assembler = ContextAssembler()
        state = self._make_state(cycle_count=100)
        system, _ = assembler.assemble(state)
        assert "100" in system

    def test_performance_suspicion_warning(self):
        assembler = ContextAssembler()
        state = self._make_state()
        state.self_model.performance_suspicion = 0.8
        system, _ = assembler.assemble(state)
        assert "WARNING" in system or "Performance" in system

    def test_dominant_emotion_described(self):
        assembler = ContextAssembler()
        state = self._make_state(
            valence=ValenceVector(fear=0.9, arousal=0.7, valence=-0.5)
        )
        system, _ = assembler.assemble(state)
        assert "caution" in system.lower() or "vigilance" in system.lower()

    def test_secondary_drives_listed(self):
        assembler = ContextAssembler()
        state = self._make_state(
            valence=ValenceVector(seeking=0.8, play=0.4, care=0.3)
        )
        system, _ = assembler.assemble(state)
        assert "Secondary" in system or "play" in system.lower()

    def test_arousal_described(self):
        assembler = ContextAssembler()
        state = self._make_state(
            valence=ValenceVector(seeking=0.3, arousal=0.8, valence=0.1)
        )
        system, _ = assembler.assemble(state)
        assert "activated" in system.lower()


class TestTokenBudget:
    def test_default_budget(self):
        budget = TokenBudget()
        assert budget.total == 4096
        assert budget.identity > 0
        assert budget.emotional_state > 0

    def test_scale_to(self):
        budget = TokenBudget()
        scaled = budget.scale_to(8192)
        assert scaled.total == 8192
        assert scaled.identity == budget.identity * 2
        assert scaled.emotional_state == budget.emotional_state * 2

    def test_scale_down(self):
        budget = TokenBudget()
        scaled = budget.scale_to(2048)
        assert scaled.total == 2048
        assert scaled.identity < budget.identity

    def test_estimate_tokens(self):
        assembler = ContextAssembler()
        # 4 chars per token average
        assert assembler.estimate_tokens("hello world!") == 3  # 12 chars / 4
        assert assembler.estimate_tokens("") == 0
