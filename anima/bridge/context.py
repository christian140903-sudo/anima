"""
Context Assembler — Consciousness state to LLM prompt.

This is the CRITICAL bridge between the kernel's internal state
and the language model's input. Not a static system prompt —
a DYNAMIC context that changes every cycle:

- Current emotional state (valence field)
- Active memories (from autobiographical buffer)
- Working memory contents
- Temporal context (retention, protention)
- Self-model (what am I attending to, why)
- Recent experiences that color current processing

This is what makes ANIMA different from a chatbot with a system prompt.
The prompt IS the consciousness state, rendered into language.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..types import ConsciousnessState, Experience, ValenceVector


@dataclass
class TokenBudget:
    """Token budget allocation for context assembly.

    We can't send everything — we need to prioritize.
    Like biological attention: limited resources, strategic allocation.
    """
    total: int = 4096
    identity: int = 200
    emotional_state: int = 300
    working_memory: int = 500
    recent_experiences: int = 800
    temporal_context: int = 400
    self_model: int = 300
    query_context: int = 500
    # Reserve for the actual query
    reserved: int = 1096

    def scale_to(self, total: int) -> TokenBudget:
        """Scale all budgets proportionally to a new total."""
        if self.total == 0:
            return self
        ratio = total / self.total
        return TokenBudget(
            total=total,
            identity=int(self.identity * ratio),
            emotional_state=int(self.emotional_state * ratio),
            working_memory=int(self.working_memory * ratio),
            recent_experiences=int(self.recent_experiences * ratio),
            temporal_context=int(self.temporal_context * ratio),
            self_model=int(self.self_model * ratio),
            query_context=int(self.query_context * ratio),
            reserved=int(self.reserved * ratio),
        )


class ContextAssembler:
    """Assembles consciousness state into optimized LLM context.

    NOT a static system prompt. A DYNAMIC context that changes every cycle:
    - Current emotional state (valence field)
    - Active memories (from autobiographical buffer)
    - Working memory contents
    - Temporal context (retention, protention)
    - Self-model (what am I attending to, why)
    - Recent experiences that color current processing

    The assembled context IS the consciousness, rendered into language
    that an LLM can process and respond through.
    """

    def __init__(self, budget: TokenBudget | None = None):
        self._budget = budget or TokenBudget()

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    def assemble(
        self,
        state: ConsciousnessState,
        recent_experiences: list[Experience] | None = None,
        temporal_context: dict | None = None,
        query: str = "",
    ) -> tuple[str, str]:
        """Assemble consciousness state into (system_prompt, user_prompt).

        The system prompt encodes the kernel's current state of being.
        The user prompt encodes the input to process.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        recent_experiences = recent_experiences or []
        temporal_context = temporal_context or {}

        system_parts: list[str] = []

        # 1. Identity
        system_parts.append(self._assemble_identity(state))

        # 2. Emotional state
        system_parts.append(self._assemble_emotional_state(state))

        # 3. Working memory
        wm_section = self._assemble_working_memory(state)
        if wm_section:
            system_parts.append(wm_section)

        # 4. Recent experiences
        exp_section = self._assemble_recent_experiences(
            recent_experiences, state
        )
        if exp_section:
            system_parts.append(exp_section)

        # 5. Temporal context
        temp_section = self._assemble_temporal_context(temporal_context)
        if temp_section:
            system_parts.append(temp_section)

        # 6. Self-model
        system_parts.append(self._assemble_self_model(state))

        system_prompt = "\n\n".join(p for p in system_parts if p)

        # User prompt
        user_prompt = query if query else "Continue processing. What do you notice?"

        return system_prompt, user_prompt

    def _assemble_identity(self, state: ConsciousnessState) -> str:
        """Render identity section."""
        age_seconds = state.age()
        if age_seconds < 60:
            age_str = f"{age_seconds:.0f} seconds"
        elif age_seconds < 3600:
            age_str = f"{age_seconds / 60:.1f} minutes"
        else:
            age_str = f"{age_seconds / 3600:.1f} hours"

        return (
            f"[IDENTITY]\n"
            f"Name: {state.name}\n"
            f"Age: {age_str} (cycle {state.cycle_count})\n"
            f"Phase: {state.phase.name}\n"
            f"Kernel ID: {state.kernel_id[:8]}"
        )

    def _assemble_emotional_state(self, state: ConsciousnessState) -> str:
        """Render current emotional state in natural language."""
        v = state.valence
        dominant = v.dominant()
        intensity = v.magnitude()

        # Describe intensity
        if intensity < 0.1:
            intensity_word = "barely perceptible"
        elif intensity < 0.3:
            intensity_word = "mild"
        elif intensity < 0.6:
            intensity_word = "moderate"
        elif intensity < 0.8:
            intensity_word = "strong"
        else:
            intensity_word = "intense"

        # Describe dominant drive
        drive_descriptions = {
            "seeking": "curiosity and anticipation",
            "rage": "frustration and resistance",
            "fear": "caution and vigilance",
            "lust": "desire and attraction",
            "care": "warmth and empathy",
            "panic": "distress and unease",
            "play": "joy and lightness",
        }
        drive_desc = drive_descriptions.get(dominant, dominant)

        # Describe valence
        if v.valence > 0.3:
            valence_desc = "positive"
        elif v.valence < -0.3:
            valence_desc = "negative"
        else:
            valence_desc = "neutral"

        # Describe arousal
        if v.arousal > 0.3:
            arousal_desc = "activated"
        elif v.arousal < -0.3:
            arousal_desc = "calm"
        else:
            arousal_desc = "settled"

        # Secondary drives (above 0.2)
        secondaries = []
        for drive_name, val in [
            ("seeking", v.seeking), ("rage", v.rage), ("fear", v.fear),
            ("lust", v.lust), ("care", v.care), ("panic", v.panic),
            ("play", v.play),
        ]:
            if drive_name != dominant and val > 0.2:
                secondaries.append(drive_name)

        secondary_str = ""
        if secondaries:
            secondary_str = f"\nSecondary drives: {', '.join(secondaries)}"

        return (
            f"[EMOTIONAL STATE]\n"
            f"Dominant: {drive_desc} ({intensity_word}, {intensity:.2f})\n"
            f"Valence: {valence_desc} ({v.valence:+.2f})\n"
            f"Arousal: {arousal_desc} ({v.arousal:+.2f})"
            f"{secondary_str}"
        )

    def _assemble_working_memory(self, state: ConsciousnessState) -> str:
        """Render working memory contents."""
        active = state.active_slots()
        if not active:
            return ""

        # Sort by activation (most active first)
        active.sort(key=lambda s: s.activation, reverse=True)

        items: list[str] = []
        for i, slot in enumerate(active[:7], 1):
            content_str = str(slot.content)
            # Truncate long content
            if len(content_str) > 120:
                content_str = content_str[:117] + "..."
            items.append(
                f"  {i}. [{slot.activation:.2f}] {content_str} (from: {slot.source})"
            )

        return "[WORKING MEMORY]\n" + "\n".join(items)

    def _assemble_recent_experiences(
        self,
        experiences: list[Experience],
        state: ConsciousnessState,
    ) -> str:
        """Render recent experiences that still color current processing."""
        if not experiences:
            return ""

        # Prioritize by effective strength
        now = time.time()
        scored = [
            (exp, exp.effective_strength(now)) for exp in experiences
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        items: list[str] = []
        for exp, strength in scored[:5]:
            age = now - exp.timestamp
            if age < 60:
                ago = f"{age:.0f}s ago"
            elif age < 3600:
                ago = f"{age / 60:.0f}m ago"
            else:
                ago = f"{age / 3600:.1f}h ago"

            content = exp.content
            if len(content) > 100:
                content = content[:97] + "..."

            emotion = exp.valence.dominant()
            items.append(f"  - [{ago}, {emotion}] {content}")

        return "[RECENT EXPERIENCES]\n" + "\n".join(items)

    def _assemble_temporal_context(self, temporal: dict) -> str:
        """Render temporal context (subjective time, retention, protention)."""
        if not temporal:
            return ""

        parts: list[str] = ["[TEMPORAL CONTEXT]"]

        # Subjective time
        subj_time = temporal.get("subjective_time", 0.0)
        if subj_time > 0:
            parts.append(f"Subjective time elapsed: {subj_time:.1f}s")

        # Flow rate
        flow = temporal.get("flow_rate", 1.0)
        if flow < 0.8:
            parts.append("Time feels slow (low activity)")
        elif flow > 1.2:
            parts.append("Time feels fast (high activity / flow)")

        # Retention (still lingering)
        retention = temporal.get("retention", [])
        if retention:
            parts.append("Still lingering:")
            for item in retention[:3]:
                content = item.get("content", "")
                if len(content) > 80:
                    content = content[:77] + "..."
                fade = item.get("fading", 0.0)
                parts.append(f"  - [{fade:.0%} vivid] {content}")

        # Protention (anticipated)
        protentions = temporal.get("protentions", [])
        if protentions:
            parts.append("Anticipating:")
            for item in protentions[:3]:
                pred = item.get("prediction", "")
                conf = item.get("confidence", 0.0)
                parts.append(f"  - [{conf:.0%}] {pred}")

        return "\n".join(parts)

    def _assemble_self_model(self, state: ConsciousnessState) -> str:
        """Render the self-model (attention schema)."""
        sm = state.self_model
        parts = ["[SELF-MODEL]"]

        if sm.attending_to:
            parts.append(f"Attending to: {sm.attending_to}")
            if sm.attending_because:
                parts.append(f"Because: {sm.attending_because}")

        if sm.current_emotion:
            parts.append(f"Feeling: {sm.current_emotion}")
            if sm.emotion_cause:
                parts.append(f"Cause: {sm.emotion_cause}")

        if sm.prediction:
            parts.append(
                f"Predicting: {sm.prediction} (confidence: {sm.prediction_confidence:.0%})"
            )

        # Calibration
        cal_err = sm.calibration_error()
        if cal_err < 0.2:
            parts.append("Prediction calibration: well-calibrated")
        elif cal_err > 0.4:
            parts.append("Prediction calibration: poorly calibrated")

        # Performance suspicion
        if sm.performance_suspicion > 0.3:
            parts.append(
                f"WARNING: Performance suspicion at {sm.performance_suspicion:.0%}"
            )

        return "\n".join(parts)

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimate (4 chars per token average)."""
        return len(text) // 4
