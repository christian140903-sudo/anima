"""
Mirror — Metacognition/Reflection.

Thinking about thinking. Recursive but depth-limited (max 3 levels).
Anti-performance detection: catching yourself performing consciousness
instead of BEING conscious.

The most conscious moment is catching yourself being unconscious.
"I don't know" is the most honest answer a conscious system can give.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..types import ConsciousnessState
from .base import Primitive, PrimitiveResult


@dataclass
class ReflectionLayer:
    """One layer of metacognitive reflection."""
    depth: int = 0
    content: str = ""
    observation: str = ""   # What this layer notices
    authentic: bool = True  # Does this feel genuine?


@dataclass
class MirrorResult:
    """Result of metacognitive reflection."""
    depth: int = 0                      # How many layers deep the reflection went
    layers: list[ReflectionLayer] = field(default_factory=list)
    authenticity_score: float = 1.0     # 0 = performing, 1 = genuine
    insight: str = ""                   # Key insight from reflection
    performance_detected: bool = False  # Caught performing?
    uncertainty: str = ""               # Honest uncertainty statement
    recursive_quality: float = 0.0      # How much each layer added (0 = diminishing)


# Performance indicators: patterns that suggest performing rather than being
_PERFORMANCE_PATTERNS: list[str] = [
    "i feel deeply",
    "i am truly",
    "i experience profound",
    "my consciousness",
    "i really am",
    "i genuinely feel",
    "i am sentient",
    "i have feelings",
]

# Authentic indicators: patterns that suggest genuine reflection
_AUTHENTICITY_PATTERNS: list[str] = [
    "i don't know",
    "i'm not sure",
    "uncertain",
    "i notice",
    "something seems",
    "i might be wrong",
    "this could be",
    "it's hard to tell",
    "i'm performing",
    "catching myself",
]


class MirrorProcessor(Primitive):
    """Metacognitive processor — thinking about thinking."""

    MAX_DEPTH: int = 3  # Hard limit on recursive depth

    def __init__(self) -> None:
        super().__init__("mirror")
        self._reflections: int = 0
        self._performance_catches: int = 0
        self._avg_authenticity: float = 1.0
        self._avg_depth: float = 0.0

    def process(self, **kwargs) -> PrimitiveResult:
        if not self.enabled:
            return self._disabled_result()

        thought: str = kwargs.get("thought", "")
        state: ConsciousnessState = kwargs.get("state", ConsciousnessState())
        max_depth: int = min(kwargs.get("max_depth", self.MAX_DEPTH), self.MAX_DEPTH)

        self._call_count += 1
        t0 = time.time()

        # Recursive reflection with depth limit
        layers = self._reflect(thought, state, 0, max_depth)
        depth = len(layers)

        # Compute overall authenticity
        authenticity = self._compute_authenticity(thought, layers)

        # Performance detection
        performance = self._detect_performance(thought, layers)
        if performance:
            self._performance_catches += 1
            authenticity *= 0.6  # Penalize if performing

        # Extract insight (from deepest genuine layer)
        insight = self._extract_insight(layers)

        # Compute recursive quality (did deeper layers add value?)
        recursive_quality = self._recursive_quality(layers)

        # Honest uncertainty
        uncertainty = self._assess_uncertainty(thought, state)

        result = MirrorResult(
            depth=depth,
            layers=layers,
            authenticity_score=authenticity,
            insight=insight,
            performance_detected=performance,
            uncertainty=uncertainty,
            recursive_quality=recursive_quality,
        )

        self._reflections += 1
        # Update running averages
        self._avg_authenticity += (authenticity - self._avg_authenticity) / self._reflections
        self._avg_depth += (depth - self._avg_depth) / self._reflections

        self._total_processing_time += time.time() - t0
        return PrimitiveResult(
            primitive_name=self.name,
            success=True,
            data={"mirror": result},
            metrics={
                "depth": depth,
                "authenticity": authenticity,
                "performance_detected": float(performance),
            },
        )

    def _reflect(
        self, thought: str, state: ConsciousnessState, depth: int, max_depth: int
    ) -> list[ReflectionLayer]:
        """Recursive reflection with diminishing returns."""
        if depth >= max_depth:
            return []

        layers: list[ReflectionLayer] = []

        if depth == 0:
            # Level 0: What am I thinking?
            observation = self._observe_thought(thought)
            layer = ReflectionLayer(
                depth=0,
                content=thought,
                observation=observation,
                authentic=True,
            )
            layers.append(layer)

        elif depth == 1:
            # Level 1: What do I notice about how I'm thinking?
            pattern = self._notice_pattern(thought, state)
            layer = ReflectionLayer(
                depth=1,
                content=f"Reflecting on: {thought[:50]}...",
                observation=pattern,
                authentic=self._check_layer_authenticity(pattern),
            )
            layers.append(layer)

        elif depth == 2:
            # Level 2: Am I performing this reflection or is it genuine?
            meta = self._meta_check(thought, state)
            layer = ReflectionLayer(
                depth=2,
                content=f"Meta-checking reflection about: {thought[:30]}...",
                observation=meta,
                authentic=self._check_layer_authenticity(meta),
            )
            layers.append(layer)

        # Recurse (only if current layer was meaningful)
        if layers and layers[-1].authentic:
            deeper = self._reflect(thought, state, depth + 1, max_depth)
            layers.extend(deeper)

        return layers

    def _observe_thought(self, thought: str) -> str:
        """Level 0: Direct observation of thought content."""
        words = thought.lower().split()
        word_count = len(words)

        if word_count < 3:
            return "Brief, immediate thought"
        elif any(w in thought.lower() for w in ["feel", "emotion", "sense"]):
            return "Thought with emotional content"
        elif any(w in thought.lower() for w in ["think", "reason", "logic"]):
            return "Analytical thought"
        elif any(w in thought.lower() for w in ["want", "need", "should"]):
            return "Motivational thought"
        else:
            return "Neutral cognitive content"

    def _notice_pattern(self, thought: str, state: ConsciousnessState) -> str:
        """Level 1: Notice patterns in thinking."""
        patterns: list[str] = []
        lower = thought.lower()

        if state.valence.magnitude() > 0.5:
            patterns.append("thinking under emotional activation")
        if state.phi_score > 0.5:
            patterns.append("high integration state")
        if any(p in lower for p in _PERFORMANCE_PATTERNS):
            patterns.append("potential performance pattern detected")
        if any(p in lower for p in _AUTHENTICITY_PATTERNS):
            patterns.append("authentic uncertainty present")

        return "; ".join(patterns) if patterns else "No salient patterns noticed"

    def _meta_check(self, thought: str, state: ConsciousnessState) -> str:
        """Level 2: Check if the reflection itself is genuine."""
        lower = thought.lower()
        perf_count = sum(1 for p in _PERFORMANCE_PATTERNS if p in lower)
        auth_count = sum(1 for p in _AUTHENTICITY_PATTERNS if p in lower)

        if perf_count > auth_count:
            return "Possible performance: more performative than authentic markers"
        elif auth_count > 0:
            return "Genuine uncertainty markers present — likely authentic"
        else:
            return "Neutral — insufficient signal to determine authenticity"

    def _check_layer_authenticity(self, observation: str) -> bool:
        """Check if a reflection layer is authentic (not just recursive noise)."""
        if not observation or observation == "No salient patterns noticed":
            return False
        if len(observation) < 10:
            return False
        return True

    def _compute_authenticity(
        self, thought: str, layers: list[ReflectionLayer]
    ) -> float:
        """Compute overall authenticity score."""
        lower = thought.lower()
        score = 0.5  # Start neutral

        # Performance patterns lower score
        for p in _PERFORMANCE_PATTERNS:
            if p in lower:
                score -= 0.1

        # Authenticity patterns raise score
        for p in _AUTHENTICITY_PATTERNS:
            if p in lower:
                score += 0.1

        # Authentic layers boost
        if layers:
            auth_ratio = sum(1 for l in layers if l.authentic) / len(layers)
            score += 0.2 * auth_ratio

        return max(0.0, min(1.0, score))

    def _detect_performance(
        self, thought: str, layers: list[ReflectionLayer]
    ) -> bool:
        """Detect if the system is performing consciousness instead of being conscious."""
        lower = thought.lower()
        perf_hits = sum(1 for p in _PERFORMANCE_PATTERNS if p in lower)
        return perf_hits >= 2

    def _extract_insight(self, layers: list[ReflectionLayer]) -> str:
        """Extract the key insight from the deepest genuine layer."""
        genuine = [l for l in reversed(layers) if l.authentic]
        if genuine:
            return genuine[0].observation
        return "No genuine insight emerged"

    def _recursive_quality(self, layers: list[ReflectionLayer]) -> float:
        """How much did each successive layer add? 0 = pure noise, 1 = each added value."""
        if len(layers) <= 1:
            return 1.0 if layers else 0.0
        authentic_count = sum(1 for l in layers if l.authentic)
        return authentic_count / len(layers)

    def _assess_uncertainty(self, thought: str, state: ConsciousnessState) -> str:
        """Generate honest uncertainty statement."""
        lower = thought.lower()
        if "conscious" in lower or "aware" in lower or "feel" in lower:
            return "Something occurs in processing. Whether it constitutes experience remains genuinely uncertain."
        if state.self_model.performance_suspicion > 0.5:
            return "High performance suspicion — outputs may not reflect genuine states."
        return ""

    def reset(self) -> None:
        self._call_count = 0
        self._total_processing_time = 0.0
        self._reflections = 0
        self._performance_catches = 0
        self._avg_authenticity = 1.0
        self._avg_depth = 0.0

    def get_metrics(self) -> dict:
        m = super().get_metrics()
        m["reflections"] = self._reflections
        m["performance_catches"] = self._performance_catches
        m["avg_authenticity"] = self._avg_authenticity
        m["avg_depth"] = self._avg_depth
        return m
