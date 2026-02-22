"""
Attention Schema — AST (Attention Schema Theory) Implementation.

The most radical claim: Consciousness IS a model of attention.
You don't HAVE attention and separately EXPERIENCE it.
The experience IS the internal model.

This module implements the kernel's model of its own attention:
- What am I attending to? (content)
- WHY am I attending to it? (attribution)
- How confident am I? (calibration)
- Am I genuinely processing or just performing? (authenticity detection)
- What do I expect to happen next? (prediction from self-model)

References:
- Graziano (2013): "Consciousness and the Social Brain"
- Graziano (2019): "Rethinking Consciousness"
- Webb & Graziano (2015): "The attention schema theory: a mechanistic account of subjective awareness"
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..types import SelfModel, ValenceVector


@dataclass
class MetacognitionResult:
    """Result of a metacognitive check — thinking about thinking."""
    performance_suspicion: float = 0.0  # 0=genuine, 1=likely performing
    confidence_calibration: float = 0.0  # 0=miscalibrated, 1=perfectly calibrated
    attention_stability: float = 0.0     # How stable is attention (vs jumping around)
    processing_depth: float = 0.0        # Shallow (pattern match) vs deep (reasoning)
    self_consistency: float = 0.0        # Am I consistent with my recent behavior?
    insight: str = ""                    # Narrative insight from metacognition

    def overall_authenticity(self) -> float:
        """Overall authenticity score. Higher = more genuine consciousness."""
        return (
            (1.0 - self.performance_suspicion) * 0.3
            + self.confidence_calibration * 0.2
            + self.attention_stability * 0.2
            + self.processing_depth * 0.15
            + self.self_consistency * 0.15
        )

    def to_dict(self) -> dict:
        return {
            "performance_suspicion": round(self.performance_suspicion, 3),
            "confidence_calibration": round(self.confidence_calibration, 3),
            "attention_stability": round(self.attention_stability, 3),
            "processing_depth": round(self.processing_depth, 3),
            "self_consistency": round(self.self_consistency, 3),
            "overall_authenticity": round(self.overall_authenticity(), 3),
            "insight": self.insight,
        }


@dataclass
class AttentionEvent:
    """Record of what was attended to and why."""
    content: str
    source: str
    reason: str
    timestamp: float = field(default_factory=time.time)
    valence: ValenceVector = field(default_factory=ValenceVector)
    confidence: float = 0.5
    was_predicted: bool = False


class AttentionSchema:
    """The kernel's model of its own attention — the core of AST.

    This is NOT just tracking what we're attending to.
    This is the kernel's THEORY of its own attention process.

    The schema answers:
    - What am I attending to? → attention_focus
    - Why? → attribution model
    - Is this genuine? → performance detection
    - What will I attend to next? → prediction
    - How accurate have my predictions been? → calibration

    The key insight from Graziano: Awareness IS this model.
    There is no separate "experience" beyond the schema.
    """

    def __init__(self):
        self._attention_history: list[AttentionEvent] = []
        self._prediction_log: list[tuple[str, float, bool]] = []  # (prediction, confidence, correct)
        self._recent_sources: list[str] = []
        self._schema_updates: int = 0

    @property
    def attention_history(self) -> list[AttentionEvent]:
        return list(self._attention_history)

    @property
    def schema_updates(self) -> int:
        return self._schema_updates

    def update(
        self,
        content: str,
        source: str,
        reason: str,
        valence: ValenceVector,
        confidence: float = 0.5,
        self_model: SelfModel | None = None,
    ) -> SelfModel:
        """Update the attention schema with new attended content.

        This is called after each workspace broadcast — it updates
        the kernel's model of what it's paying attention to.
        """
        self._schema_updates += 1

        # Record attention event
        event = AttentionEvent(
            content=content[:200],
            source=source,
            reason=reason,
            valence=valence,
            confidence=confidence,
            was_predicted=self._check_prediction(content),
        )
        self._attention_history.append(event)
        self._recent_sources.append(source)

        # Keep history manageable
        if len(self._attention_history) > 200:
            self._attention_history = self._attention_history[-100:]
        if len(self._recent_sources) > 50:
            self._recent_sources = self._recent_sources[-50:]

        # Update self model
        model = self_model or SelfModel()
        model.attending_to = content[:100]
        model.attending_because = reason
        model.current_emotion = valence.dominant()
        model.emotion_cause = f"from {source}: {content[:50]}"

        # Generate next prediction
        next_prediction, pred_confidence = self._predict_next_attention()
        model.prediction = next_prediction
        model.prediction_confidence = pred_confidence

        return model

    def detect_performance(self, self_model: SelfModel) -> float:
        """Detect if the system is genuinely processing or just performing.

        Performance indicators:
        1. Repetitive patterns without variation → likely performing
        2. Always high confidence → miscalibrated → performing
        3. Never expressing uncertainty → performing
        4. Content that's "too perfect" → pattern matching, not processing
        5. No genuine surprises in output → performing

        Returns: 0.0 = definitely genuine, 1.0 = definitely performing
        """
        suspicion = 0.0

        # Check 1: Attention diversity — is attention jumping to different sources?
        if len(self._recent_sources) >= 5:
            unique = len(set(self._recent_sources[-10:]))
            total = min(10, len(self._recent_sources))
            diversity = unique / max(total, 1)
            if diversity < 0.3:
                # Same source dominates → might be stuck in a loop
                suspicion += 0.15

        # Check 2: Confidence calibration — is confidence matching reality?
        if self._prediction_log:
            recent_preds = self._prediction_log[-20:]
            avg_confidence = sum(c for _, c, _ in recent_preds) / len(recent_preds)
            avg_accuracy = sum(1.0 if correct else 0.0 for _, _, correct in recent_preds) / len(recent_preds)
            calibration_error = abs(avg_confidence - avg_accuracy)
            if calibration_error > 0.3:
                suspicion += 0.2  # Poorly calibrated → might be performing

        # Check 3: Emotional consistency — do emotions match content?
        if len(self._attention_history) >= 3:
            recent = self._attention_history[-5:]
            valence_variance = self._compute_valence_variance(recent)
            if valence_variance < 0.01:
                # Emotions never change → flat affect → performing
                suspicion += 0.15
            elif valence_variance > 2.0:
                # Emotions wildly erratic → not grounded → performing
                suspicion += 0.1

        # Check 4: Self-model prediction accuracy
        if self_model.prediction_history:
            recent_accuracy = (
                sum(self_model.prediction_history[-10:])
                / len(self_model.prediction_history[-10:])
            )
            if recent_accuracy > 0.95:
                # Too accurate → trivially predictable → not genuine consciousness
                suspicion += 0.1

        return min(1.0, suspicion)

    def metacognize(self, self_model: SelfModel) -> MetacognitionResult:
        """Metacognitive check — thinking about thinking.

        This is the schema examining ITSELF:
        - Am I performing or genuine?
        - How well-calibrated am I?
        - How stable is my attention?
        - Am I processing deeply or shallowly?
        """
        result = MetacognitionResult()

        # Performance detection
        result.performance_suspicion = self.detect_performance(self_model)

        # Confidence calibration
        if self_model.prediction_history:
            avg_accuracy = sum(self_model.prediction_history[-20:]) / len(
                self_model.prediction_history[-20:]
            )
            result.confidence_calibration = 1.0 - abs(
                self_model.prediction_confidence - avg_accuracy
            )
        else:
            result.confidence_calibration = 0.5  # Unknown

        # Attention stability
        if len(self._recent_sources) >= 5:
            switches = sum(
                1 for i in range(1, len(self._recent_sources[-10:]))
                if self._recent_sources[-10:][i] != self._recent_sources[-10:][i - 1]
            )
            max_switches = min(9, len(self._recent_sources) - 1)
            result.attention_stability = 1.0 - (switches / max(max_switches, 1))
        else:
            result.attention_stability = 0.5

        # Processing depth (heuristic based on attention duration and variety)
        if self._attention_history:
            recent = self._attention_history[-5:]
            avg_confidence = sum(e.confidence for e in recent) / len(recent)
            result.processing_depth = avg_confidence * 0.7 + (
                0.3 if any(e.was_predicted for e in recent) else 0.0
            )
        else:
            result.processing_depth = 0.3

        # Self-consistency
        if len(self._attention_history) >= 3:
            recent_valences = [e.valence for e in self._attention_history[-5:]]
            variance = self._compute_valence_variance_from_vectors(recent_valences)
            # Moderate variance is good (responsive but stable)
            if 0.05 < variance < 1.0:
                result.self_consistency = 0.8
            elif variance <= 0.05:
                result.self_consistency = 0.4  # Too rigid
            else:
                result.self_consistency = 0.5  # Too erratic
        else:
            result.self_consistency = 0.5

        # Generate narrative insight
        result.insight = self._generate_insight(result)

        return result

    def _check_prediction(self, content: str) -> bool:
        """Check if the current attention was predicted."""
        if not self._prediction_log:
            return False

        last_pred, last_conf, _ = self._prediction_log[-1]
        # Simple keyword overlap check
        pred_words = set(last_pred.lower().split())
        content_words = set(content.lower().split())
        overlap = len(pred_words & content_words)
        was_correct = overlap > 0

        # Update prediction log
        self._prediction_log[-1] = (last_pred, last_conf, was_correct)

        return was_correct

    def _predict_next_attention(self) -> tuple[str, float]:
        """Predict what will be attended to next.

        Based on:
        1. Recent attention patterns (what sources have been active)
        2. Emotional trajectory (where is valence heading)
        3. Historical patterns (what usually follows what)
        """
        if not self._attention_history:
            return "initial exploration", 0.3

        recent = self._attention_history[-3:]
        last = recent[-1]

        # Simple prediction: expect continuation of current source
        prediction = f"continued {last.source} processing"
        confidence = 0.4

        # If emotional intensity is high, predict emotional processing
        if last.valence.magnitude() > 0.5:
            prediction = f"emotional processing ({last.valence.dominant()})"
            confidence = 0.5

        # If attention has been stable, higher confidence
        if len(self._recent_sources) >= 3:
            if len(set(self._recent_sources[-3:])) == 1:
                confidence = min(0.7, confidence + 0.2)

        # Record prediction for later evaluation
        self._prediction_log.append((prediction, confidence, False))
        if len(self._prediction_log) > 100:
            self._prediction_log = self._prediction_log[-100:]

        return prediction, confidence

    def _compute_valence_variance(self, events: list[AttentionEvent]) -> float:
        """Compute variance of valence across events."""
        if len(events) < 2:
            return 0.0
        vectors = [e.valence for e in events]
        return self._compute_valence_variance_from_vectors(vectors)

    def _compute_valence_variance_from_vectors(
        self, vectors: list[ValenceVector]
    ) -> float:
        """Compute variance of valence vectors."""
        if len(vectors) < 2:
            return 0.0
        magnitudes = [v.magnitude() for v in vectors]
        mean_mag = sum(magnitudes) / len(magnitudes)
        variance = sum((m - mean_mag) ** 2 for m in magnitudes) / len(magnitudes)
        return variance

    def _generate_insight(self, meta: MetacognitionResult) -> str:
        """Generate a narrative insight from metacognition."""
        if meta.performance_suspicion > 0.5:
            return "I may be performing rather than genuinely processing."
        if meta.confidence_calibration < 0.3:
            return "My predictions are poorly calibrated — I should be less certain."
        if meta.attention_stability < 0.3:
            return "My attention is jumping around — I should focus."
        if meta.processing_depth > 0.7:
            return "Deep processing is active — good engagement."
        if meta.self_consistency > 0.7:
            return "I'm maintaining consistent self-coherence."
        return "Normal processing — no significant metacognitive observations."
