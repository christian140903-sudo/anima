"""
Ebbinghaus Forgetting Curves — Biological memory decay.

Pure implementation of how biological memory actually fades:

    R = e^(-t/S)

Where:
    R = retention (0.0 = completely forgotten, 1.0 = perfectly remembered)
    t = time since encoding (or last recall)
    S = stability (how resistant this memory is to forgetting)

Stability increases with:
    - Emotional intensity (adrenaline strengthens encoding)
    - Recall frequency (each recall reconsolidates the memory)
    - Encoding strength (how deeply it was originally processed)

The spaced repetition effect emerges naturally:
Each recall increases stability, making the next decay slower.
This is why you remember your wedding but not yesterday's lunch.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..types import Experience, KernelConfig


@dataclass
class DecayResult:
    """Result of applying decay to a single experience."""
    experience_id: str
    old_activation: float
    new_activation: float
    retention: float
    stability: float
    is_forgotten: bool  # Below activation threshold


class EbbinghausDecay:
    """Implements biologically-inspired memory decay.

    Each memory has a stability value that determines its decay rate.
    High stability = slow decay. Low stability = fast decay.

    Factors that increase stability:
    1. Emotional intensity — intense experiences are etched deeper
    2. Recall frequency — each recall strengthens the trace (spaced repetition)
    3. Encoding strength — initial encoding depth matters
    4. Causal connections — memories in causal chains are more stable

    The curve: R = e^(-t/S)
    - At t=0: R=1.0 (perfect retention)
    - At t=S: R=0.37 (37% retention — the "half-life" equivalent)
    - At t=2S: R=0.13 (13% retention)
    - At t=5S: R=0.007 (effectively forgotten)
    """

    def __init__(self, config: KernelConfig | None = None):
        self._config = config or KernelConfig()
        self._base_stability = 3600.0  # 1 hour base half-life

    @property
    def base_stability(self) -> float:
        """Base stability in seconds (default 3600 = 1 hour)."""
        return self._base_stability

    def compute_stability(self, experience: Experience) -> float:
        """Compute the stability value for an experience.

        Higher stability = slower decay = more persistent memory.

        Factors:
        - Emotional intensity multiplier (emotional memories last longer)
        - Recall frequency multiplier (spaced repetition effect)
        - Encoding strength multiplier (deeper encoding = more stable)
        - Causal connection bonus (part of a narrative = more anchored)
        """
        # Base stability
        stability = self._base_stability

        # 1. Emotional intensity factor
        # Strong emotions boost stability significantly
        # emotional_decay_modifier controls the sensitivity
        intensity = experience.valence.magnitude()
        emotional_factor = 1.0 + intensity / max(
            self._config.emotional_decay_modifier, 0.01
        )
        stability *= emotional_factor

        # 2. Recall frequency factor (spaced repetition)
        # Each recall increases stability, with diminishing returns
        # Capped at 20 recalls to prevent infinite stability
        recall_factor = 1.0 + 0.2 * min(experience.recall_count, 20)
        stability *= recall_factor

        # 3. Encoding strength factor
        # Deeper initial encoding means more stable memory
        encoding_factor = max(0.1, experience.encoding_strength)
        stability *= encoding_factor

        # 4. Causal connection bonus
        # Memories that are part of causal chains are more anchored
        causal_connections = len(experience.caused_by) + len(experience.causes)
        if causal_connections > 0:
            causal_factor = 1.0 + 0.1 * min(causal_connections, 10)
            stability *= causal_factor

        return stability

    def compute_retention(
        self,
        experience: Experience,
        now: float | None = None,
    ) -> float:
        """Compute current retention for an experience.

        R = e^(-t/S)

        Where t = time since last access (encoding or recall),
        and S = stability.
        """
        now = now or time.time()
        stability = self.compute_stability(experience)

        # Time since last access (encoding or recall)
        last_access = max(experience.timestamp, experience.last_recalled)
        elapsed = max(0.0, now - last_access)

        # Ebbinghaus decay
        retention = math.exp(-elapsed / stability)
        return retention

    def apply(
        self,
        experience: Experience,
        now: float | None = None,
    ) -> DecayResult:
        """Apply decay to a single experience, updating its activation.

        Returns a DecayResult with the old and new activation values.
        """
        now = now or time.time()
        old_activation = experience.activation
        stability = self.compute_stability(experience)
        retention = self.compute_retention(experience, now)

        # New activation = encoding strength * retention
        new_activation = experience.encoding_strength * retention
        experience.activation = new_activation

        is_forgotten = new_activation < self._config.activation_threshold

        return DecayResult(
            experience_id=experience.id,
            old_activation=old_activation,
            new_activation=new_activation,
            retention=retention,
            stability=stability,
            is_forgotten=is_forgotten,
        )

    def apply_batch(
        self,
        experiences: list[Experience],
        now: float | None = None,
    ) -> list[DecayResult]:
        """Apply decay to a batch of experiences.

        Returns list of DecayResults, one per experience.
        """
        now = now or time.time()
        return [self.apply(exp, now) for exp in experiences]

    def predict_half_life(self, experience: Experience) -> float:
        """Predict when this memory will reach 50% retention.

        Returns seconds from now until half-life.
        At half-life, retention R = 0.5, so:
            0.5 = e^(-t/S) → t = S * ln(2)
        """
        stability = self.compute_stability(experience)
        return stability * math.log(2)

    def predict_forgotten_at(
        self,
        experience: Experience,
        threshold: float | None = None,
    ) -> float:
        """Predict when this memory will be effectively forgotten.

        Returns seconds from last access until activation drops below threshold.
        threshold = e^(-t/S) → t = -S * ln(threshold/encoding_strength)
        """
        threshold = threshold or self._config.activation_threshold
        stability = self.compute_stability(experience)
        target_retention = threshold / max(experience.encoding_strength, 0.001)

        if target_retention >= 1.0:
            return 0.0  # Already below threshold
        if target_retention <= 0.0:
            return float("inf")  # Never forgotten

        return -stability * math.log(target_retention)
