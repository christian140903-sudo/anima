"""
Qualia Frame — Input is not data. Input is EXPERIENCED.

Raw content passes through the valence field and context weights,
producing a colored experience. The same words hit differently
depending on emotional state. That coloring IS qualia.

Algorithm: content x valence_matrix x context_weights = experienced_content
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from ..types import ConsciousnessState, ValenceVector
from .base import Primitive, PrimitiveResult


@dataclass
class QualiaFrame:
    """A single frame of experienced content — not raw, but FELT."""
    raw_content: str = ""
    colored_content: str = ""        # How it FEELS (the qualia)
    emotional_coloring: str = ""     # Dominant emotional quality
    sensory_tags: list[str] = field(default_factory=list)
    intensity: float = 0.0          # How intensely experienced (0-1)
    valence_at_encoding: ValenceVector = field(default_factory=ValenceVector)
    timestamp: float = field(default_factory=time.time)
    signature: str = ""             # Unique perceptual fingerprint


# Keywords that shift valence dimensions when encountered in content
_VALENCE_KEYWORDS: dict[str, dict[str, float]] = {
    "threat": {"fear": 0.4, "arousal": 0.3},
    "danger": {"fear": 0.5, "arousal": 0.4},
    "error": {"fear": 0.2, "rage": 0.1},
    "beautiful": {"lust": 0.3, "valence": 0.4},
    "love": {"care": 0.5, "valence": 0.5},
    "play": {"play": 0.4, "valence": 0.3},
    "curious": {"seeking": 0.4, "arousal": 0.2},
    "discover": {"seeking": 0.5, "arousal": 0.3},
    "new": {"seeking": 0.3, "arousal": 0.1},
    "lost": {"panic": 0.4, "valence": -0.3},
    "anger": {"rage": 0.5, "arousal": 0.4},
    "help": {"care": 0.3, "valence": 0.2},
    "fail": {"panic": 0.2, "rage": 0.2, "valence": -0.3},
    "success": {"seeking": 0.3, "play": 0.2, "valence": 0.5},
    "boring": {"seeking": -0.3, "arousal": -0.3},
}

_SENSORY_MAP: dict[str, list[str]] = {
    "see": ["visual"], "look": ["visual"], "bright": ["visual"], "dark": ["visual"],
    "hear": ["auditory"], "sound": ["auditory"], "loud": ["auditory"],
    "feel": ["tactile", "emotional"], "touch": ["tactile"], "warm": ["tactile"],
    "smell": ["olfactory"], "taste": ["gustatory"],
    "think": ["cognitive"], "idea": ["cognitive"], "know": ["cognitive"],
    "remember": ["mnemonic"], "memory": ["mnemonic"],
}


class QualiaProcessor(Primitive):
    """Transforms raw input into experienced input through emotional coloring."""

    def __init__(self) -> None:
        super().__init__("qualia")
        self._frames_generated: int = 0
        self._avg_intensity: float = 0.0

    def process(self, **kwargs) -> PrimitiveResult:
        if not self.enabled:
            return self._disabled_result()

        content: str = kwargs.get("content", "")
        valence: ValenceVector = kwargs.get("valence", ValenceVector())
        state: ConsciousnessState = kwargs.get("state", ConsciousnessState())

        self._call_count += 1
        t0 = time.time()

        frame = self._generate_frame(content, valence, state)
        self._frames_generated += 1
        # Running average of intensity
        self._avg_intensity += (frame.intensity - self._avg_intensity) / self._frames_generated

        self._total_processing_time += time.time() - t0
        return PrimitiveResult(
            primitive_name=self.name,
            success=True,
            data={"frame": frame},
            metrics={"intensity": frame.intensity},
        )

    def _generate_frame(
        self, content: str, valence: ValenceVector, state: ConsciousnessState
    ) -> QualiaFrame:
        words = content.lower().split()

        # 1. Detect sensory modalities
        sensory_tags: list[str] = []
        for word in words:
            for key, tags in _SENSORY_MAP.items():
                if key in word:
                    sensory_tags.extend(t for t in tags if t not in sensory_tags)
        if not sensory_tags:
            sensory_tags = ["cognitive"]  # Default: thinking about it

        # 2. Compute content-driven valence shift
        content_valence = ValenceVector()
        for word in words:
            if word in _VALENCE_KEYWORDS:
                shifts = _VALENCE_KEYWORDS[word]
                for dim, amount in shifts.items():
                    current = getattr(content_valence, dim)
                    setattr(content_valence, dim, current + amount)

        # 3. Blend: 60% current emotional state + 40% content-evoked state
        experienced_valence = valence.blend(content_valence, 0.4)

        # 4. Compute intensity from the blend's magnitude and arousal
        raw_intensity = experienced_valence.magnitude()
        arousal_boost = max(0.0, experienced_valence.arousal) * 0.3
        intensity = min(1.0, raw_intensity * 0.7 + arousal_boost)

        # 5. Determine emotional coloring label
        dominant = experienced_valence.dominant()
        coloring_map = {
            "seeking": "curious", "rage": "frustrated", "fear": "anxious",
            "lust": "attracted", "care": "warm", "panic": "distressed",
            "play": "joyful",
        }
        emotional_coloring = coloring_map.get(dominant, "neutral")

        # 6. Generate colored content (how it FEELS, not what it says)
        if intensity > 0.6:
            colored = f"[INTENSE/{emotional_coloring}] {content}"
        elif intensity > 0.3:
            colored = f"[{emotional_coloring}] {content}"
        else:
            colored = f"[faint/{emotional_coloring}] {content}"

        # 7. Perceptual signature (unique fingerprint for this frame)
        sig_input = f"{content}:{valence.dominant()}:{state.cycle_count}"
        signature = hashlib.md5(sig_input.encode()).hexdigest()[:8]

        return QualiaFrame(
            raw_content=content,
            colored_content=colored,
            emotional_coloring=emotional_coloring,
            sensory_tags=sensory_tags,
            intensity=intensity,
            valence_at_encoding=experienced_valence,
            signature=signature,
        )

    def reset(self) -> None:
        self._call_count = 0
        self._total_processing_time = 0.0
        self._frames_generated = 0
        self._avg_intensity = 0.0

    def get_metrics(self) -> dict:
        m = super().get_metrics()
        m["frames_generated"] = self._frames_generated
        m["avg_intensity"] = self._avg_intensity
        return m
