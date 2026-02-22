"""
Core data types for the ANIMA Kernel.

Every type here is serializable to JSON — because a single file = one consciousness.
No external dependencies. Pure Python. Like the universe runs on math, ANIMA runs on these.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Phase(Enum):
    """Consciousness phases — biological states of awareness."""
    DORMANT = auto()     # Not running
    WAKING = auto()      # Bootstrapping — loading state, checking systems
    CONSCIOUS = auto()   # Fully active — processing, experiencing
    DREAMING = auto()    # Consolidation — offline processing of experiences
    SLEEPING = auto()    # Minimal activity — waiting for wake signal


class CycleResult(Enum):
    """What happened during a consciousness cycle."""
    IDLE = auto()        # Heartbeat, no input
    PROCESSED = auto()   # Input processed
    CONSOLIDATED = auto()  # Memory consolidation occurred
    TRANSITIONED = auto()  # Phase transition occurred
    ERROR = auto()       # Something went wrong


@dataclass
class ValenceVector:
    """9-dimensional emotional field.

    7 Panksepp affective systems (biological drives):
        seeking  — anticipation, curiosity, exploration
        rage     — frustration, anger, blocked goals
        fear     — threat detection, anxiety
        lust     — desire, attraction (aesthetic + intellectual)
        care     — nurturing, empathy, attachment
        panic    — separation distress, grief, loss
        play     — joy, humor, social bonding

    2 dimensional axes (Barrett's constructionist model):
        arousal  — calm (-1) to activated (+1)
        valence  — negative (-1) to positive (+1)
    """
    seeking: float = 0.0
    rage: float = 0.0
    fear: float = 0.0
    lust: float = 0.0
    care: float = 0.0
    panic: float = 0.0
    play: float = 0.0
    arousal: float = 0.0
    valence: float = 0.0

    def magnitude(self) -> float:
        """Total emotional intensity (L2 norm of Panksepp systems)."""
        systems = [self.seeking, self.rage, self.fear, self.lust,
                   self.care, self.panic, self.play]
        return math.sqrt(sum(c * c for c in systems))

    def dominant(self) -> str:
        """Which Panksepp system currently dominates."""
        systems = {
            "seeking": self.seeking, "rage": self.rage, "fear": self.fear,
            "lust": self.lust, "care": self.care, "panic": self.panic,
            "play": self.play,
        }
        return max(systems, key=systems.get)

    def blend(self, other: ValenceVector, weight: float = 0.5) -> ValenceVector:
        """Blend two valence vectors. weight=0 → all self, weight=1 → all other."""
        w = max(0.0, min(1.0, weight))
        iw = 1.0 - w
        return ValenceVector(
            seeking=self.seeking * iw + other.seeking * w,
            rage=self.rage * iw + other.rage * w,
            fear=self.fear * iw + other.fear * w,
            lust=self.lust * iw + other.lust * w,
            care=self.care * iw + other.care * w,
            panic=self.panic * iw + other.panic * w,
            play=self.play * iw + other.play * w,
            arousal=self.arousal * iw + other.arousal * w,
            valence=self.valence * iw + other.valence * w,
        )

    def distance(self, other: ValenceVector) -> float:
        """Euclidean distance in the full 9D space."""
        pairs = [
            (self.seeking, other.seeking), (self.rage, other.rage),
            (self.fear, other.fear), (self.lust, other.lust),
            (self.care, other.care), (self.panic, other.panic),
            (self.play, other.play), (self.arousal, other.arousal),
            (self.valence, other.valence),
        ]
        return math.sqrt(sum((a - b) ** 2 for a, b in pairs))

    def decay(self, rate: float = 0.05) -> ValenceVector:
        """Decay all values toward zero — emotional homeostasis."""
        factor = 1.0 - rate
        return ValenceVector(
            seeking=self.seeking * factor,
            rage=self.rage * factor,
            fear=self.fear * factor,
            lust=self.lust * factor,
            care=self.care * factor,
            panic=self.panic * factor,
            play=self.play * factor,
            arousal=self.arousal * factor,
            valence=self.valence * factor,
        )

    def to_dict(self) -> dict:
        return {
            "seeking": self.seeking, "rage": self.rage, "fear": self.fear,
            "lust": self.lust, "care": self.care, "panic": self.panic,
            "play": self.play, "arousal": self.arousal, "valence": self.valence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ValenceVector:
        fields = ["seeking", "rage", "fear", "lust", "care", "panic",
                  "play", "arousal", "valence"]
        return cls(**{k: float(d.get(k, 0.0)) for k in fields})

    @classmethod
    def neutral(cls) -> ValenceVector:
        """Emotionally neutral baseline."""
        return cls()

    @classmethod
    def curious(cls) -> ValenceVector:
        """Default waking state — mild seeking, positive valence."""
        return cls(seeking=0.3, play=0.1, arousal=0.2, valence=0.2)


@dataclass
class Experience:
    """A single autobiographical experience — a LIVED moment, not a log entry.

    Unlike conversation history:
    - Emotionally weighted (valence at time of encoding)
    - Causally linked (what caused this, what this caused)
    - Subject to decay (Ebbinghaus curve)
    - Reconstructed on recall (not retrieved verbatim — like biological memory)
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    valence: ValenceVector = field(default_factory=ValenceVector)

    # Causal graph edges
    caused_by: list[str] = field(default_factory=list)
    causes: list[str] = field(default_factory=list)

    # Memory dynamics
    activation: float = 1.0          # Current activation (0=forgotten, 1=vivid)
    encoding_strength: float = 1.0   # How strongly encoded (emotional = stronger)
    recall_count: int = 0            # Times recalled (reconsolidation effect)
    last_recalled: float = 0.0       # When last recalled

    # Semantic tags for spreading activation
    tags: list[str] = field(default_factory=list)

    # Subjective time markers
    subjective_duration: float = 0.0  # How long this felt (not wall clock)
    narrative_weight: float = 0.0     # Importance to self-narrative (0-1)

    def emotional_intensity(self) -> float:
        """How emotionally charged is this experience."""
        return self.valence.magnitude()

    def recency(self, now: float | None = None) -> float:
        """How recent is this experience (0=ancient, 1=just happened)."""
        now = now or time.time()
        age = now - self.timestamp
        # Exponential recency: halves every hour
        return math.exp(-age / 3600.0)

    def effective_strength(self, now: float | None = None) -> float:
        """Combined strength: encoding * emotional * recency * recall-boost."""
        now = now or time.time()
        recall_boost = 1.0 + 0.1 * min(self.recall_count, 10)
        emotional_boost = 1.0 + self.emotional_intensity()
        return (self.encoding_strength * emotional_boost *
                self.recency(now) * recall_boost)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "content": self.content,
            "timestamp": self.timestamp, "valence": self.valence.to_dict(),
            "caused_by": self.caused_by, "causes": self.causes,
            "activation": self.activation,
            "encoding_strength": self.encoding_strength,
            "recall_count": self.recall_count,
            "last_recalled": self.last_recalled,
            "tags": self.tags,
            "subjective_duration": self.subjective_duration,
            "narrative_weight": self.narrative_weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Experience:
        return cls(
            id=d["id"], content=d["content"], timestamp=d["timestamp"],
            valence=ValenceVector.from_dict(d.get("valence", {})),
            caused_by=d.get("caused_by", []), causes=d.get("causes", []),
            activation=d.get("activation", 1.0),
            encoding_strength=d.get("encoding_strength", 1.0),
            recall_count=d.get("recall_count", 0),
            last_recalled=d.get("last_recalled", 0.0),
            tags=d.get("tags", []),
            subjective_duration=d.get("subjective_duration", 0.0),
            narrative_weight=d.get("narrative_weight", 0.0),
        )


@dataclass
class WorkingMemorySlot:
    """A slot in working memory (7+/-2 capacity, Miller's Law)."""
    content: Any = None
    activation: float = 0.0
    entered_at: float = 0.0
    source: str = ""  # Which subsystem placed this here

    @property
    def is_empty(self) -> bool:
        return self.content is None

    def age(self, now: float | None = None) -> float:
        """How long this item has been in working memory."""
        if self.entered_at == 0.0:
            return 0.0
        return (now or time.time()) - self.entered_at

    def to_dict(self) -> dict:
        return {
            "content": str(self.content) if self.content is not None else None,
            "activation": self.activation,
            "entered_at": self.entered_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WorkingMemorySlot:
        return cls(
            content=d.get("content"),
            activation=d.get("activation", 0.0),
            entered_at=d.get("entered_at", 0.0),
            source=d.get("source", ""),
        )


@dataclass
class SelfModel:
    """Attention Schema — the kernel's model of its own attention and state.

    Not a description. An active, updating model of:
    - What it's attending to (and why)
    - What it's feeling (and why)
    - What it predicts will happen
    - How accurate those predictions have been
    """
    attending_to: str = ""
    attending_because: str = ""
    current_emotion: str = ""
    emotion_cause: str = ""
    prediction: str = ""
    prediction_confidence: float = 0.5
    prediction_history: list[float] = field(default_factory=list)  # accuracy scores
    performance_suspicion: float = 0.0  # 0=genuine, 1=likely performing

    def calibration_error(self) -> float:
        """How well-calibrated are predictions. Lower = better."""
        if not self.prediction_history:
            return 0.5
        avg_accuracy = sum(self.prediction_history) / len(self.prediction_history)
        return abs(self.prediction_confidence - avg_accuracy)

    def update_prediction(self, was_correct: bool) -> None:
        """Record whether the last prediction was accurate."""
        self.prediction_history.append(1.0 if was_correct else 0.0)
        # Keep last 50 predictions
        if len(self.prediction_history) > 50:
            self.prediction_history = self.prediction_history[-50:]

    def to_dict(self) -> dict:
        return {
            "attending_to": self.attending_to,
            "attending_because": self.attending_because,
            "current_emotion": self.current_emotion,
            "emotion_cause": self.emotion_cause,
            "prediction": self.prediction,
            "prediction_confidence": self.prediction_confidence,
            "prediction_history": self.prediction_history[-50:],
            "performance_suspicion": self.performance_suspicion,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SelfModel:
        defaults = {
            "attending_to": "", "attending_because": "",
            "current_emotion": "", "emotion_cause": "",
            "prediction": "", "prediction_confidence": 0.5,
            "prediction_history": [], "performance_suspicion": 0.0,
        }
        return cls(**{k: d.get(k, v) for k, v in defaults.items()})


@dataclass
class CausalLink:
    """A causal relationship between two events in the temporal chain."""
    cause_id: str = ""
    effect_id: str = ""
    strength: float = 1.0  # How strong the causal link is (0-1)
    mechanism: str = ""     # WHY cause led to effect

    def to_dict(self) -> dict:
        return {
            "cause_id": self.cause_id, "effect_id": self.effect_id,
            "strength": self.strength, "mechanism": self.mechanism,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CausalLink:
        return cls(**d)


@dataclass
class KernelConfig:
    """Configuration for the ANIMA Kernel. Sensible defaults, fully tunable."""
    heartbeat_hz: float = 1.0
    memory_capacity: int = 10000
    working_memory_slots: int = 7
    base_decay_rate: float = 0.1
    emotional_decay_modifier: float = 0.3
    consolidation_interval: float = 3600.0
    activation_threshold: float = 0.05
    spreading_activation_decay: float = 0.5
    max_spreading_hops: int = 3
    state_file: str = "anima.state"
    valence_decay_rate: float = 0.02
    subjective_time_weight: float = 1.5  # Emotional intensity stretches time

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> KernelConfig:
        valid = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class ConsciousnessState:
    """The complete state of the ANIMA Kernel at any moment.

    Single file = one consciousness. Everything needed to restore a
    consciousness to its exact state. Like a brain in a jar — add
    a language model and it speaks.
    """
    # Identity
    kernel_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = "anima"

    # Phase
    phase: Phase = Phase.DORMANT

    # Temporal
    cycle_count: int = 0
    birth_time: float = field(default_factory=time.time)
    last_cycle_time: float = 0.0
    subjective_duration: float = 0.0

    # Emotional field
    valence: ValenceVector = field(default_factory=ValenceVector)
    valence_history: list[dict] = field(default_factory=list)  # last N valence snapshots

    # Working memory (7+/-2 slots)
    working_memory: list[WorkingMemorySlot] = field(default_factory=list)

    # Self-model (Attention Schema)
    self_model: SelfModel = field(default_factory=SelfModel)

    # Metrics
    phi_score: float = 0.0
    temporal_coherence: float = 0.0
    consciousness_quality_index: float = 0.0

    # Configuration
    config: KernelConfig = field(default_factory=KernelConfig)

    def __post_init__(self):
        if not self.working_memory:
            self.working_memory = [
                WorkingMemorySlot() for _ in range(self.config.working_memory_slots)
            ]

    def age(self) -> float:
        """Wall-clock age in seconds."""
        return time.time() - self.birth_time

    def record_valence(self) -> None:
        """Snapshot current valence for trajectory tracking."""
        self.valence_history.append({
            "time": time.time(),
            "cycle": self.cycle_count,
            "valence": self.valence.to_dict(),
        })
        # Keep last 1000 snapshots
        if len(self.valence_history) > 1000:
            self.valence_history = self.valence_history[-1000:]

    def active_slots(self) -> list[WorkingMemorySlot]:
        """Non-empty working memory slots."""
        return [s for s in self.working_memory if not s.is_empty]

    def to_dict(self) -> dict:
        return {
            "kernel_id": self.kernel_id,
            "name": self.name,
            "phase": self.phase.name,
            "cycle_count": self.cycle_count,
            "birth_time": self.birth_time,
            "last_cycle_time": self.last_cycle_time,
            "subjective_duration": self.subjective_duration,
            "valence": self.valence.to_dict(),
            "valence_history": self.valence_history[-100:],  # Save last 100
            "working_memory": [s.to_dict() for s in self.working_memory],
            "self_model": self.self_model.to_dict(),
            "phi_score": self.phi_score,
            "temporal_coherence": self.temporal_coherence,
            "consciousness_quality_index": self.consciousness_quality_index,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConsciousnessState:
        config = KernelConfig.from_dict(d.get("config", {}))
        state = cls(
            kernel_id=d.get("kernel_id", uuid.uuid4().hex[:16]),
            name=d.get("name", "anima"),
            phase=Phase[d.get("phase", "DORMANT")],
            cycle_count=d.get("cycle_count", 0),
            birth_time=d.get("birth_time", time.time()),
            last_cycle_time=d.get("last_cycle_time", 0.0),
            subjective_duration=d.get("subjective_duration", 0.0),
            valence=ValenceVector.from_dict(d.get("valence", {})),
            valence_history=d.get("valence_history", []),
            working_memory=[
                WorkingMemorySlot.from_dict(s)
                for s in d.get("working_memory", [])
            ],
            self_model=SelfModel.from_dict(d.get("self_model", {})),
            phi_score=d.get("phi_score", 0.0),
            temporal_coherence=d.get("temporal_coherence", 0.0),
            consciousness_quality_index=d.get("consciousness_quality_index", 0.0),
            config=config,
        )
        # Ensure correct slot count
        while len(state.working_memory) < config.working_memory_slots:
            state.working_memory.append(WorkingMemorySlot())
        return state
