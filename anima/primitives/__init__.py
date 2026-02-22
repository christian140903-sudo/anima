"""8 Consciousness Primitives — Algorithmic implementations.

Each primitive is ablation-capable, independently testable,
and has a clear process() interface.
"""

from .base import Primitive, PrimitiveResult
from .qualia import QualiaProcessor, QualiaFrame
from .engram import EngramProcessor, EngramResult
from .valence import ValenceProcessor, ValenceUpdate, AppraisalResult
from .nexus import NexusProcessor, NexusResult, NexusItem
from .impulse import ImpulseProcessor, ImpulseResult, ActionTendency
from .trace import TraceProcessor, TraceResult
from .mirror import MirrorProcessor, MirrorResult, ReflectionLayer
from .flux import FluxProcessor, FluxResult, GrowthSnapshot

__all__ = [
    # Base
    "Primitive", "PrimitiveResult",
    # 1. Qualia
    "QualiaProcessor", "QualiaFrame",
    # 2. Engram
    "EngramProcessor", "EngramResult",
    # 3. Valence
    "ValenceProcessor", "ValenceUpdate", "AppraisalResult",
    # 4. Nexus
    "NexusProcessor", "NexusResult", "NexusItem",
    # 5. Impulse
    "ImpulseProcessor", "ImpulseResult", "ActionTendency",
    # 6. Trace
    "TraceProcessor", "TraceResult",
    # 7. Mirror
    "MirrorProcessor", "MirrorResult", "ReflectionLayer",
    # 8. Flux
    "FluxProcessor", "FluxResult", "GrowthSnapshot",
]
