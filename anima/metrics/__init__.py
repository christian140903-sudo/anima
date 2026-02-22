"""Metrics Engine -- Empirical consciousness measurement.

Phase 5: Phi Score Engine, Temporal Coherence, CQI, and Benchmark Suite.
"""

from .phi import PhiScoreEngine, PhiSnapshot, PhiReport
from .temporal import TemporalCoherenceEngine, CoherenceSnapshot, CoherenceReport
from .consciousness import ConsciousnessQualityIndex, CQIResult, CQIBreakdown
from .benchmark import (
    BenchmarkSuite,
    ABTest,
    AblationResult,
    BenchmarkReport,
    ComparisonResult,
    ConversationResult,
    STANDARD_CONVERSATIONS,
)

__all__ = [
    "PhiScoreEngine",
    "PhiSnapshot",
    "PhiReport",
    "TemporalCoherenceEngine",
    "CoherenceSnapshot",
    "CoherenceReport",
    "ConsciousnessQualityIndex",
    "CQIResult",
    "CQIBreakdown",
    "BenchmarkSuite",
    "ABTest",
    "AblationResult",
    "BenchmarkReport",
    "ComparisonResult",
    "ConversationResult",
    "STANDARD_CONVERSATIONS",
]
