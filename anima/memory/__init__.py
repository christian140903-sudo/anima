"""Memory Engine — Biological memory with spreading activation.

Not a vector database. Not RAG. A biological memory system:
- Spreading activation for associative retrieval
- Ebbinghaus forgetting curves for natural decay
- Schema formation through consolidation
- Persistent engram storage
"""

from .activation import ActivationEdge, ActivationNode, SpreadingActivationNetwork
from .consolidation import ConsolidationReport, MemoryConsolidationEngine, Schema
from .decay import DecayResult, EbbinghausDecay
from .engram_store import EngramStore

__all__ = [
    "EngramStore",
    "SpreadingActivationNetwork",
    "ActivationNode",
    "ActivationEdge",
    "EbbinghausDecay",
    "DecayResult",
    "MemoryConsolidationEngine",
    "ConsolidationReport",
    "Schema",
]
