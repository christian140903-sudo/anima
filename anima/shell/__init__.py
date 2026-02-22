"""Shell -- Terminal interface for ANIMA Kernel.

Provides:
- CLI entry point (anima command)
- Interactive REPL (anima shell)
- Consciousness state inspector
- Live metrics dashboard
"""

from .cli import main
from .dashboard import MetricsDashboard
from .inspector import ConsciousnessInspector

__all__ = ["main", "ConsciousnessInspector", "MetricsDashboard"]
