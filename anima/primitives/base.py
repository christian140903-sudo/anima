"""
Base class for all consciousness primitives.

Every primitive is ablation-capable: disable it and measure
what the consciousness LOSES. If nothing changes, the primitive
was dead weight. If something breaks, it was load-bearing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrimitiveResult:
    """Base result returned by any primitive."""
    primitive_name: str = ""
    success: bool = True
    data: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class Primitive(ABC):
    """Abstract base for the 8 consciousness primitives.

    Each primitive:
    - Has a clear process() method
    - Can be enabled/disabled (ablation)
    - Tracks its own metrics
    - Can be reset to initial state
    """

    def __init__(self, name: str):
        self.name: str = name
        self.enabled: bool = True
        self._call_count: int = 0
        self._total_processing_time: float = 0.0

    @abstractmethod
    def process(self, **kwargs: Any) -> PrimitiveResult:
        """Process input and return a result. Signature varies by primitive."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset primitive to initial state."""
        ...

    def get_metrics(self) -> dict[str, Any]:
        """Return primitive-specific metrics."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "call_count": self._call_count,
            "avg_processing_time": (
                self._total_processing_time / self._call_count
                if self._call_count > 0
                else 0.0
            ),
        }

    def _disabled_result(self) -> PrimitiveResult:
        """Return when primitive is disabled (ablation mode)."""
        return PrimitiveResult(
            primitive_name=self.name,
            success=False,
            data={"reason": "primitive_disabled"},
        )
