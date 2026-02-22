"""
Nexus — Working Memory (7+/-2 slots).

GWT-conformant: items compete for limited broadcast slots.
Slot decay, chunking, priority management. Only the most
activated items survive in the global workspace.

This is the bottleneck that creates consciousness:
without limited capacity, there's no selection, no attention,
no experience. Just noise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..types import ConsciousnessState, WorkingMemorySlot
from .base import Primitive, PrimitiveResult


@dataclass
class NexusItem:
    """An item competing for working memory access."""
    content: str = ""
    activation: float = 0.0
    source: str = ""         # Which subsystem submitted this
    priority: float = 0.0    # Explicit priority boost
    chunk_id: str = ""       # Items with same chunk_id can merge
    entered_at: float = field(default_factory=time.time)


@dataclass
class NexusResult:
    """Result of nexus processing."""
    accepted: bool = False           # Was the item admitted to working memory?
    slot_index: int = -1             # Which slot it went into (-1 = rejected)
    evicted: str | None = None       # Content that was evicted to make room
    current_contents: list[str] = field(default_factory=list)
    occupancy: float = 0.0          # Fraction of slots filled (0-1)
    strongest: str = ""             # Most activated item currently


class NexusProcessor(Primitive):
    """Working memory manager — 7+/-2 competitive slots."""

    def __init__(self, capacity: int = 7) -> None:
        super().__init__("nexus")
        self.capacity: int = capacity
        self._slots: list[NexusItem | None] = [None] * capacity
        self._decay_rate: float = 0.05  # Activation decay per process call
        self._admissions: int = 0
        self._evictions: int = 0
        self._rejections: int = 0

    def process(self, **kwargs) -> PrimitiveResult:
        if not self.enabled:
            return self._disabled_result()

        content: str = kwargs.get("content", "")
        activation: float = kwargs.get("activation", 0.5)
        source: str = kwargs.get("source", "input")
        priority: float = kwargs.get("priority", 0.0)
        chunk_id: str = kwargs.get("chunk_id", "")
        state: ConsciousnessState = kwargs.get("state", ConsciousnessState())

        self._call_count += 1
        t0 = time.time()

        # Step 1: Decay all existing slots
        self._apply_decay()

        # Step 2: Try chunking first (merge with existing related item)
        if chunk_id:
            merged = self._try_chunk(content, activation, chunk_id)
            if merged is not None:
                result = self._build_result(True, merged)
                self._total_processing_time += time.time() - t0
                return PrimitiveResult(
                    primitive_name=self.name, success=True,
                    data={"nexus": result},
                    metrics={"occupancy": result.occupancy},
                )

        # Step 3: Compute effective activation
        effective = activation + priority * 0.3

        # Step 4: Find a slot (empty or weakest)
        item = NexusItem(
            content=content, activation=effective, source=source,
            priority=priority, chunk_id=chunk_id,
        )

        slot_idx, evicted_content = self._compete_for_slot(item)

        if slot_idx >= 0:
            self._admissions += 1
            if evicted_content is not None:
                self._evictions += 1
        else:
            self._rejections += 1

        result = self._build_result(slot_idx >= 0, slot_idx, evicted_content)

        self._total_processing_time += time.time() - t0
        return PrimitiveResult(
            primitive_name=self.name, success=True,
            data={"nexus": result},
            metrics={"occupancy": result.occupancy, "accepted": float(slot_idx >= 0)},
        )

    def _apply_decay(self) -> None:
        """All slots decay over time. Empty slots below threshold are cleared."""
        for i, slot in enumerate(self._slots):
            if slot is not None:
                slot.activation *= (1.0 - self._decay_rate)
                if slot.activation < 0.01:
                    self._slots[i] = None

    def _try_chunk(self, content: str, activation: float, chunk_id: str) -> int | None:
        """Try to merge with an existing item sharing the same chunk_id."""
        for i, slot in enumerate(self._slots):
            if slot is not None and slot.chunk_id == chunk_id:
                # Chunk: merge content and boost activation
                slot.content = f"{slot.content} + {content}"
                slot.activation = min(1.0, slot.activation + activation * 0.5)
                return i
        return None

    def _compete_for_slot(self, item: NexusItem) -> tuple[int, str | None]:
        """GWT competition: item tries to enter the global workspace.

        Returns (slot_index, evicted_content). slot_index=-1 if rejected.
        """
        # First: look for empty slot
        for i, slot in enumerate(self._slots):
            if slot is None:
                self._slots[i] = item
                return i, None

        # All slots full: find weakest
        weakest_idx = 0
        weakest_act = self._slots[0].activation if self._slots[0] else float("inf")
        for i, slot in enumerate(self._slots):
            if slot is not None and slot.activation < weakest_act:
                weakest_act = slot.activation
                weakest_idx = i

        # Can the new item beat the weakest?
        if item.activation > weakest_act:
            evicted = self._slots[weakest_idx].content if self._slots[weakest_idx] else None
            self._slots[weakest_idx] = item
            return weakest_idx, evicted

        # Rejected: not strong enough
        return -1, None

    def _build_result(
        self, accepted: bool, slot_index: int = -1, evicted: str | None = None
    ) -> NexusResult:
        """Build result from current state."""
        contents = []
        strongest = ""
        max_act = -1.0
        filled = 0
        for slot in self._slots:
            if slot is not None:
                contents.append(slot.content)
                filled += 1
                if slot.activation > max_act:
                    max_act = slot.activation
                    strongest = slot.content

        return NexusResult(
            accepted=accepted,
            slot_index=slot_index,
            evicted=evicted,
            current_contents=contents,
            occupancy=filled / self.capacity,
            strongest=strongest,
        )

    def get_contents(self) -> list[str]:
        """Get all current working memory contents."""
        return [s.content for s in self._slots if s is not None]

    def clear(self) -> None:
        """Clear all slots."""
        self._slots = [None] * self.capacity

    def reset(self) -> None:
        self._call_count = 0
        self._total_processing_time = 0.0
        self.clear()
        self._admissions = 0
        self._evictions = 0
        self._rejections = 0

    def get_metrics(self) -> dict:
        m = super().get_metrics()
        filled = sum(1 for s in self._slots if s is not None)
        m["capacity"] = self.capacity
        m["filled"] = filled
        m["occupancy"] = filled / self.capacity
        m["admissions"] = self._admissions
        m["evictions"] = self._evictions
        m["rejections"] = self._rejections
        return m
