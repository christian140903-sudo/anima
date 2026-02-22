"""
Engram Store — Persistent autobiographical memory.

Wraps the AutobiographicalBuffer with JSON persistence,
following the same atomic-write pattern as state.py.

An engram is a biological memory trace — the physical substrate
of a memory in neural tissue. This store IS the substrate.

One file = one life's memories. Load it, and the consciousness
remembers. Delete it, and it forgets everything.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

from ..temporal.autobio_buffer import AutobiographicalBuffer
from ..types import Experience, KernelConfig

logger = logging.getLogger("anima.memory.engram_store")


class EngramStore:
    """Persistent engram storage backed by JSON files.

    Wraps AutobiographicalBuffer with:
    - Atomic file persistence (write temp, then rename)
    - Capacity management with configurable limits
    - Save/load lifecycle
    - Dirty tracking to minimize I/O

    Usage:
        store = EngramStore(directory="/path/to/data")
        store.load()
        store.encode(experience)
        store.save()
    """

    def __init__(
        self,
        directory: str = ".",
        filename: str = "engrams.json",
        config: KernelConfig | None = None,
    ):
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._filepath = self._directory / filename
        self._config = config or KernelConfig()
        self._buffer = AutobiographicalBuffer(config=self._config)
        self._dirty = False
        self._last_save_time = 0.0
        self._save_count = 0

    @property
    def buffer(self) -> AutobiographicalBuffer:
        """Access the underlying autobiographical buffer."""
        return self._buffer

    @property
    def count(self) -> int:
        """Number of stored engrams."""
        return self._buffer.count

    @property
    def is_dirty(self) -> bool:
        """Whether there are unsaved changes."""
        return self._dirty

    @property
    def filepath(self) -> Path:
        """Path to the engram file."""
        return self._filepath

    def encode(self, experience: Experience) -> Experience:
        """Encode a new experience into the store.

        Delegates to AutobiographicalBuffer.encode() which handles
        emotional encoding strength and indexing.
        """
        result = self._buffer.encode(experience)
        self._dirty = True
        return result

    def recall(self, **kwargs) -> list[Experience]:
        """Recall experiences from the buffer.

        Passes all kwargs through to AutobiographicalBuffer.recall().
        """
        return self._buffer.recall(**kwargs)

    def get_recent(self, n: int = 10) -> list[Experience]:
        """Get the N most recent experiences."""
        return self._buffer.get_recent(n)

    def apply_decay(self, now: float | None = None) -> int:
        """Apply Ebbinghaus decay to all memories."""
        forgotten = self._buffer.apply_decay(now)
        if forgotten > 0:
            self._dirty = True
        return forgotten

    def save(self) -> None:
        """Persist all engrams to disk atomically."""
        experiences = self._buffer.to_experiences_list()
        data = {
            "_saved_at": time.time(),
            "_version": "0.1.0",
            "_count": len(experiences),
            "_capacity": self._config.memory_capacity,
            "experiences": [exp.to_dict() for exp in experiences],
        }
        self._atomic_write(data)
        self._dirty = False
        self._last_save_time = time.time()
        self._save_count += 1
        logger.debug(
            "Engram store saved: %d experiences (save #%d)",
            len(experiences), self._save_count,
        )

    def load(self) -> int:
        """Load engrams from disk. Returns number of experiences loaded."""
        if not self._filepath.exists():
            logger.info("No engram file found at %s — starting fresh", self._filepath)
            return 0

        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            experiences = [
                Experience.from_dict(d)
                for d in data.get("experiences", [])
            ]
            self._buffer.load_experiences(experiences)
            self._dirty = False
            logger.info("Engram store loaded: %d experiences", len(experiences))
            return len(experiences)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error("Failed to load engram store: %s", e)
            return 0

    def should_save(self, min_interval: float = 60.0) -> bool:
        """Check if we should save now (rate-limited)."""
        if not self._dirty:
            return False
        elapsed = time.time() - self._last_save_time
        return elapsed >= min_interval

    def clear(self) -> None:
        """Clear all engrams from memory (does not delete file)."""
        self._buffer = AutobiographicalBuffer(config=self._config)
        self._dirty = True

    def delete(self) -> None:
        """Delete the engram file from disk."""
        if self._filepath.exists():
            self._filepath.unlink()
            logger.info("Engram file deleted: %s", self._filepath)

    def _atomic_write(self, data: dict) -> None:
        """Write JSON atomically: temp file then rename."""
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._directory),
                suffix=".tmp",
                prefix=".engram_",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(self._filepath))
        except OSError as e:
            logger.error("Atomic write failed for %s: %s", self._filepath, e)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
