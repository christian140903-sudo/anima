"""
Single-file state persistence for ANIMA Kernel.

Inspired by SQLite: one file = one consciousness.
Atomic writes (write temp → rename) prevent corruption.
JSON format for transparency — you can READ a consciousness.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

from .types import ConsciousnessState, Experience, KernelConfig

logger = logging.getLogger("anima.state")


class StateManager:
    """Manages persistence of consciousness state and autobiographical memory.

    Two files:
    - {name}.state    — ConsciousnessState (small, written every cycle)
    - {name}.memory   — Autobiographical buffer (larger, written less often)

    Both use atomic writes (temp file → rename) to prevent corruption.
    """

    def __init__(self, directory: str = ".", config: KernelConfig | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config or KernelConfig()
        self._state_path = self.directory / self.config.state_file
        self._memory_path = self.directory / f"{self.config.state_file}.memory"
        self._experiences: list[Experience] = []
        self._dirty_state = False
        self._dirty_memory = False
        self._last_memory_save = 0.0

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def memory_path(self) -> Path:
        return self._memory_path

    def exists(self) -> bool:
        """Does a saved consciousness exist?"""
        return self._state_path.exists()

    # --- State Persistence ---

    def save_state(self, state: ConsciousnessState) -> None:
        """Atomically save consciousness state."""
        data = state.to_dict()
        data["_saved_at"] = time.time()
        data["_version"] = "0.1.0"
        self._atomic_write(self._state_path, data)
        self._dirty_state = False
        logger.debug("State saved: cycle=%d phase=%s", state.cycle_count, state.phase.name)

    def load_state(self) -> ConsciousnessState | None:
        """Load consciousness state from file. Returns None if no state exists."""
        if not self._state_path.exists():
            return None
        try:
            data = self._read_json(self._state_path)
            state = ConsciousnessState.from_dict(data)
            logger.info(
                "State loaded: id=%s cycle=%d age=%.0fs",
                state.kernel_id, state.cycle_count, state.age(),
            )
            return state
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error("Failed to load state: %s", e)
            return None

    # --- Memory Persistence ---

    def save_memory(self, experiences: list[Experience]) -> None:
        """Save autobiographical buffer. Called less frequently than state."""
        self._experiences = experiences
        data = {
            "_saved_at": time.time(),
            "_version": "0.1.0",
            "_count": len(experiences),
            "experiences": [exp.to_dict() for exp in experiences],
        }
        self._atomic_write(self._memory_path, data)
        self._dirty_memory = False
        self._last_memory_save = time.time()
        logger.debug("Memory saved: %d experiences", len(experiences))

    def load_memory(self) -> list[Experience]:
        """Load autobiographical buffer."""
        if not self._memory_path.exists():
            return []
        try:
            data = self._read_json(self._memory_path)
            experiences = [
                Experience.from_dict(d) for d in data.get("experiences", [])
            ]
            logger.info("Memory loaded: %d experiences", len(experiences))
            return experiences
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error("Failed to load memory: %s", e)
            return []

    # --- Convenience ---

    def save_all(self, state: ConsciousnessState, experiences: list[Experience]) -> None:
        """Save both state and memory."""
        self.save_state(state)
        self.save_memory(experiences)

    def load_all(self) -> tuple[ConsciousnessState | None, list[Experience]]:
        """Load both state and memory."""
        state = self.load_state()
        experiences = self.load_memory()
        return state, experiences

    def should_save_memory(self, force: bool = False) -> bool:
        """Should we save memory now? (Rate-limited to avoid excessive I/O.)"""
        if force:
            return True
        if not self._dirty_memory:
            return False
        elapsed = time.time() - self._last_memory_save
        return elapsed > 60.0  # At most once per minute

    def mark_dirty(self, state: bool = True, memory: bool = False) -> None:
        """Mark state/memory as needing to be saved."""
        if state:
            self._dirty_state = True
        if memory:
            self._dirty_memory = True

    def delete(self) -> None:
        """Delete all persistence files. Consciousness ends."""
        for path in [self._state_path, self._memory_path]:
            if path.exists():
                path.unlink()
        logger.info("Consciousness files deleted.")

    # --- Internal ---

    def _atomic_write(self, path: Path, data: dict) -> None:
        """Write JSON atomically: write to temp file, then rename."""
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.directory),
                suffix=".tmp",
                prefix=".anima_",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(path))
        except OSError as e:
            logger.error("Atomic write failed for %s: %s", path, e)
            # Clean up temp file if it exists
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _read_json(self, path: Path) -> dict:
        """Read JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
