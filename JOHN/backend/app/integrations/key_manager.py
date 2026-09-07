from __future__ import annotations

import threading
from dataclasses import dataclass

from app.config import gemini_api_keys


@dataclass(frozen=True)
class GeminiKeySlot:
    slot: int
    key: str


class GeminiKeyManager:
    """Thread-safe deterministic round-robin selector for Gemini API keys."""

    def __init__(self, keys: list[str] | None = None) -> None:
        self._keys = [key.strip() for key in (keys if keys is not None else gemini_api_keys()) if key and key.strip()]
        self._next_index = 0
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        return len(self._keys)

    def next_slot(self) -> GeminiKeySlot | None:
        with self._lock:
            if not self._keys:
                return None
            index = self._next_index
            self._next_index = (self._next_index + 1) % len(self._keys)
            return GeminiKeySlot(index + 1, self._keys[index])


_manager: GeminiKeyManager | None = None
_manager_lock = threading.Lock()


def get_gemini_key_manager() -> GeminiKeyManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = GeminiKeyManager()
        return _manager
