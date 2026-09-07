from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app.config import gemini_api_keys
from app.integrations.key_manager import GeminiKeyManager


def test_round_robin_order_and_missing_keys() -> None:
    manager = GeminiKeyManager(["k1", "", "k3", " ", "k5", "k6", "k7"])
    assert [manager.next_slot().slot for _ in range(9)] == [1, 2, 3, 4, 5, 1, 2, 3, 4]
    assert manager.count == 5


def test_existing_environment_has_seven_ordered_slots() -> None:
    keys = gemini_api_keys()
    assert len(keys) >= 7
    assert all(keys)


def test_concurrent_access_preserves_round_robin_multiset() -> None:
    manager = GeminiKeyManager([f"k{index}" for index in range(1, 8)])
    with ThreadPoolExecutor(max_workers=16) as executor:
        slots = list(executor.map(lambda _: manager.next_slot().slot, range(70)))
    assert sorted(slots) == sorted(list(range(1, 8)) * 10)
