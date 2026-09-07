from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def gemini_api_keys() -> list[str]:
    values = [
        os.getenv("GEMINI_API_KEY"),
        *(os.getenv(f"GEMINI_API_KEY_{index}") for index in range(2, 8)),
    ]
    values.extend(os.getenv("GEMINI_API_KEYS", "").split(","))
    return [value.strip() for value in values if value and value.strip()]


def gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
