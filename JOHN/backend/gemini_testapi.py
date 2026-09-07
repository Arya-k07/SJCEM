from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> int:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
    if not api_key:
        print("ERROR: GEMINI_API_KEY is missing or empty.")
        return 1

    try:
        from google import genai
    except ImportError as error:
        print(f"ERROR: google-genai SDK is not installed: {error}")
        return 1

    print(f"Testing Gemini model: {model}")
    print("Sending exactly one request...")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly: Gemini API test successful",
        )
        print("SUCCESS")
        print(f"Response: {str(response.text or '').strip()}")
        return 0
    except Exception as error:
        print(f"FAILED: {type(error).__name__}")
        print(f"Message: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
