from __future__ import annotations

import os
from typing import Any


class SupabaseDecisionStore:
    """Isolated persistence adapter; graph execution does not require Supabase."""

    def __init__(self, client: Any | None = None) -> None:
        self.client = client
        if self.client is None and os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
            from supabase import create_client
            self.client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    def save_decision(self, decision: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            return {"status": "NOT_CONFIGURED"}
        result = self.client.table("decisions").insert(decision).execute()
        return {"status": "PERSISTED", "data": result.data}
