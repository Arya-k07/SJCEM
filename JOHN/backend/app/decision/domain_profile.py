from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DomainProfile:
    domain_name: str = "generic"
    category_field: str | None = None
    category_mapping: dict[str, str] = field(default_factory=dict)
    terminology: dict[str, str] = field(default_factory=dict)

    def category_for(self, state: dict[str, Any]) -> tuple[str, str]:
        if self.category_field and state.get(self.category_field):
            value = str(state[self.category_field])
            return value, f"Used configured category field '{self.category_field}'."
        text = " ".join(
            [str(state.get("feedback", "")), *map(str, state.get("topics", [])), *map(str, state.get("keywords", []))]
        ).casefold()
        for term, category in self.category_mapping.items():
            if term.casefold() in text:
                return category, f"Matched configured domain term '{term}'."
        return "General", "No configured category match was found."


GENERIC_PROFILE = DomainProfile()


def college_profile() -> DomainProfile:
    return DomainProfile(
        domain_name="college",
        category_field="department",
        category_mapping={
            "wifi": "IT", "wi-fi": "IT", "network": "IT", "internet": "IT",
            "computer": "IT", "pc": "IT", "projector": "IT",
            "classroom equipment": "Facilities", "lab equipment": "Facilities",
            "fans": "Facilities", "ac": "Facilities", "lift": "Facilities",
            "washroom": "Facilities", "maintenance": "Facilities",
            "library": "Library", "placement": "Placement Cell", "career": "Placement Cell",
            "companies": "Placement Cell", "professor": "Academic Affairs",
            "faculty": "Academic Affairs", "teaching quality": "Academic Affairs",
            "attendance": "Academic Affairs", "class": "Academic Affairs",
            "food": "Cafeteria", "canteen": "Cafeteria", "price": "Administration",
            "billing": "Finance", "fee": "Finance",
        },
    )
