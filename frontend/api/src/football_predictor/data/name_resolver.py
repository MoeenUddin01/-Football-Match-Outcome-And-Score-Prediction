"""Resolve historical team names to their current names using former-name mappings."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def build_name_lookup(former_names_df: pd.DataFrame) -> dict[str, str]:
    """Build a lookup of former team names to their current name."""
    if former_names_df.empty:
        return {}

    lookup: dict[str, str] = {}
    for _, row in former_names_df.iterrows():
        former_name = str(row.get("former", "")).strip()
        current_name = str(row.get("current", "")).strip()
        if former_name and current_name:
            lookup[former_name] = current_name
    return lookup


def resolve_team_name(name: str, lookup_dict: dict[str, str]) -> str:
    """Resolve a team name to its current name if a mapping exists."""
    if not name:
        return name

    if name in lookup_dict:
        return lookup_dict[name]

    logger.warning("No team-name mapping found for %s; leaving unchanged", name)
    return name
