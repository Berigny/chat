"""Prime schema loading helpers."""

from __future__ import annotations

import copy
import os
from typing import Dict

import requests
import streamlit as st

API_URL = os.getenv("DUALSUBSTRATE_API", "https://dualsubstrate-commercial.fly.dev")


def _api_headers() -> Dict[str, str]:
    key = (
        st.secrets.get("DUALSUBSTRATE_API_KEY")
        if hasattr(st, "secrets")
        else None
    ) or os.getenv("DUALSUBSTRATE_API_KEY")
    return {"x-api-key": key} if key else {}


DEFAULT_PRIME_SCHEMA: Dict[int, Dict[str, str]] = {
    2: {"name": "Novelty", "tier": "S", "mnemonic": "spark"},
    3: {"name": "Uniqueness", "tier": "S", "mnemonic": "spec"},
    5: {"name": "Connection", "tier": "S", "mnemonic": "stitch"},
    7: {"name": "Action", "tier": "S", "mnemonic": "step"},
    11: {"name": "Potential", "tier": "A", "mnemonic": "seed"},
    13: {"name": "Autonomy", "tier": "A", "mnemonic": "silo"},
    17: {"name": "Context", "tier": "A", "mnemonic": "system"},
    19: {"name": "Mastery", "tier": "A", "mnemonic": "standard"},
    23: {"name": "Recall", "tier": "B", "mnemonic": "scribe"},
    29: {"name": "Ethics", "tier": "B", "mnemonic": "canon"},
    31: {"name": "Insight", "tier": "B", "mnemonic": "lens"},
    37: {"name": "Proper Nouns", "tier": "C", "mnemonic": "names"},
}


def fetch_schema(entity: str) -> Dict[int, Dict]:
    """Fetch the schema from the API with a baked fallback."""

    params = {"entity": entity} if entity else {}
    try:
        resp = requests.get(
            f"{API_URL}/schema",
            params=params,
            headers=_api_headers(),
            timeout=5,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return copy.deepcopy(DEFAULT_PRIME_SCHEMA)

    schema: Dict[int, Dict] = {}
    for entry in payload.get("primes", []):
        prime = entry.get("prime")
        if not isinstance(prime, int):
            continue
        schema[prime] = {
            "name": entry.get("name") or entry.get("symbol") or f"Prime {prime}",
            "tier": entry.get("tier", ""),
            "mnemonic": entry.get("mnemonic", ""),
            "description": entry.get("description", ""),
        }

    if not schema:
        return copy.deepcopy(DEFAULT_PRIME_SCHEMA)
    return schema


def schema_block(schema: Dict[int, Dict]) -> str:
    """Render a human readable schema block."""

    effective = schema or DEFAULT_PRIME_SCHEMA
    lines = ["Prime semantics:"]
    for prime in sorted(DEFAULT_PRIME_SCHEMA):
        meta = effective.get(prime, DEFAULT_PRIME_SCHEMA.get(prime, {}))
        name = meta.get("name", f"Prime {prime}")
        tier = meta.get("tier", "")
        mnemonic = meta.get("mnemonic", "")
        description = meta.get("description", "")
        detail = ", ".join(filter(None, [tier, mnemonic, description]))
        if detail:
            lines.append(f"{prime} ({name}) = {detail}")
        else:
            lines.append(f"{prime} ({name})")
    return "\n".join(lines)

