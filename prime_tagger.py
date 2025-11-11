"""Deterministic prime tagging based on keyword heuristics."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

from prime_schema import DEFAULT_PRIME_SCHEMA


KEYWORD_MAP: Dict[int, Iterable[str]] = {
    2: {"i", "we", "our", "team", "customer", "client"},
    3: {"met", "meet", "meeting", "call", "discuss", "plan", "review"},
    5: {"with", "for", "about", "regarding", "together"},
    7: {"at", "in", "office", "room", "zoom", "campus"},
    11: {"am", "pm", "today", "tomorrow", "yesterday"},
    13: {"to", "so that", "in order", "so we can", "ensure"},
    17: {"because", "blocked", "blocker", "risk", "issue", "ethics"},
    19: {"excited", "worried", "happy", "frustrated", "concerned"},
    23: {"remember", "recall", "remind"},
    29: {"moral", "virtue"},
    31: {"insight", "lesson", "learned"},
    37: set(),
}

PROPER_NOUN_PATTERN = re.compile(r"\b[A-Z][a-z]+\b")
TIME_PATTERN = re.compile(r"\b\d{1,2}\s*(?:am|pm)\b")


def _schema_filter(schema: Dict[int, Dict]) -> Dict[int, Dict]:
    if not schema:
        return DEFAULT_PRIME_SCHEMA
    return schema


def tag_primes(text: str, schema: Dict[int, Dict]) -> List[int]:
    """Return the deterministic list of primes present in text."""

    if not text:
        return [min(DEFAULT_PRIME_SCHEMA)]

    lowered = text.lower()
    available = _schema_filter(schema)
    hits: set[int] = set()

    def keyword_hit(keyword: str) -> bool:
        if " " in keyword:
            return keyword in lowered
        return re.search(rf"\b{re.escape(keyword)}\b", lowered) is not None

    for prime, keywords in KEYWORD_MAP.items():
        if prime not in available:
            continue
        if prime == 37:
            continue
        if any(keyword_hit(keyword) for keyword in keywords):
            hits.add(prime)

    if 11 in available and TIME_PATTERN.search(lowered):
        hits.add(11)

    if 37 in available:
        nouns = []
        for match in PROPER_NOUN_PATTERN.finditer(text):
            if match.start() == 0:
                continue
            word = match.group()
            if word in {"I", "We", "The"}:
                continue
            nouns.append(word)
        if nouns:
            hits.add(37)

    if 17 in available and "ethic" in lowered:
        hits.add(17)

    if not hits:
        return [min(available)]
    return sorted(hits)


def tag_modifiers(text: str, schema: Dict[int, Dict]) -> List[int]:
    """Placeholder hook for future embedding-based modifiers."""

    _ = text, schema
    return []

