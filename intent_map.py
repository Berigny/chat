"""Intent routing helpers."""

from __future__ import annotations

from typing import List, Tuple


def route_topic(question: str) -> str:
    lowered = (question or "").lower()
    if "god" in lowered:
        return "theology/god"
    if "memory" in lowered:
        return "memory/management"
    return "general"


def intent_primes(intent: str) -> Tuple[List[int], List[int], List[int]]:
    lowered = (intent or "").lower()
    if "god" in lowered or "definition" in lowered:
        return [3], [19, 5, 13, 11], [37, 23, 29]
    if "memory" in lowered:
        return [5], [2, 17], [23]
    return [], [], []

