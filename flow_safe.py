"""Flow-safe sequence helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List


def sequence(primes: Iterable[int], delta: int = 1) -> List[Dict[str, int]]:
    ordered = [int(p) for p in primes if isinstance(p, (int, float))]
    if not ordered:
        ordered = [2]

    safe: List[Dict[str, int]] = []
    for prime in ordered:
        safe.append({"prime": prime, "delta": delta})
        safe.append({"prime": prime, "delta": delta})
    return safe
