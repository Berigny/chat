"""Flow-safe sequence helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List


def sequence(primes: Iterable[int]) -> List[Dict[str, int]]:
    ordered = list(primes)
    if not ordered:
        ordered = [2]

    seq: List[Dict[str, int]] = []
    for prime in ordered:
        seq.append({"p": prime, "d": 1})
        seq.append({"p": prime, "d": 1})
    return seq

