"""Prime tagging and anchoring utilities."""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Sequence

from prime_tagger import tag_primes
from validators import get_tier_value

PrimeSchema = Dict[int, Dict[str, object]]
Factors = List[Dict[str, int]]
FactorBatches = List[Factors]


def _normalize_factors_override(factors: object, valid_primes: Sequence[int]) -> Factors:
    normalized: Factors = []
    if not isinstance(factors, list):
        return normalized
    for item in factors:
        if not isinstance(item, dict):
            continue
        prime = item.get("prime")
        delta = item.get("delta", 1)
        if prime in valid_primes:
            try:
                normalized.append({"prime": int(prime), "delta": int(delta)})
            except (TypeError, ValueError):
                continue
    return normalized


def normalize_override_factors(factors: object, valid_primes: Sequence[int]) -> Factors:
    """Public wrapper for validating override payloads."""

    return _normalize_factors_override(factors, valid_primes)


def _flow_safe_factors(factors: Factors, valid_primes: Sequence[int]) -> Factors:
    filtered: Factors = []
    seen: set[int] = set()
    for factor in factors:
        prime = factor.get("prime")
        if prime not in valid_primes or prime in seen:
            continue
        delta = factor.get("delta", 1)
        try:
            entry = {"prime": int(prime), "delta": int(delta)}
        except (TypeError, ValueError):
            continue
        filtered.append(entry)
        seen.add(entry["prime"])
    odds = [f for f in filtered if f["prime"] % 2 == 1]
    evens = [f for f in filtered if f["prime"] % 2 == 0]
    return odds + evens


def _extract_json_object(raw: str) -> Dict | None:
    if not raw:
        return None
    trimmed = raw.strip()
    if "{" not in trimmed:
        return None
    candidate = trimmed
    if "```" in trimmed:
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = trimmed[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def call_factor_extraction_llm(
    text: str,
    schema: PrimeSchema,
    *,
    genai_module=None,
    model_name: str = "gemini-2.0-flash",
) -> Factors:
    if not text or genai_module is None:
        return []
    schema_lines = "\n".join(
        f"{prime} ({meta.get('name', f'Prime {prime}')}) = {meta.get('tier', '')} {meta.get('mnemonic', '')} {meta.get('description', '')}".strip()
        for prime, meta in schema.items()
    )
    prompt = (
        "You extract ledger factors. "
        "Given the transcript below, identify subject (prime 2), action (3), object (5), "
        "location/channel (7), time/date (11), intent/outcome (13), context (17), sentiment/priority (19). "
        "Only include a prime if the transcript clearly expresses that facet. "
        "Return STRICT JSON with keys `text` (repeat the transcript) and `factors` "
        "(an array of objects with `prime` and `delta`). Example: "
        '{"text":"Met Priya","factors":[{"prime":2,"delta":1},{"prime":3,"delta":1}]}. '
        "Prime semantics:\n"
        f"{schema_lines}\n"
        f"Transcript:\n{text}"
    )
    try:
        model = genai_module.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        raw = getattr(response, "text", None) or ""
    except Exception:
        return []
    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        return []
    valid_primes = tuple(schema.keys())
    return _normalize_factors_override(data.get("factors"), valid_primes)


def map_to_primes(
    text: str,
    schema: PrimeSchema,
    *,
    fallback_prime: int,
    valid_primes: Sequence[int] | None = None,
    llm_extractor: Callable[[str, PrimeSchema], Factors] | None = None,
) -> Factors:
    valid = tuple(valid_primes or schema.keys())
    if llm_extractor:
        llm_factors = llm_extractor(text, schema)
        if llm_factors:
            return llm_factors
    primes = tag_primes(text, schema)
    if not primes:
        return [{"prime": fallback_prime, "delta": 1}]
    return [{"prime": int(p), "delta": 1} for p in primes if p in valid]


def build_anchor_batches(
    text: str,
    schema: PrimeSchema,
    *,
    fallback_prime: int,
    factors_override: Sequence[Dict[str, int]] | None = None,
    llm_extractor: Callable[[str, PrimeSchema], Factors] | None = None,
) -> FactorBatches:
    valid_primes = tuple(schema.keys()) or (fallback_prime,)
    batches: FactorBatches = []

    def _append_batches(candidates: Factors) -> None:
        normalized = _normalize_factors_override(list(candidates), valid_primes)
        safe = _flow_safe_factors(normalized, valid_primes)
        if not safe:
            safe = [{"prime": fallback_prime, "delta": 1}]
        for factor in safe:
            batches.append(
                [
                    {"prime": factor["prime"], "delta": factor["delta"]},
                    {"prime": factor["prime"], "delta": factor["delta"]},
                ]
            )

    if factors_override:
        _append_batches(list(factors_override))
    else:
        mapped = map_to_primes(
            text,
            schema,
            fallback_prime=fallback_prime,
            valid_primes=valid_primes,
            llm_extractor=llm_extractor,
        )
        tiered: Dict[int, Factors] = {}
        for factor in mapped:
            prime = factor["prime"]
            tier = get_tier_value(prime, schema)
            tiered.setdefault(tier, []).append(factor)
        for tier in sorted(tiered.keys()):
            _append_batches(tiered[tier])

    if not batches:
        _append_batches([{ "prime": fallback_prime, "delta": 1 }])

    return batches


__all__ = [
    "build_anchor_batches",
    "call_factor_extraction_llm",
    "map_to_primes",
    "normalize_override_factors",
]
