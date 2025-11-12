"""Ledger memory helpers that delegate to the DualSubstrate API."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Mapping

from prime_tagger import tag_primes


_QUOTE_KEYWORD_PATTERN = re.compile(r"\b(remember|recall|quote|show)\b")
_PREFIXES = (
    "what did we talk about",
    "what did we discuss",
    "remind me",
    "what was the update",
    "recall",
)


def _keywords_from_prompt(text: str) -> list[str]:
    lowered = (text or "").lower()
    words = re.findall(r"\b[a-z0-9]{3,}\b", lowered)
    return list(dict.fromkeys(words))


def _strip_ledger_noise(text: str, *, user_only: bool = False) -> str:
    if not text:
        return ""

    clean_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            continue
        if lowered.startswith("bot:") or "• ledger" in lowered:
            continue
        if user_only and lowered.startswith(("assistant:", "system:")):
            continue
        if len(stripped) > 20:
            clean_lines.append(stripped)
    return "\n".join(clean_lines)


def _is_user_content(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if len(normalized) < 20:
        return False
    bot_prefixes = (
        "bot:",
        "assistant:",
        "system:",
        "model:",
        "ai:",
        "llm:",
        "response:",
        "here’s what the ledger currently recalls:",
        "[",
    )
    if normalized.startswith(bot_prefixes):
        return False
    if "• ledger" in normalized and "ledger factor" in normalized:
        return False
    return True


def _encode_prime_signature(text: str, schema: Mapping[int, Mapping[str, object]], prime_weights: Mapping[int, float]) -> dict[int, float]:
    signature: dict[int, float] = {}
    primes = tag_primes(text, schema)
    factors = [{"prime": p, "delta": 1} for p in primes]
    for factor in factors:
        prime = factor.get("prime")
        delta = factor.get("delta", 0)
        if not isinstance(prime, int):
            continue
        weight = prime_weights.get(prime, 1.0)
        signature[prime] = signature.get(prime, 0.0) + float(delta) * weight
    return signature


def _prime_topological_distance(query_sig: Mapping[int, float], memory_sig: Mapping[int, float], prime_weights: Mapping[int, float]) -> float:
    if not query_sig and not memory_sig:
        return 0.0
    distance = 0.0
    all_primes = set(query_sig.keys()) | set(memory_sig.keys())
    for prime in all_primes:
        query_val = query_sig.get(prime, 0.0)
        memory_val = memory_sig.get(prime, 0.0)
        weight = prime_weights.get(prime, 1.0)
        if query_val > 0 and memory_val == 0:
            distance += abs(query_val) * weight * 2.0
        else:
            distance += abs(query_val - memory_val) * weight
    return distance


@dataclass
class MemoryService:
    """High-level wrapper for fetching and ranking ledger memories."""

    api_service: "ApiService"
    prime_weights: Mapping[int, float]

    def memory_lookup(
        self,
        entity: str,
        *,
        ledger_id: str | None = None,
        limit: int = 3,
        since: int | None = None,
    ) -> list[dict]:
        try:
            data = self.api_service.fetch_memories(
                entity,
                ledger_id=ledger_id,
                limit=limit,
                since=since,
            )
            if any(isinstance(item, dict) and item.get("text") for item in data):
                return data
        except Exception:
            pass

        try:
            ledger_payload = self.api_service.fetch_ledger(entity, ledger_id=ledger_id)
        except Exception:
            return []

        factors = []
        if isinstance(ledger_payload, dict):
            factors = ledger_payload.get("factors") or []

        now_ms = int(time.time() * 1000)
        synthetic: list[dict] = []
        for entry in factors:
            if len(synthetic) >= max(1, limit):
                break
            if not isinstance(entry, dict):
                continue
            prime = entry.get("prime")
            value = entry.get("value", 0)
            if isinstance(prime, int) and value:
                synthetic.append(
                    {
                        "timestamp": now_ms,
                        "text": f"(Ledger factor) Prime {prime} = {value}",
                        "meta": {"source": "ledger"},
                    }
                )
        return synthetic

    def select_context(
        self,
        entity: str,
        query: str,
        schema: Mapping[int, Mapping[str, object]],
        *,
        ledger_id: str | None = None,
        limit: int = 5,
        time_window_hours: int = 72,
        since: int | None = None,
        until: int | None = None,
    ) -> list[dict]:
        keywords = _keywords_from_prompt(query)
        fetch_limit = min(100, max(limit * 5, 25))
        window_start = since
        if window_start is None:
            window_start = int((time.time() - time_window_hours * 3600) * 1000)

        raw_memories = self.memory_lookup(
            entity,
            ledger_id=ledger_id,
            limit=fetch_limit,
            since=window_start,
        )
        if until is not None:
            raw_memories = [m for m in raw_memories if m.get("timestamp", 0) <= until]

        scored_memories = []
        now = time.time()
        query_sig = _encode_prime_signature(query, schema, self.prime_weights)

        for entry in raw_memories:
            text = _strip_ledger_noise(entry.get("text", ""))
            if not text or not _is_user_content(text):
                continue

            score = 0.0
            lowered = text.lower()
            for keyword in keywords:
                score += lowered.count(keyword.lower())

            timestamp = entry.get("timestamp", 0)
            age_hours = (now - timestamp / 1000) / 3600 if timestamp else 0
            score += max(0.0, 10.0 - age_hours * 0.1)

            memory_sig = _encode_prime_signature(text, schema, self.prime_weights)
            distance = _prime_topological_distance(query_sig, memory_sig, self.prime_weights)
            score -= distance * 0.5

            scored_memories.append((score, timestamp, text, entry))

        scored_memories.sort(key=lambda item: item[0], reverse=True)
        selected = []
        for score, timestamp, text, entry in scored_memories:
            payload = dict(entry)
            payload["_sanitized_text"] = text
            payload["_score"] = score
            payload["_timestamp"] = timestamp
            selected.append(payload)
            if len(selected) >= limit:
                break

        return selected

    def latest_user_transcript(
        self,
        entity: str,
        schema: Mapping[int, Mapping[str, object]],
        *,
        ledger_id: str | None = None,
        limit: int = 5,
        since: int | None = None,
    ) -> str | None:
        entries = self.memory_lookup(
            entity,
            ledger_id=ledger_id,
            limit=max(1, limit * 2),
            since=since,
        )
        if not entries:
            return None

        for entry in entries:
            sanitized = _strip_ledger_noise(entry.get("text") or "", user_only=True)
            if sanitized:
                return sanitized
        return None


def is_recall_query(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return normalized.startswith(_PREFIXES) or _QUOTE_KEYWORD_PATTERN.search(normalized) is not None


__all__ = ["MemoryService", "is_recall_query", "_strip_ledger_noise"]
