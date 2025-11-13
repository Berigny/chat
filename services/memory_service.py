from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping, Sequence, TYPE_CHECKING

from prime_tagger import tag_primes

if TYPE_CHECKING:
    from services.api import ApiService


__all__ = [
    "MemoryService",
    "derive_time_filters",
    "estimate_quote_count",
    "infer_relative_timestamp",
    "is_recall_query",
    "parse_time_range",
    "strip_ledger_noise",
]


_QUOTE_KEYWORD_PATTERN = re.compile(r"\b(remember|recall|quote|show)\b")
_PREFIXES = (
    "what did we talk about",
    "what did we discuss",
    "remind me",
    "what was the update",
    "recall",
)

_DIGIT_PATTERN = re.compile(r"\b\d+\b")
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_QUANTITY_HINTS = {
    "a couple": 2,
    "couple": 2,
    "a few": 3,
    "few": 3,
    "handful": 4,
    "some": 4,
    "several": 6,
    "many": 8,
    "plenty": 8,
    "all": 15,
    "entire": 15,
}
RELATIVE_WORD_OFFSETS = {
    "yesterday": 24 * 3600,
    "last night": 12 * 3600,
    "earlier today": 6 * 3600,
    "this morning": 6 * 3600,
    "this afternoon": 6 * 3600,
    "this evening": 6 * 3600,
    "an hour ago": 3600,
    "an hour earlier": 3600,
    "last week": 7 * 24 * 3600,
}
UNIT_TO_SECONDS = {"minute": 60, "hour": 3600, "day": 86400, "week": 7 * 86400}
RELATIVE_NUMBER_PATTERN = re.compile(r"\b(\d+)\s+(minute|hour|day|week)s?\s+ago\b", re.IGNORECASE)
RELATIVE_ARTICLE_PATTERN = re.compile(r"\b(an|a)\s+(minute|hour|day|week)\s+ago\b", re.IGNORECASE)
LAST_RANGE_PATTERN = re.compile(r"\blast\s+(\d+)\s+(minute|hour|day|week)s?\b", re.IGNORECASE)
PAST_RANGE_PATTERN = re.compile(r"\bpast\s+(\d+)\s+(minute|hour|day|week)s?\b", re.IGNORECASE)
TIME_RANGE_PATTERN = re.compile(
    r"(?:yesterday|today)\s+between\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*-\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    re.IGNORECASE,
)


def _keywords_from_prompt(text: str) -> list[str]:
    lowered = (text or "").lower()
    words = re.findall(r"\b[a-z0-9]{3,}\b", lowered)
    return list(dict.fromkeys(words))


def strip_ledger_noise(text: str, *, user_only: bool = False) -> str:
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
        has_quote = any(q in stripped for q in ('"', "“", "”", "‘", "’"))
        if len(stripped) > 20 or has_quote:
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


def _encode_prime_signature(
    text: str,
    schema: Mapping[int, Mapping[str, object]],
    prime_weights: Mapping[int, float],
) -> dict[int, float]:
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


def _prime_topological_distance(
    query_sig: Mapping[int, float],
    memory_sig: Mapping[int, float],
    prime_weights: Mapping[int, float],
) -> float:
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


def _extract_requested_count(text: str) -> int | None:
    if not text:
        return None

    digits = [int(match) for match in _DIGIT_PATTERN.findall(text)]
    if digits:
        return digits[-1]

    lowered = text.lower()
    for phrase, value in _QUANTITY_HINTS.items():
        if phrase in lowered:
            return value

    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return value

    return None


def estimate_quote_count(text: str, *, default: int = 5) -> int:
    explicit = _extract_requested_count(text)
    if explicit:
        return max(1, min(explicit, 25))

    lowered = (text or "").lower()
    for phrase, value in _QUANTITY_HINTS.items():
        if phrase in lowered:
            return max(1, min(value, 25))

    return default


def parse_time_range(text: str) -> tuple[int | None, int | None]:
    if not text:
        return None, None
    match = TIME_RANGE_PATTERN.search(text)
    if not match:
        return None, None

    lowered = text.lower()
    now = datetime.now()
    base_date = now - timedelta(days=1) if "yesterday" in lowered else now

    try:
        start_hour = int(match.group(1))
        start_minute = int(match.group(2) or 0)
        start_ampm = (match.group(3) or "").lower()
        end_hour = int(match.group(4))
        end_minute = int(match.group(5) or 0)
        end_ampm = (match.group(6) or "").lower()

        if not start_ampm and end_ampm:
            start_ampm = end_ampm
        if start_ampm == "pm" and start_hour != 12:
            start_hour += 12
        if start_ampm == "am" and start_hour == 12:
            start_hour = 0
        if not end_ampm and start_ampm:
            end_ampm = start_ampm
        if end_ampm == "pm" and end_hour != 12:
            end_hour += 12
        if end_ampm == "am" and end_hour == 12:
            end_hour = 0

        start_dt = base_date.replace(hour=start_hour % 24, minute=start_minute, second=0, microsecond=0)
        end_dt = base_date.replace(hour=end_hour % 24, minute=end_minute, second=0, microsecond=0)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)
    except Exception:
        return None, None


def infer_relative_timestamp(text: str, *, now: float | None = None) -> int | None:
    if not text:
        return None
    lowered = text.lower()
    now = now or time.time()
    for phrase, seconds in RELATIVE_WORD_OFFSETS.items():
        if phrase in lowered:
            return int((now - seconds) * 1000)
    match = RELATIVE_NUMBER_PATTERN.search(lowered)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower().rstrip("s")
        seconds = UNIT_TO_SECONDS.get(unit)
        if seconds:
            return int((now - amount * seconds) * 1000)
    match = RELATIVE_ARTICLE_PATTERN.search(lowered)
    if match:
        unit = match.group(2).lower()
        seconds = UNIT_TO_SECONDS.get(unit)
        if seconds:
            return int((now - seconds) * 1000)
    match = LAST_RANGE_PATTERN.search(lowered) or PAST_RANGE_PATTERN.search(lowered)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower().rstrip("s")
        seconds = UNIT_TO_SECONDS.get(unit)
        if seconds:
            return int((now - amount * seconds) * 1000)
    return None


def derive_time_filters(text: str) -> tuple[int | None, int | None]:
    since, until = parse_time_range(text)
    if since is None:
        since = infer_relative_timestamp(text)
    return since, until


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
        return []

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
            text = strip_ledger_noise(entry.get("text", ""))
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

            payload = dict(entry)
            payload["_sanitized_text"] = text
            payload["_score"] = score
            payload["_timestamp"] = timestamp
            scored_memories.append(payload)

        scored_memories.sort(key=lambda item: item.get("_score", 0.0), reverse=True)
        return scored_memories[:limit]

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
            sanitized = strip_ledger_noise(entry.get("text") or "", user_only=True)
            if sanitized:
                return sanitized
        return None

    def render_context_block(
        self,
        entity: str,
        schema: Mapping[int, Mapping[str, object]],
        *,
        ledger_id: str | None = None,
        limit: int = 3,
        since: int | None = None,
        keywords: Sequence[str] | None = None,
    ) -> str:
        if not entity:
            return ""
        fetch_limit = limit
        if keywords:
            fetch_limit = min(100, max(limit * 4, 20))
        entries = self.memory_lookup(
            entity,
            ledger_id=ledger_id,
            limit=fetch_limit,
            since=since,
        )
        snippets: list[str] = []
        normalized_keywords = [k.lower() for k in (keywords or []) if len(k) >= 3]
        for entry in entries:
            text = strip_ledger_noise((entry.get("text") or "").strip())
            if not text:
                continue
            lowered = text.lower()
            if lowered.startswith("(ledger reset"):
                continue
            if normalized_keywords and not any(k in lowered for k in normalized_keywords):
                continue
            source = entry.get("meta", {}).get("source") or entry.get("name") or entry.get("attachment") or ""
            snippet = text[:240].replace("\n", " ")
            if len(text) > len(snippet):
                snippet += "…"
            label = f" ({source})" if source else ""
            snippets.append(f"- {snippet}{label}")
            if len(snippets) >= limit:
                break
        if not snippets and keywords:
            focus = ", ".join(list(keywords)[:3])
            return f"- No ledger memories matched the topic ({focus})."
        return "\n".join(snippets)

    def build_recall_response(
        self,
        entity: str | None,
        query: str,
        schema: Mapping[int, Mapping[str, object]],
        *,
        ledger_id: str | None = None,
        limit: int | None = None,
        since: int | None = None,
        until: int | None = None,
    ) -> str | None:
        if not entity:
            return None
        if limit is None:
            limit = estimate_quote_count(query)
        derived_since, derived_until = derive_time_filters(query)
        if since is None:
            since = derived_since
        if until is None:
            until = derived_until

        memories = self.select_context(
            entity,
            query,
            schema,
            ledger_id=ledger_id,
            limit=limit,
            since=since,
            until=until,
            time_window_hours=168,
        )
        if not memories:
            return None

        lines = ["Here’s what the ledger currently recalls:"]
        for entry in memories:
            timestamp = entry.get("timestamp")
            if timestamp:
                date_str = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M")
            else:
                date_str = "unknown time"
            text_content = entry.get("_sanitized_text") or strip_ledger_noise((entry.get("text") or "").strip())
            if not text_content:
                continue
            lines.append(f"[{date_str} • ledger] {text_content}")

        if len(lines) == 1:
            return None
        return "\n".join(lines)


def is_recall_query(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return normalized.startswith(_PREFIXES) or _QUOTE_KEYWORD_PATTERN.search(normalized) is not None
