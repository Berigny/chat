from __future__ import annotations

import copy
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

import requests

from prime_tagger import tag_primes

if TYPE_CHECKING:
    from services.api import ApiService


logger = logging.getLogger(__name__)


__all__ = [
    "MemoryService",
    "derive_time_filters",
    "estimate_quote_count",
    "infer_relative_timestamp",
    "is_recall_query",
    "parse_time_range",
    "strip_ledger_noise",
]


_QUOTE_KEYWORD_PATTERN = re.compile(r"\b(remember|recall|quote(?:s|d|ing)?|show)\b")
_DIALOGUE_LABEL_PATTERN = re.compile(
    r"^(?:you|user|assistant|bot|system|model|ai|llm)\s*:", re.IGNORECASE
)
_LEDGER_RECALL_PREFIXES = (
    "here’s what the ledger currently recalls:",
    "here's what the ledger currently recalls:",
)
_TIMESTAMP_INLINE_PATTERN = re.compile(r"\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}")
_PREFIXES = (
    "what did we talk about",
    "what did we discuss",
    "remind me",
    "what was the update",
    "recall",
)
_LEDGER_LOOKUP_HINTS = (
    "quote",
    "quotes",
    "quoted",
    "quoting",
    "sentence",
    "paragraph",
    "entry",
    "line",
    "memory",
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
ASSEMBLY_CACHE_TTL = 1.5
INFERENCE_CACHE_TTL = 5.0
TRAVERSAL_CACHE_TTL = 5.0
MOBIUS_REFRESH_INTERVAL = 180.0


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
        if _DIALOGUE_LABEL_PATTERN.match(stripped):
            continue
        if user_only and lowered.startswith(("assistant:", "system:")):
            continue
        has_quote = any(q in stripped for q in ('"', "“", "”", "‘", "’"))
        if len(stripped) > 20 or has_quote:
            clean_lines.append(stripped)
    return "\n".join(clean_lines)


def _looks_like_transcript(text: str) -> bool:
    if not text:
        return False

    timestamp_hits = _TIMESTAMP_INLINE_PATTERN.findall(text)
    if len(timestamp_hits) >= 2:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    timestamp_lines = sum(1 for line in lines if _TIMESTAMP_INLINE_PATTERN.search(line))
    if timestamp_lines >= max(1, len(lines) // 2):
        return True

    return False


def _is_user_content(text: str) -> bool:
    normalized = (text or "").strip()
    lowered = normalized.lower()
    if len(lowered) < 20:
        return False
    bot_prefixes = (
        "bot:",
        "assistant:",
        "system:",
        "model:",
        "ai:",
        "llm:",
        "response:",
        "[",
        "you:",
        "user:",
    )
    if lowered.startswith(bot_prefixes):
        return False
    ascii_lowered = lowered.replace("’", "'")
    if ascii_lowered.startswith(tuple(prefix.replace("’", "'") for prefix in _LEDGER_RECALL_PREFIXES)):
        return False
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) >= 2 and all(_DIALOGUE_LABEL_PATTERN.match(line) for line in lines):
        return False
    if "• ledger" in lowered and "ledger factor" in lowered:
        return False
    if _looks_like_transcript(normalized):
        return False
    return True


def _coerce_mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _normalize_timestamp(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _normalize_tags(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        tags = []
        for item in raw:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    tags.append(stripped)
        return tuple(tags)
    return ()


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
    _assembly_cache: dict[
        tuple[str, str | None, int | None, bool, int | None],
        tuple[float, dict[str, list[dict]]],
    ] = field(default_factory=dict, init=False, repr=False)
    _body_cache: dict[int, list[str]] = field(default_factory=dict, init=False, repr=False)
    _mobius_state: dict[tuple[str, str | None], dict[str, float]] = field(
        default_factory=dict, init=False, repr=False
    )
    _inference_cache: dict[
        tuple[str, str | None, bool, int | None],
        tuple[float, dict[str, Any]],
    ] = field(default_factory=dict, init=False, repr=False)
    _traverse_cache: dict[
        tuple[str, str | None, int | None, int | None, int | None, str | None, bool],
        tuple[float, dict[str, Any]],
    ] = field(default_factory=dict, init=False, repr=False)

    def clear_entity_cache(self, *, entity: str | None = None, ledger_id: str | None = None) -> None:
        """Clear cached assembly data for the provided entity/ledger."""

        if entity is None and ledger_id is None:
            self._assembly_cache.clear()
            self._body_cache.clear()
            self._inference_cache.clear()
            self._traverse_cache.clear()
            return

        ledger_key = ledger_id or None
        keys_to_drop = [
            key
            for key in self._assembly_cache.keys()
            if (entity is None or key[0] == entity) and (ledger_id is None or key[1] == ledger_key)
        ]
        for key in keys_to_drop:
            self._assembly_cache.pop(key, None)

        if entity is not None or ledger_id is not None:
            self._body_cache.clear()

        if entity is not None or ledger_id is not None:
            inference_keys = [
                key
                for key in self._inference_cache.keys()
                if (entity is None or key[0] == entity)
                and (ledger_id is None or key[1] == ledger_key)
            ]
            for key in inference_keys:
                self._inference_cache.pop(key, None)

            traverse_keys = [
                key
                for key in self._traverse_cache.keys()
                if (entity is None or key[0] == entity)
                and (ledger_id is None or key[1] == ledger_key)
            ]
            for key in traverse_keys:
                self._traverse_cache.pop(key, None)

    def assemble_context(
        self,
        entity: str | None,
        *,
        ledger_id: str | None = None,
        k: int | None = None,
        quote_safe: bool | None = None,
        since: int | None = None,
    ) -> dict[str, list[dict]]:
        """Fetch prompt assembly payloads (S2 summaries, S1 bodies, claims)."""

        if not entity or not getattr(self, "api_service", None):
            return {"summaries": [], "bodies": [], "claims": []}

        normalized_k = int(k) if isinstance(k, (int, float)) else None
        normalized_quote_safe = None if quote_safe is None else bool(quote_safe)
        cache_key = (entity, ledger_id or None, normalized_k, normalized_quote_safe, since or None)

        cached = self._assembly_cache.get(cache_key)
        now = time.time()
        if cached and now - cached[0] <= ASSEMBLY_CACHE_TTL:
            return copy.deepcopy(cached[1])

        try:
            payload = self.api_service.fetch_assembly(
                entity,
                ledger_id=ledger_id,
                k=normalized_k,
                quote_safe=normalized_quote_safe,
                since=since,
            )
        except Exception:
            if cached:
                return copy.deepcopy(cached[1])
            return {"summaries": [], "bodies": [], "claims": []}

        filter_quote_safe = bool(normalized_quote_safe)
        normalized = self._normalize_assembly_payload(payload, quote_safe=filter_quote_safe)
        normalized_copy = copy.deepcopy(normalized)
        self._assembly_cache[cache_key] = (now, normalized_copy)
        return copy.deepcopy(normalized_copy)

    def _normalize_assembly_payload(
        self,
        payload: Mapping[str, Any] | None,
        *,
        quote_safe: bool,
    ) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {"summaries": [], "bodies": [], "claims": []}
        if not isinstance(payload, Mapping):
            return result

        s2_section = payload.get("s2") if isinstance(payload.get("s2"), Mapping) else None
        summary_candidates: list[Mapping[str, Any]] = []
        if isinstance(s2_section, Mapping):
            summary_candidates.extend(_coerce_mapping_sequence(s2_section.get("summaries")))
            summary_candidates.extend(_coerce_mapping_sequence(s2_section.get("summary_refs")))
            summary_candidates.extend(_coerce_mapping_sequence(s2_section.get("refs")))
        summary_candidates.extend(_coerce_mapping_sequence(payload.get("summaries")))
        summary_candidates.extend(_coerce_mapping_sequence(payload.get("summary_refs")))

        summaries: list[dict] = []
        for entry in summary_candidates:
            normalized = self._normalize_summary_entry(entry)
            if not normalized:
                continue
            if quote_safe and normalized.get("quote_safe") is False:
                continue
            summaries.append(normalized)

        bodies_section = payload.get("s1") if isinstance(payload.get("s1"), Mapping) else None
        body_candidates: list[Mapping[str, Any]] = []
        if isinstance(bodies_section, Mapping):
            body_candidates.extend(_coerce_mapping_sequence(bodies_section.get("bodies")))
            body_candidates.extend(_coerce_mapping_sequence(bodies_section.get("body_refs")))
        body_candidates.extend(_coerce_mapping_sequence(payload.get("bodies")))
        body_candidates.extend(_coerce_mapping_sequence(payload.get("body_refs")))

        bodies: list[dict] = []
        for entry in body_candidates:
            normalized = self._normalize_body_entry(entry)
            if not normalized:
                continue
            if quote_safe and normalized.get("quote_safe") is False:
                continue
            bodies.append(normalized)

        claim_candidates: list[Mapping[str, Any]] = []
        if isinstance(s2_section, Mapping):
            claim_candidates.extend(_coerce_mapping_sequence(s2_section.get("claims")))
        claim_candidates.extend(_coerce_mapping_sequence(payload.get("claims")))

        claims: list[dict] = []
        for entry in claim_candidates:
            normalized = self._normalize_claim_entry(entry)
            if not normalized:
                continue
            claims.append(normalized)

        summaries = self._dedupe_and_sort(
            summaries,
            key=lambda item: (item.get("prime"), item.get("summary")),
            sort_key=lambda item: (item.get("timestamp") or 0, item.get("score", 0.0)),
        )
        bodies = self._dedupe_and_sort(
            bodies,
            key=lambda item: (item.get("prime"), tuple(item.get("body", []))),
            sort_key=lambda item: (item.get("timestamp") or 0, item.get("prime") or 0),
        )
        claims = self._dedupe_and_sort(
            claims,
            key=lambda item: (item.get("prime"), item.get("claim")),
            sort_key=lambda item: (item.get("timestamp") or 0, item.get("score", 0.0)),
        )

        result["summaries"] = summaries
        result["bodies"] = bodies
        result["claims"] = claims
        return result

    def _dedupe_and_sort(
        self,
        items: list[dict],
        *,
        key: Callable[[dict], Any],
        sort_key: Callable[[dict], Any],
    ) -> list[dict]:
        seen: set = set()
        ordered: list[dict] = []
        for item in items:
            try:
                marker = key(item)
            except Exception:
                marker = None
            if marker in seen:
                continue
            seen.add(marker)
            ordered.append(item)
        ordered.sort(key=lambda item: sort_key(item), reverse=True)
        return ordered

    def _normalize_summary_entry(self, entry: Mapping[str, Any]) -> dict | None:
        summary = entry.get("summary") or entry.get("text") or entry.get("content")
        if not isinstance(summary, str):
            return None
        summary_text = summary.strip()
        if not summary_text:
            return None

        prime = entry.get("prime")
        if not isinstance(prime, int):
            candidate = entry.get("prime_ref") or entry.get("summary_prime") or entry.get("id")
            prime = candidate if isinstance(candidate, int) else None

        title = entry.get("title") or entry.get("name")
        title = title.strip() if isinstance(title, str) else None

        timestamp = _normalize_timestamp(entry.get("timestamp") or entry.get("ts") or entry.get("time"))
        score_raw = entry.get("score")
        score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0
        quote_flag = entry.get("quote_safe")
        if isinstance(quote_flag, bool):
            quote_safe = quote_flag
        else:
            quote_safe = bool(entry.get("quoteSafe") or entry.get("quote_allowed"))

        body_prime = (
            entry.get("body_prime")
            or entry.get("bodyPrime")
            or entry.get("prime_ref")
            or entry.get("body_ref")
        )

        normalized = {
            "prime": prime,
            "title": title,
            "summary": summary_text,
            "tags": _normalize_tags(entry.get("tags") or entry.get("labels")),
            "timestamp": timestamp,
            "score": score,
            "quote_safe": quote_safe,
        }
        if isinstance(body_prime, int):
            normalized["body_prime"] = body_prime
        return normalized

    def _normalize_body_entry(self, entry: Mapping[str, Any]) -> dict | None:
        prime = entry.get("prime") or entry.get("body_prime") or entry.get("prime_ref")
        if not isinstance(prime, int):
            candidate = entry.get("bodyPrime") or entry.get("body_ref")
            prime = candidate if isinstance(candidate, int) else None
        if prime is None:
            return None

        body_source = entry.get("body") or entry.get("text") or entry.get("content") or entry.get("chunks")
        body_chunks: list[str] = []
        if isinstance(body_source, str):
            chunk = body_source.strip()
            if chunk:
                body_chunks.append(chunk)
        elif isinstance(body_source, Sequence) and not isinstance(body_source, (str, bytes)):
            for chunk in body_source:
                if isinstance(chunk, str):
                    snippet = chunk.strip()
                    if snippet:
                        body_chunks.append(snippet)
        if not body_chunks:
            cached = self._body_cache.get(prime)
            if cached:
                body_chunks = list(cached)
        else:
            self._body_cache[prime] = list(body_chunks)

        summary = entry.get("summary")
        if isinstance(summary, str):
            summary = summary.strip() or None
        else:
            summary = None

        title = entry.get("title") or entry.get("name")
        title = title.strip() if isinstance(title, str) else None
        timestamp = _normalize_timestamp(entry.get("timestamp") or entry.get("ts") or entry.get("time"))
        tags = _normalize_tags(entry.get("tags") or entry.get("labels"))
        quote_flag = entry.get("quote_safe")
        if isinstance(quote_flag, bool):
            quote_safe = quote_flag
        else:
            quote_safe = bool(entry.get("quoteSafe") or entry.get("quote_allowed"))

        if not body_chunks and not summary:
            return None

        payload = {
            "prime": prime,
            "title": title,
            "summary": summary,
            "body": body_chunks,
            "tags": tags,
            "timestamp": timestamp,
            "quote_safe": quote_safe,
        }
        return payload

    def _normalize_claim_entry(self, entry: Mapping[str, Any]) -> dict | None:
        claim_text = entry.get("claim") or entry.get("text") or entry.get("content") or entry.get("summary")
        if not isinstance(claim_text, str):
            return None
        claim = claim_text.strip()
        if not claim:
            return None

        prime = entry.get("prime") or entry.get("prime_ref") or entry.get("claim_prime")
        if not isinstance(prime, int):
            prime = None

        timestamp = _normalize_timestamp(entry.get("timestamp") or entry.get("ts") or entry.get("time"))
        tags = _normalize_tags(entry.get("tags") or entry.get("labels"))
        score_raw = entry.get("score")
        score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0

        return {
            "prime": prime,
            "claim": claim,
            "tags": tags,
            "timestamp": timestamp,
            "score": score,
        }

    def _sanitize_inference_entry(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(entry, Mapping):
            return {}

        sanitized: dict[str, Any] = {}

        prime = entry.get("prime") or entry.get("prime_ref") or entry.get("body_prime")
        if isinstance(prime, int):
            sanitized["prime"] = prime

        label: str | None = None
        for key in ("label", "summary", "title", "name", "task", "description"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                label = value.strip()
                break
        if label is None and isinstance(prime, int):
            label = f"Prime {prime}"
        identifier = entry.get("id") or entry.get("job_id") or entry.get("token")
        if label is None and isinstance(identifier, str) and identifier.strip():
            label = identifier.strip()
        if label:
            sanitized["label"] = label

        status = entry.get("status") or entry.get("state") or entry.get("phase")
        if isinstance(status, str) and status.strip():
            sanitized["status"] = status.strip()

        note = entry.get("note") or entry.get("message") or entry.get("detail")
        if isinstance(note, str) and note.strip():
            sanitized["note"] = note.strip()

        score = entry.get("score") or entry.get("weight") or entry.get("confidence")
        if isinstance(score, (int, float)):
            sanitized["score"] = float(score)

        timestamp = None
        for key in ("timestamp", "ts", "updated_at", "queued_at", "started_at", "created_at", "completed_at"):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                timestamp = int(value)
                break
        if timestamp is not None:
            sanitized["timestamp"] = timestamp

        metadata = entry.get("metadata") or entry.get("meta")
        if isinstance(metadata, Mapping):
            filtered_meta = {
                str(key): value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool))
            }
            if filtered_meta:
                sanitized["metadata"] = filtered_meta

        if not sanitized:
            primitive_only = {
                str(key): value
                for key, value in entry.items()
                if isinstance(value, (str, int, float, bool))
            }
            if primitive_only:
                sanitized["raw"] = primitive_only
        return sanitized

    def _normalize_inference_state(
        self,
        payload: Mapping[str, Any] | None,
        *,
        include_history: bool,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": None,
            "active": None,
            "queue": [],
            "history": [],
            "updated_at": None,
            "metrics": {},
            "message": None,
            "supported": True,
        }
        if not isinstance(payload, Mapping):
            return result

        status = payload.get("status") or payload.get("state")
        if isinstance(status, str) and status.strip():
            result["status"] = status.strip()

        active_entry = payload.get("active") or payload.get("current") or payload.get("inflight")
        if isinstance(active_entry, Mapping):
            sanitized = self._sanitize_inference_entry(active_entry)
            result["active"] = sanitized or None

        queue_entries = payload.get("queue") or payload.get("pending") or payload.get("backlog")
        queue_list: list[dict[str, Any]] = []
        for entry in _coerce_mapping_sequence(queue_entries):
            sanitized = self._sanitize_inference_entry(entry)
            if sanitized:
                queue_list.append(sanitized)
        result["queue"] = queue_list

        history_entries = payload.get("history") or payload.get("recent") or payload.get("completed")
        history_list: list[dict[str, Any]] = []
        if include_history:
            for entry in _coerce_mapping_sequence(history_entries):
                sanitized = self._sanitize_inference_entry(entry)
                if sanitized:
                    history_list.append(sanitized)
        result["history"] = history_list

        updated = payload.get("updated_at") or payload.get("timestamp") or payload.get("ts")
        if isinstance(updated, (int, float)):
            result["updated_at"] = int(updated)

        metrics = payload.get("metrics") or payload.get("telemetry")
        if isinstance(metrics, Mapping):
            filtered_metrics = {
                str(key): float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
            result["metrics"] = filtered_metrics

        message = payload.get("message") or payload.get("detail")
        if isinstance(message, str) and message.strip():
            result["message"] = message.strip()

        return result

    def _sanitize_traverse_node(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(entry, Mapping):
            return {}

        sanitized: dict[str, Any] = {}
        prime = entry.get("prime") or entry.get("prime_ref") or entry.get("node")
        if isinstance(prime, int):
            sanitized["prime"] = prime

        label = None
        for key in ("label", "summary", "title", "name", "mnemonic"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                label = value.strip()
                break
        if label is None and isinstance(prime, int):
            label = f"Prime {prime}"
        if label:
            sanitized["label"] = label

        weight = entry.get("weight") or entry.get("score") or entry.get("delta")
        if isinstance(weight, (int, float)):
            sanitized["weight"] = float(weight)

        note = entry.get("note") or entry.get("description") or entry.get("text")
        if isinstance(note, str) and note.strip():
            sanitized["note"] = note.strip()

        timestamp = entry.get("timestamp") or entry.get("ts")
        if isinstance(timestamp, (int, float)):
            sanitized["timestamp"] = int(timestamp)

        tags = entry.get("tags")
        if isinstance(tags, Mapping):
            tag_values = [value for value in tags.values() if isinstance(value, str)]
        elif isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
            tag_values = [str(value) for value in tags if isinstance(value, (str, int))]
        else:
            tag_values = []
        if tag_values:
            sanitized["tags"] = tag_values[:5]

        if not sanitized:
            primitive_only = {
                str(key): value
                for key, value in entry.items()
                if isinstance(value, (str, int, float, bool))
            }
            if primitive_only:
                sanitized["raw"] = primitive_only
        return sanitized

    def _normalize_traverse_payload(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "origin": None,
            "paths": [],
            "metadata": {},
            "message": None,
            "supported": True,
        }
        if not isinstance(payload, Mapping):
            return result

        origin = payload.get("origin") or payload.get("source_prime") or payload.get("start")
        if isinstance(origin, int):
            result["origin"] = origin

        path_candidates: Sequence[Mapping[str, Any]] | None = None
        for key in ("paths", "traversals", "routes", "walks", "results"):
            entries = payload.get(key)
            if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
                mappings = [entry for entry in entries if isinstance(entry, Mapping)]
                if mappings:
                    path_candidates = mappings
                    break

        normalized_paths: list[dict[str, Any]] = []
        if path_candidates:
            for entry in path_candidates:
                nodes_source = None
                for node_key in ("nodes", "path", "steps", "primes"):
                    value = entry.get(node_key)
                    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                        nodes_source = value
                        break
                nodes = [
                    self._sanitize_traverse_node(node)
                    for node in (nodes_source or [])
                    if isinstance(node, Mapping)
                ]
                nodes = [node for node in nodes if node]
                if not nodes:
                    continue
                score = entry.get("score") or entry.get("weight") or entry.get("confidence")
                normalized_paths.append(
                    {
                        "nodes": nodes,
                        "score": float(score) if isinstance(score, (int, float)) else None,
                    }
                )
        result["paths"] = normalized_paths

        metadata = payload.get("metadata") or payload.get("meta")
        if isinstance(metadata, Mapping):
            filtered_meta = {
                str(key): value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool))
            }
            result["metadata"] = filtered_meta

        message = payload.get("message") or payload.get("detail")
        if isinstance(message, str) and message.strip():
            result["message"] = message.strip()

        return result

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

    def supports_inference_state(self) -> bool:
        try:
            return bool(self.api_service.supports_inference_state())
        except AttributeError:
            return False

    def supports_traverse(self) -> bool:
        try:
            return bool(self.api_service.supports_traverse())
        except AttributeError:
            return False

    def fetch_inference_state(
        self,
        entity: str | None,
        *,
        ledger_id: str | None = None,
        include_history: bool = False,
        limit: int | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        if not entity or not self.supports_inference_state():
            return {
                "status": None,
                "active": None,
                "queue": [],
                "history": [],
                "updated_at": None,
                "metrics": {},
                "message": None,
                "supported": False,
            }

        cache_key = (entity, ledger_id or None, include_history, limit if isinstance(limit, int) else None)
        cached = self._inference_cache.get(cache_key)
        now = time.time()
        if cached and not refresh and now - cached[0] <= INFERENCE_CACHE_TTL:
            return copy.deepcopy(cached[1])

        try:
            payload = self.api_service.fetch_inference_state(
                entity,
                ledger_id=ledger_id,
                include_history=include_history,
                limit=limit,
            )
        except requests.HTTPError as exc:
            logger.warning("Inference state request failed for %s: %s", entity, exc)
            normalized = {
                "status": None,
                "active": None,
                "queue": [],
                "history": [],
                "updated_at": None,
                "metrics": {},
                "message": str(exc),
                "supported": False,
            }
        except requests.RequestException as exc:
            logger.warning("Inference state request failed for %s: %s", entity, exc)
            normalized = {
                "status": None,
                "active": None,
                "queue": [],
                "history": [],
                "updated_at": None,
                "metrics": {},
                "message": str(exc),
                "supported": False,
            }
        else:
            normalized = self._normalize_inference_state(payload, include_history=include_history)
            normalized["supported"] = True

        self._inference_cache[cache_key] = (now, copy.deepcopy(normalized))
        return copy.deepcopy(normalized)

    def traversal_paths(
        self,
        entity: str | None,
        *,
        ledger_id: str | None = None,
        origin: int | None = None,
        limit: int | None = None,
        depth: int | None = None,
        direction: str | None = None,
        include_metadata: bool = False,
        refresh: bool = False,
    ) -> dict[str, Any]:
        if not entity or not self.supports_traverse():
            return {
                "origin": origin,
                "paths": [],
                "metadata": {},
                "message": None,
                "supported": False,
            }

        cache_key = (
            entity,
            ledger_id or None,
            origin,
            limit,
            depth,
            direction,
            bool(include_metadata),
        )
        cached = self._traverse_cache.get(cache_key)
        now = time.time()
        if cached and not refresh and now - cached[0] <= TRAVERSAL_CACHE_TTL:
            return copy.deepcopy(cached[1])

        try:
            payload = self.api_service.traverse(
                entity,
                ledger_id=ledger_id,
                origin=origin,
                limit=limit,
                depth=depth,
                direction=direction,
                include_metadata=include_metadata,
            )
        except requests.HTTPError as exc:
            logger.warning("Traversal request failed for %s: %s", entity, exc)
            normalized = {
                "origin": origin,
                "paths": [],
                "metadata": {},
                "message": str(exc),
                "supported": False,
            }
        except requests.RequestException as exc:
            logger.warning("Traversal request failed for %s: %s", entity, exc)
            normalized = {
                "origin": origin,
                "paths": [],
                "metadata": {},
                "message": str(exc),
                "supported": False,
            }
        else:
            normalized = self._normalize_traverse_payload(payload)
            normalized["supported"] = True

        self._traverse_cache[cache_key] = (now, copy.deepcopy(normalized))
        return copy.deepcopy(normalized)

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
        if not entity or not query:
            return []

        try:
            slots = self.api_service.search_slots(
                entity,
                query,
                ledger_id=ledger_id,
                mode="slots",
                limit=limit,
            )
        except requests.RequestException as exc:
            logger.warning(
                "Failed to search ledger slots for entity %s: %s", entity, exc
            )
            return []
        normalized: list[dict] = []
        for slot in slots:
            if isinstance(slot, Mapping):
                normalized.append(dict(slot))
        if limit is not None:
            return normalized[: int(limit)]
        return normalized

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
        query_hint = " ".join([kw for kw in (keywords or []) if kw]) or "recent ledger context"
        try:
            slots = self.api_service.search_slots(
                entity,
                query_hint,
                ledger_id=ledger_id,
                mode="slots",
                limit=limit,
            )
        except Exception:
            slots = []
        snippets: list[str] = []
        for slot in slots:
            title = slot.get("title")
            summary = slot.get("summary")
            body_source = slot.get("body")
            if isinstance(body_source, Sequence) and not isinstance(body_source, (str, bytes)):
                body = [chunk for chunk in body_source if isinstance(chunk, str)]
            elif isinstance(body_source, str):
                body = [body_source]
            else:
                body = []
            snippet = summary or (body[0] if body else title)
            snippet = (snippet or "").strip()
            if not snippet:
                continue
            tags = slot.get("tags") or []
            label = f" [tags: {', '.join(tags[:3])}]" if tags else ""
            snippets.append(f"- {snippet}{label}")
            if len(snippets) >= limit:
                break
        if not snippets:
            entries = self.memory_lookup(
                entity,
                ledger_id=ledger_id,
                limit=limit,
                since=since,
            )
            normalized_keywords = [k.lower() for k in (keywords or []) if len(k) >= 3]
            for entry in entries:
                raw_text = (
                    entry.get("summary")
                    or entry.get("text")
                    or entry.get("snippet")
                )
                text = strip_ledger_noise((raw_text or "").strip())
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
        mode: str | None = None,
    ) -> str | None:
        if not entity:
            return None

        resolved_limit = limit if limit is not None else estimate_quote_count(query)
        search_mode = mode or "all"
        payload = self.api_service.search(
            entity,
            query,
            ledger_id=ledger_id,
            mode=search_mode,
            limit=resolved_limit,
        )
        response = payload.get("response") if isinstance(payload, Mapping) else None
        if isinstance(response, str):
            response = response.strip()
        return response or None

    # ------------------------------------------------------------------
    # Structured ledger helpers
    # ------------------------------------------------------------------
    def realign_with_ledger(
        self,
        entity: str | None,
        *,
        ledger_id: str | None = None,
        quote_safe: bool | None = None,
    ) -> None:
        """Clear caches and warm context following Möbius/enrichment events."""

        if not entity:
            return

        self.clear_entity_cache(entity=entity, ledger_id=ledger_id)
        try:
            self.assemble_context(
                entity,
                ledger_id=ledger_id,
                quote_safe=quote_safe,
            )
        except Exception:
            # Assembly fetch is best-effort; surface resets still succeed even
            # if the remote API temporarily fails.
            return
        key = (entity, ledger_id or None)
        state = self._mobius_state.setdefault(key, {"last_rotation": 0.0, "last_refresh": 0.0})
        state["last_refresh"] = time.time()

    def note_mobius_rotation(
        self,
        entity: str | None,
        *,
        ledger_id: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        if not entity:
            return
        key = (entity, ledger_id or None)
        state = self._mobius_state.setdefault(key, {"last_rotation": 0.0, "last_refresh": 0.0})
        recorded = timestamp if isinstance(timestamp, (int, float)) else time.time()
        state["last_rotation"] = float(recorded)

    def maybe_refresh_mobius_alignment(
        self,
        entity: str | None,
        *,
        ledger_id: str | None = None,
        now: float | None = None,
        force: bool = False,
    ) -> bool:
        if not entity:
            return False
        key = (entity, ledger_id or None)
        state = self._mobius_state.get(key)
        if not state:
            return False
        current_time = float(now if isinstance(now, (int, float)) else time.time())
        last_rotation = float(state.get("last_rotation", 0.0))
        last_refresh = float(state.get("last_refresh", 0.0))
        if not force and (not last_rotation or current_time - last_refresh < MOBIUS_REFRESH_INTERVAL):
            return False
        self.realign_with_ledger(entity, ledger_id=ledger_id)
        refreshed_state = self._mobius_state.setdefault(
            key, {"last_rotation": last_rotation, "last_refresh": current_time}
        )
        refreshed_state["last_refresh"] = current_time
        return True


def _mentions_ledger_lookup(text: str) -> bool:
    if "ledger" not in text:
        return False
    return any(hint in text for hint in _LEDGER_LOOKUP_HINTS)


def is_recall_query(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    if normalized.startswith(_PREFIXES):
        return True
    if _QUOTE_KEYWORD_PATTERN.search(normalized):
        return True
    if _mentions_ledger_lookup(normalized):
        return True
    return False
