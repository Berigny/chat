"""Helpers for persisting structured ledger payloads via the API."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from services.api import ApiService


_ALLOWED_S2_PRIME_KEYS = {"11", "13", "17", "19"}

def _sanitize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = []
            for entry in value:
                if isinstance(entry, (str, int, float, bool)) or entry is None:
                    items.append(entry)
            if items:
                sanitized[key] = items
    return sanitized


def _coerce_s1_slot(slot: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(slot, Mapping):
        return None
    prime = slot.get("prime")
    body_prime = slot.get("body_prime")
    if not isinstance(prime, int) or not isinstance(body_prime, int):
        return None
    payload: dict[str, Any] = {
        "prime": prime,
        "value": int(slot.get("value", 1)) or 1,
        "body_prime": body_prime,
    }
    title = slot.get("title")
    if isinstance(title, str) and title.strip():
        payload["title"] = title.strip()
    tags = slot.get("tags")
    if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes, bytearray)):
        cleaned = [str(tag).strip() for tag in tags if str(tag).strip()]
        if cleaned:
            payload["tags"] = cleaned
    metadata = _sanitize_metadata(slot.get("metadata"))
    if metadata:
        payload["metadata"] = metadata
    score = slot.get("score")
    if isinstance(score, (int, float)):
        payload["score"] = float(score)
    timestamp = slot.get("timestamp")
    if isinstance(timestamp, (int, float)):
        payload["timestamp"] = int(timestamp)
    return payload


def _coerce_s2_slot(
    slot: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(slot, Mapping):
        return None
    prime = slot.get("prime")
    body_prime = slot.get("body_prime")
    if not isinstance(prime, int) or not isinstance(body_prime, int):
        return None
    payload: dict[str, Any] = {
        "prime": prime,
        "body_prime": body_prime,
    }
    summary = slot.get("summary")
    if isinstance(summary, str) and summary.strip():
        sanitized_summary = summary.strip()
        payload["summary"] = sanitized_summary
    metadata = _sanitize_metadata(slot.get("metadata"))
    if metadata:
        payload["metadata"] = metadata
    score = slot.get("score")
    if isinstance(score, (int, float)):
        normalized_score = float(score)
        payload["score"] = normalized_score
    timestamp = slot.get("timestamp")
    if isinstance(timestamp, (int, float)):
        normalized_timestamp = int(timestamp)
        payload["timestamp"] = normalized_timestamp
    tags = slot.get("tags")
    if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes, bytearray)):
        cleaned_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        if cleaned_tags:
            payload["tags"] = cleaned_tags
    return payload


def write_s1_slots(
    api_service: ApiService,
    entity: str,
    slots: Sequence[Mapping[str, Any]] | None,
    *,
    ledger_id: str | None = None,
) -> list[dict[str, Any]]:
    sanitized = [slot for slot in (_coerce_s1_slot(item) for item in slots or []) if slot]
    if sanitized:
        api_service.put_ledger_s1(entity, {"slots": sanitized}, ledger_id=ledger_id)
    return sanitized


def write_s2_slots(
    api_service: ApiService,
    entity: str,
    slots: Sequence[Mapping[str, Any]] | None,
    *,
    ledger_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    sanitized_map: dict[str, dict[str, Any]] = {}
    for item in slots or []:
        payload = _coerce_s2_slot(item)
        if not payload:
            continue
        prime_key = str(payload["prime"])
        if prime_key not in _ALLOWED_S2_PRIME_KEYS:
            continue
        sanitized_map[prime_key] = payload
    if sanitized_map:
        api_service.put_ledger_s2(entity, sanitized_map, ledger_id=ledger_id)
    return sanitized_map


def write_structured_views(
    api_service: ApiService,
    entity: str,
    structured: Mapping[str, Any] | None,
    *,
    ledger_id: str | None = None,
) -> dict[str, Any]:
    structured = structured or {}
    s1_payload = write_s1_slots(api_service, entity, structured.get("s1"), ledger_id=ledger_id)
    s2_payload = write_s2_slots(api_service, entity, structured.get("s2"), ledger_id=ledger_id)
    result = {
        "slots": list(structured.get("slots", []) or []),
        "s1": s1_payload,
        "s2": s2_payload,
        "bodies": list(structured.get("bodies", []) or []),
    }
    return result


__all__ = ["write_structured_views", "write_s1_slots", "write_s2_slots"]
