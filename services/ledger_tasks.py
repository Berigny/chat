"""Utility helpers for ledger-centric operations (metrics, resets, enrichment)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests
import time

def fetch_metrics_snapshot(
    api_service,
    entity: str | None,
    *,
    ledger_id: str | None,
    metric_floors: Dict[str, float],
) -> Dict[str, Any]:
    """Return sanitized metrics plus durability estimates."""

    metrics: Dict[str, Any] = {}
    metrics_error: Optional[str] = None
    memories: list[dict] = []
    if entity:
        try:
            metrics = api_service.fetch_metrics(ledger_id=ledger_id)
        except requests.RequestException as exc:
            metrics_error = str(exc)
        try:
            memories = api_service.fetch_memories(entity, ledger_id=ledger_id, limit=1)
        except requests.RequestException:
            memories = []
    tokens_saved = metrics.get("tokens_deduped")
    if tokens_saved is None:
        tokens_saved = metric_floors.get("tokens_deduped", 0)
    integrity = metrics.get("ledger_integrity")
    if integrity is None:
        integrity = metric_floors.get("ledger_integrity", 1.0)
    durability = metrics.get("durability_hours")
    if durability is None:
        if memories:
            timestamp = memories[-1].get("timestamp")
            if timestamp:
                durability = max(
                    metric_floors.get("durability_h", 0.0),
                    (time.time() - timestamp / 1000) / 3600,
                )
    if durability is None:
        durability = metric_floors.get("durability_h", 0.0)
    return {
        "tokens_saved": tokens_saved,
        "ledger_integrity": integrity,
        "durability_hours": durability,
        "raw_metrics": metrics,
        "error": metrics_error,
    }


def reset_discrete_ledger(
    api_service,
    prime_service,
    entity: str,
    *,
    ledger_id: str | None,
    schema: Dict[int, Dict[str, object]],
) -> tuple[bool, str | None]:
    """Drive ledger factors back to zero by anchoring negative deltas one prime at a time."""

    try:
        ledger_payload = api_service.fetch_ledger(entity, ledger_id=ledger_id)
    except requests.RequestException as exc:
        return False, f"Could not fetch ledger for reset: {exc}"

    factors = ledger_payload.get("factors") if isinstance(ledger_payload, dict) else None
    if not isinstance(factors, list):
        return True, None

    for entry in factors:
        prime = entry.get("prime")
        value = entry.get("value", 0)
        if not isinstance(prime, int) or not value:
            continue
        delta = -int(value)
        override = [{"prime": prime, "delta": delta}]
        try:
            prime_service.anchor(
                entity,
                f"[reset] prime {prime}",
                schema,
                ledger_id=ledger_id,
                factors_override=override,
            )
        except requests.RequestException as exc:
            return False, f"Reset failed for prime {prime}: {exc}"
    return True, None


def run_enrichment_job(
    api_service,
    prime_service,
    entity: str,
    *,
    ledger_id: str | None,
    schema: Dict[int, Dict[str, object]],
    llm_extractor,
    limit: int = 200,
    reset_first: bool = True,
) -> Dict[str, Any]:
    """Replay stored transcripts with richer prime coverage."""

    try:
        memories = api_service.fetch_memories(
            entity,
            ledger_id=ledger_id,
            limit=max(1, min(limit, 100)),
        )
    except requests.RequestException as exc:
        return {"error": f"Failed to load memories: {exc}"}
    if not memories:
        return {"message": "No memories found to enrich.", "enriched": 0, "total": 0}

    if reset_first:
        ok, error = reset_discrete_ledger(
            api_service,
            prime_service,
            entity,
            ledger_id=ledger_id,
            schema=schema,
        )
        if not ok:
            return {"error": error or "Reset failed before enrichment."}

    enriched = 0
    failures: list[str] = []
    for entry in memories:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        try:
            prime_service.anchor(
                entity,
                text,
                schema,
                ledger_id=ledger_id,
                llm_extractor=llm_extractor,
            )
            enriched += 1
        except requests.RequestException as exc:
            failures.append(f"{entry.get('timestamp', 'unknown')}: {exc}")
    result: Dict[str, Any] = {"enriched": enriched, "total": len(memories)}
    if failures:
        result["failures"] = failures
    return result


def perform_lattice_rotation(
    api_service,
    entity: str,
    *,
    ledger_id: str | None,
    axis: tuple[float, float, float],
    angle: float,
) -> Dict[str, Any]:
    """Invoke /rotate with the supplied parameters."""

    return api_service.rotate(
        entity,
        ledger_id=ledger_id,
        axis=axis,
        angle=angle,
    )


__all__ = [
    "fetch_metrics_snapshot",
    "perform_lattice_rotation",
    "reset_discrete_ledger",
    "run_enrichment_job",
]
