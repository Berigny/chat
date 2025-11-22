"""Utility helpers for ledger-centric operations (metrics, resets, enrichment)."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests


def _coerce_float(value: Any) -> float | None:
    """Best-effort conversion of Prometheus/JSON values into floats."""

    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _parse_prometheus_metrics(payload: str) -> Dict[str, float]:
    """Parse a subset of Prometheus exposition format into a numeric dict."""

    metrics: Dict[str, float] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            metric_part, value_part = line.rsplit(" ", 1)
        except ValueError:
            continue
        name = metric_part.split("{", 1)[0]
        numeric = _coerce_float(value_part)
        if numeric is None:
            continue
        metrics[name] = numeric
    return metrics


def _resolve_metric(values: Dict[str, float], *candidates: str) -> float | None:
    """Return the first present metric from ``values`` matching ``candidates``."""

    for candidate in candidates:
        if candidate in values:
            return values[candidate]
    return None


def fetch_metrics_snapshot(
    api_service,
    entity: str | None,
    *,
    ledger_id: str | None,
    metric_floors: Dict[str, float],
    advanced_probes_enabled: bool = True,
) -> Dict[str, Any]:
    """Return sanitized metrics plus durability estimates and inference telemetry."""

    metrics_payload: Any = {}
    numeric_metrics: Dict[str, float] = {}
    metrics_error: Optional[str] = None
    memories: list[dict] = []
    inference_state: Dict[str, Any] | None = None
    inference_traverse: list[dict] | None = None
    inference_memories: list[dict] | None = None
    inference_retrieve: Dict[str, Any] | None = None
    inference_errors: list[str] = []
    inference_supported: Optional[bool] = None

    if entity and advanced_probes_enabled:
        inference_supported = True
        try:
            metrics_payload = api_service.fetch_metrics(ledger_id=ledger_id)
        except requests.RequestException as exc:
            metrics_error = str(exc)
            metrics_payload = {}
        else:
            if isinstance(metrics_payload, str):
                numeric_metrics = _parse_prometheus_metrics(metrics_payload)
            elif isinstance(metrics_payload, dict):
                numeric_metrics = {
                    key: coerced
                    for key in metrics_payload
                    if (coerced := _coerce_float(metrics_payload[key])) is not None
                }
            else:
                numeric_metrics = {}

        try:
            memories = api_service.fetch_memories(entity, ledger_id=ledger_id, limit=1)
        except requests.RequestException:
            memories = []

        def _safe_fetch(name: str, call):
            nonlocal inference_supported
            try:
                return call()
            except requests.HTTPError as exc:
                response = exc.response
                if response is not None and response.status_code == 404:
                    inference_supported = False
                    return None
                inference_errors.append(f"{name}: {exc}")
            except requests.RequestException as exc:
                inference_errors.append(f"{name}: {exc}")
            return None

        inference_state = _safe_fetch(
            "state",
            lambda: api_service.fetch_inference_state(entity, ledger_id=ledger_id),
        )
        inference_traverse = _safe_fetch(
            "traverse",
            lambda: api_service.fetch_inference_traverse(entity, ledger_id=ledger_id),
        )
        inference_memories = _safe_fetch(
            "memories",
            lambda: api_service.fetch_inference_memories(entity, ledger_id=ledger_id),
        )
        inference_retrieve = _safe_fetch(
            "retrieve",
            lambda: api_service.fetch_inference_retrieve(entity, ledger_id=ledger_id),
        )
    elif entity:
        try:
            metrics_payload = api_service.fetch_metrics(ledger_id=ledger_id)
        except requests.RequestException as exc:
            metrics_error = str(exc)
            metrics_payload = {}
        else:
            if isinstance(metrics_payload, str):
                numeric_metrics = _parse_prometheus_metrics(metrics_payload)
            elif isinstance(metrics_payload, dict):
                numeric_metrics = {
                    key: coerced
                    for key in metrics_payload
                    if (coerced := _coerce_float(metrics_payload[key])) is not None
                }
            else:
                numeric_metrics = {}

    tokens_saved = _resolve_metric(
        numeric_metrics,
        "dualsubstrate_tokens_deduped_total",
        "dualsubstrate_tokens_deduped",
        "tokens_deduped",
        "tokens_saved",
    )
    if tokens_saved is None:
        tokens_saved = metric_floors.get("tokens_deduped", 0)

    integrity = _resolve_metric(
        numeric_metrics,
        "dualsubstrate_ledger_integrity_ratio",
        "ledger_integrity",
    )
    if integrity is None:
        integrity = metric_floors.get("ledger_integrity", 1.0)

    durability = _resolve_metric(
        numeric_metrics,
        "dualsubstrate_durability_hours",
        "durability_hours",
    )
    if durability is None and memories:
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
        "raw_metrics": metrics_payload,
        "metrics_source": "prometheus" if isinstance(metrics_payload, str) else "json",
        "prometheus_metrics": numeric_metrics if isinstance(metrics_payload, str) else None,
        "error": metrics_error,
        "inference_state": inference_state,
        "inference_traverse": inference_traverse,
        "inference_memories": inference_memories,
        "inference_retrieve": inference_retrieve,
        "inference_supported": inference_supported,
        "inference_errors": inference_errors,
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
