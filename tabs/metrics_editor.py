"""Entity metrics editor helpers for the ledger tab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import streamlit as st

from services.api import requests


DEFAULT_METRICS_PAYLOAD = {"dE": -1.0, "dDrift": -0.5, "dRetention": 0.8, "K": 0.0}
METRIC_LABELS = {
    "dE": "ΔE",
    "dDrift": "ΔDrift",
    "dRetention": "ΔRetention",
    "K": "K",
}
METRIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("dE", "ΔE (must remain negative to unlock S2/search)"),
    ("dDrift", "ΔDrift (must remain negative)"),
    ("dRetention", "ΔRetention (must remain positive)"),
    ("K", "K (must be ≥ 0)"),
)


@dataclass
class MetricsSnapshot:
    metrics: dict[str, float]
    error: str | None = None


def _coerce_metrics(payload: Mapping[str, Any] | None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not isinstance(payload, Mapping):
        return dict(DEFAULT_METRICS_PAYLOAD)
    for key, default in DEFAULT_METRICS_PAYLOAD.items():
        value = payload.get(key)
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            metrics[key] = float(default)
    return metrics


def fetch_metrics_snapshot(api_service: Any, entity: str, *, ledger_id: str | None) -> MetricsSnapshot:
    try:
        payload = api_service.fetch_ledger(entity, ledger_id=ledger_id)
    except requests.RequestException as exc:
        return MetricsSnapshot(metrics=dict(DEFAULT_METRICS_PAYLOAD), error=str(exc))
    metrics = _coerce_metrics(payload.get("r_metrics") if isinstance(payload, Mapping) else {})
    return MetricsSnapshot(metrics=metrics)


def _apply_metrics_patch(
    api_service: Any,
    entity: str,
    payload: Mapping[str, float],
    *,
    ledger_id: str | None,
) -> Mapping[str, Any] | None:
    try:
        return api_service.patch_metrics(
            entity,
            payload,
            ledger_id=ledger_id,
        )
    except requests.RequestException as exc:
        st.error("Metrics update failed; backend rejected the patch.")
        st.caption(str(exc))
        return None


def render_entity_metrics_panel(
    api_service: Any,
    *,
    entity: str | None,
    ledger_id: str | None,
    advanced_probes_enabled: bool = True,
) -> None:
    """Render current metrics and provide an editor form."""

    st.markdown("### Entity metrics overview")
    if not entity:
        st.info("Select an entity to inspect per-ledger metrics.")
        return

    snapshot = fetch_metrics_snapshot(api_service, entity, ledger_id=ledger_id)
    if snapshot.error:
        st.warning(f"Failed to load metrics: {snapshot.error}")
    metrics = snapshot.metrics
    probes_disabled = not advanced_probes_enabled
    st.caption("Current ℝ metrics (r_metrics) stored for this entity/ledger.")
    if probes_disabled:
        st.caption("Advanced probes are disabled for this deployment.")
    table_rows = []
    for key, _ in METRIC_FIELDS:
        value = metrics.get(key)
        if value is None:
            continue
        label = METRIC_LABELS.get(key, key)
        table_rows.append({"Metric": label, "Value": f"{float(value):.3f}"})
    st.table(table_rows)

    st.caption(
        "ΔE / ΔDrift must remain negative; ΔRetention must remain positive; K must stay ≥ 0."
    )
    form_key_suffix = ledger_id or "default"
    with st.form(f"entity_metrics_form_{form_key_suffix}"):
        updated_payload: dict[str, float] = {}
        col_a, col_b = st.columns(2)
        field_columns = [col_a, col_b]
        for idx, (field, help_text) in enumerate(METRIC_FIELDS):
            column = field_columns[idx % 2]
            with column:
                default_value = metrics.get(field, DEFAULT_METRICS_PAYLOAD[field])
                updated_payload[field] = st.number_input(
                    METRIC_LABELS.get(field, field),
                    value=float(default_value),
                    help=help_text,
                    disabled=probes_disabled,
                    key=f"metric_input_{field}_{form_key_suffix}",
                )
        submitted = st.form_submit_button(
            "Update metrics",
            disabled=probes_disabled,
            help="Not enabled on this deployment" if probes_disabled else None,
        )
        if submitted:
            response = _apply_metrics_patch(api_service, entity, updated_payload, ledger_id=ledger_id)
            if response is not None:
                st.success("Metrics updated.")
                if isinstance(response, Mapping):
                    st.json(response)

    reset_label = f"Reset metrics to defaults ({DEFAULT_METRICS_PAYLOAD})"
    if st.button(
        reset_label,
        key=f"metrics_reset_{form_key_suffix}",
        disabled=probes_disabled,
        help="Not enabled on this deployment" if probes_disabled else None,
    ):
        response = _apply_metrics_patch(api_service, entity, DEFAULT_METRICS_PAYLOAD, ledger_id=ledger_id)
        if response is not None:
            st.success("Metrics reset to defaults.")
            if isinstance(response, Mapping):
                st.json(response)


__all__ = ["render_entity_metrics_panel", "fetch_metrics_snapshot", "DEFAULT_METRICS_PAYLOAD"]
