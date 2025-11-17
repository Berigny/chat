"""Inference status tab renderer."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import streamlit as st

from services.api_helpers import fetch_inference_state, get_ledger_id
from ui_components import render_metrics_card


def render_tab(session_state) -> None:
    entity = session_state.get("entity")
    if not entity:
        st.info("Select an entity to view inference status.")
        return

    payload = fetch_inference_state(
        session_state,
        entity=entity,
        ledger_id=get_ledger_id(session_state),
        include_history=True,
        limit=10,
    )
    if not payload.get("supported", True):
        st.info("Inference state endpoint unavailable on this backend.")
        return

    status = payload.get("status") if isinstance(payload.get("status"), str) else None
    if status:
        st.markdown(f"**State:** {status}")
    active = payload.get("active") if isinstance(payload.get("active"), Mapping) else None
    if active:
        st.markdown(f"**Active:** {_format_row(active)}")
    queue = payload.get("queue") if isinstance(payload.get("queue"), Sequence) else []
    if queue:
        st.subheader("Queue")
        for entry in queue[:10]:
            summary = _format_row(entry)
            if summary:
                st.caption(f"• {summary}")
    history = payload.get("history") if isinstance(payload.get("history"), Sequence) else []
    if history:
        st.subheader("Recent Completions")
        for entry in history[:10]:
            summary = _format_row(entry)
            if summary:
                st.caption(f"• {summary}")
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    if metrics:
        render_metrics_card("Metrics", metrics)
    message = payload.get("message") if isinstance(payload.get("message"), str) else None
    if message and not (queue or history or active):
        st.info(message)


def _format_row(entry: Mapping[str, Any] | None) -> str:
    if not entry:
        return ""
    label = entry.get("label") if isinstance(entry.get("label"), str) else None
    if not label and isinstance(entry.get("prime"), int):
        label = f"Prime {entry['prime']}"
    status = entry.get("status") if isinstance(entry.get("status"), str) else None
    score = entry.get("score") if isinstance(entry.get("score"), (int, float)) else None
    note = entry.get("note") if isinstance(entry.get("note"), str) else None
    timestamp = entry.get("timestamp") if isinstance(entry.get("timestamp"), (int, float)) else None
    ts_label = None
    if timestamp is not None:
        try:
            from datetime import datetime

            ts_label = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M")
        except (ValueError, OverflowError, OSError):
            ts_label = None
    parts = [part for part in (label, status, ts_label) if part]
    summary = " | ".join(parts) if parts else "(entry)"
    if score is not None:
        summary = f"{summary} — score {score:.2f}"
    if note:
        summary = f"{summary} — {note}"
    return summary
