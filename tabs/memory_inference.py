"""Memory and inference tab renderer."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import streamlit as st


RenderCallback = Callable[[str | None], None]


def _render_section(title: str, callback: RenderCallback, entity: str | None) -> None:
    with st.expander(title, expanded=True):
        callback(entity)


def render_tab(
    *,
    entity: str | None,
    traversal_supported: bool,
    inference_supported: bool,
    render_traversal_callback: RenderCallback,
    render_inference_callback: RenderCallback,
    inference_snapshot: Mapping[str, Any] | None,
) -> None:
    """Render traversal, inference, and telemetry panels."""

    if traversal_supported:
        _render_section("Traversal snapshot", render_traversal_callback, entity)
    else:
        st.info("Traversal endpoint unavailable on this deployment.")

    st.divider()

    if inference_supported:
        _render_section("Inference status", render_inference_callback, entity)
    else:
        st.info("Inference state endpoint unavailable on this deployment.")

    telemetry_snapshot = inference_snapshot or {}
    telemetry_state = telemetry_snapshot.get("inference_state")
    telemetry_traverse = telemetry_snapshot.get("inference_traverse")
    telemetry_memories = telemetry_snapshot.get("inference_memories")
    telemetry_retrieve = telemetry_snapshot.get("inference_retrieve")
    telemetry_supported = telemetry_snapshot.get("inference_supported")
    telemetry_errors = telemetry_snapshot.get("inference_errors") or []

    telemetry_payloads = [
        ("State", telemetry_state),
        ("Traverse", telemetry_traverse),
        ("Memories", telemetry_memories),
        ("Retrieve", telemetry_retrieve),
    ]
    telemetry_any = any(payload is not None for _, payload in telemetry_payloads)

    if telemetry_any:
        st.divider()
        st.markdown("#### Inference telemetry")
        for label, payload in telemetry_payloads:
            if payload is None:
                continue
            expanded = label == "State"
            with st.expander(label, expanded=expanded):
                if isinstance(payload, (list, dict)):
                    st.json(payload)
                else:
                    st.write(payload)
    elif telemetry_supported is False:
        st.caption("Inference telemetry endpoints are not available on this deployment.")
    elif telemetry_errors:
        st.warning("Inference telemetry unavailable: " + "; ".join(telemetry_errors))
