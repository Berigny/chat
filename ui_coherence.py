"""Streamlit UI for coherence evaluation."""

from __future__ import annotations

import os
from typing import Any, Mapping

import requests
import streamlit as st

from api_client import DualSubstrateV2Client
from ui_components import render_json_viewer
from utils_streamlit import parse_json_input, show_api_error

DEFAULT_API = os.getenv("DUALSUBSTRATE_API", "https://dualsubstrate-commercial.fly.dev")


def _client(api_url: str, api_key: str | None) -> DualSubstrateV2Client:
    return DualSubstrateV2Client(api_url, api_key=api_key)


def _coherence_payload(entity: str, text: str, deltas: Any, metadata: Any) -> Mapping[str, Any]:
    payload: dict[str, Any] = {"entity": entity}
    if text:
        payload["text"] = text
    if deltas is not None:
        payload["deltas"] = deltas
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


def render() -> None:
    """Render the coherence evaluation dashboard."""

    st.title("Coherence Evaluator")
    st.caption("Submit candidate deltas and text to the v2 coherence endpoint.")

    api_url = st.text_input("API URL", value=DEFAULT_API, key="coherence_api_url")
    api_key = st.text_input(
        "API Key",
        value=os.getenv("DUALSUBSTRATE_API_KEY", ""),
        type="password",
        key="coherence_api_key",
    )

    col_entity, col_text = st.columns(2)
    entity = col_entity.text_input("Entity", value="demo_user")
    text = col_text.text_input("Text", value="", placeholder="Optional text to assess")

    deltas_raw = st.text_area(
        "Prime deltas JSON",
        value='[{"prime": 2, "delta": 1}]',
        height=140,
        help="Provide a JSON array of prime deltas.",
    )
    metadata_raw = st.text_area(
        "Metadata JSON",
        value="{}",
        height=100,
        help="Optional metadata forwarded to the API.",
    )

    deltas, deltas_error = parse_json_input(deltas_raw)
    metadata, metadata_error = parse_json_input(metadata_raw)
    if deltas_error:
        st.warning(deltas_error)
    if metadata_error:
        st.warning(metadata_error)

    if st.button("Evaluate Coherence", type="primary"):
        client = _client(api_url, api_key or None)
        payload = _coherence_payload(entity, text, deltas, metadata)
        try:
            response = client.evaluate_coherence(payload)
        except requests.RequestException as exc:
            show_api_error(exc)
            return

        st.success("Coherence evaluation completed.")
        st.metric("Overall Coherence", f"{response.score.value:.2f}")

        if response.strains:
            st.subheader("Strains")
            for strain in response.strains:
                cols = st.columns([2, 1, 1, 3])
                cols[0].markdown(f"**{strain.name}**")
                cols[1].metric("Score", f"{strain.score:.2f}")
                cols[2].metric("Weight", f"{strain.weight:.2f}" if strain.weight is not None else "–")
                if strain.notes:
                    cols[3].write(strain.notes)

        if response.notes:
            st.subheader("Notes")
            for note in response.notes:
                st.info(note)

        render_json_viewer("Raw response", response.raw or {})


def render_coherence_tab(client: DualSubstrateV2Client | None = None) -> None:
    """Render the coherence tab with a shared API client."""

    if client:
        st.session_state.setdefault("coherence_api_url", client.base_url)
        st.session_state.setdefault("coherence_api_key", client.api_key or "")
    render()


__all__ = ["render", "render_coherence_tab"]
