"""Streamlit UI for ethics and policy evaluation."""

from __future__ import annotations

import os
from typing import Any, Mapping

import requests
import streamlit as st

from api_client import DualSubstrateV2Client, build_action_request
from ui_components import render_json_viewer
from utils_streamlit import parse_json_input, show_api_error

DEFAULT_API = os.getenv("DUALSUBSTRATE_API", "https://dualsubstrate-commercial.fly.dev")


def _client(api_url: str, api_key: str | None) -> DualSubstrateV2Client:
    return DualSubstrateV2Client(api_url, api_key=api_key)


def render() -> None:
    """Render the ethics evaluation dashboard."""

    st.title("Ethics & Policy Evaluator")
    st.caption("Send ledger state and enrichment deltas to the ethics guardrail endpoint.")

    api_url = st.text_input("API URL", value=DEFAULT_API, key="ethics_api_url")
    api_key = st.text_input(
        "API Key", value=os.getenv("DUALSUBSTRATE_API_KEY", ""), type="password", key="ethics_api_key"
    )
    entity = st.text_input("Entity", value="Demo_dev", key="ethics_entity")

    ledger_raw = st.text_area(
        "Ledger snapshot JSON",
        value="{}",
        height=140,
        help="Optional ledger snapshot to provide context for the evaluation.",
    )
    deltas_raw = st.text_area(
        "Prime deltas JSON",
        value='[{"prime": 2, "delta": 1}]',
        height=140,
        help="Optional prime deltas associated with the request.",
    )
    minted_raw = st.text_area(
        "Minted bodies JSON",
        value='[{"body": "Generated output"}]',
        height=140,
        help="Optional generated bodies to audit for policy decisions.",
    )

    ledger_snapshot, ledger_error = parse_json_input(ledger_raw)
    deltas, deltas_error = parse_json_input(deltas_raw)
    minted_bodies, minted_error = parse_json_input(minted_raw)

    for error in (ledger_error, deltas_error, minted_error):
        if error:
            st.warning(error)

    if st.button("Evaluate Ethics", type="primary"):
        client = _client(api_url, api_key or None)
        action_request = build_action_request(
            actor=entity or "demo_user",
            action="ethics_probe",
            key_namespace="default",
            key_identifier=f"{(entity or 'demo_user').lower()}-probe",
            parameters={"snapshot": float(bool(ledger_snapshot))},
        )
        try:
            response = client.evaluate_ethics(action_request)
        except requests.RequestException as exc:
            show_api_error(exc)
            return

        st.success("Ethics evaluation completed.")
        st.metric("Decision", response.decision)

        if response.scores:
            st.subheader("Scores")
            columns = st.columns(min(len(response.scores), 3) or 1)
            for index, score in enumerate(response.scores):
                column = columns[index % len(columns)]
                column.metric(score.name.title(), f"{score.value:.2f}")
                if score.rationale:
                    column.caption(score.rationale)

        if response.notes:
            st.subheader("Reviewer Notes")
            for note in response.notes:
                st.info(note)

        if response.metadata:
            render_json_viewer("Metadata", response.metadata)

        render_json_viewer("Raw response", response.raw or {"decision": response.decision})


def render_ethics_tab(client: DualSubstrateV2Client | None = None) -> None:
    """Render the ethics tab with a shared API client."""

    if client:
        st.session_state.setdefault("ethics_api_url", client.base_url)
        st.session_state.setdefault("ethics_api_key", client.api_key or "")
    render()


__all__ = ["render", "render_ethics_tab"]
