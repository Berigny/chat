import streamlit as st

import admin_app
from prime_schema import DEFAULT_PRIME_SCHEMA


def test_recall_no_matches(monkeypatch):
    st.session_state.clear()
    st.session_state.entity = "tester"
    st.session_state.prime_schema = DEFAULT_PRIME_SCHEMA
    st.session_state.chat_history = []

    monkeypatch.setattr(admin_app, "query_shards", lambda *a, **k: [])

    handled = admin_app._maybe_handle_recall_query("definitions of God")
    assert handled
    assert st.session_state.chat_history[-1]["content"] == "No stored memories matched “definitions of God” yet."

