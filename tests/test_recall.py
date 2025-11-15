from types import SimpleNamespace

import requests
import streamlit as st

import admin_app
from prime_schema import DEFAULT_PRIME_SCHEMA


class _DummyApi(SimpleNamespace):
    def __init__(self, *, payload=None):
        super().__init__(payload=payload or {})
        self.calls: list[dict] = []

    def search(self, entity, query, **kwargs):
        self.calls.append({"entity": entity, "query": query, **kwargs})
        return self.payload


def test_recall_skips_when_engine_returns_no_response(monkeypatch):
    st.session_state.clear()
    st.session_state.entity = "tester"
    st.session_state.prime_schema = DEFAULT_PRIME_SCHEMA
    st.session_state.chat_history = []

    api = _DummyApi(payload={"response": ""})
    monkeypatch.setattr(admin_app, "_api_service", lambda: api)

    handled = admin_app._maybe_handle_recall_query("what did we discuss")

    assert not handled
    assert st.session_state.chat_history == []
    assert api.calls and api.calls[0]["mode"] == "s1"


def test_recall_records_engine_response(monkeypatch):
    st.session_state.clear()
    st.session_state.entity = "tester"
    st.session_state.prime_schema = DEFAULT_PRIME_SCHEMA
    st.session_state.chat_history = []

    st.session_state.recall_mode = "body"

    payload = {
        "response": "Here’s what the ledger currently recalls:\n• Meeting summary",
        "slots": [{"prime": 11, "summary": "Meeting summary"}],
        "s2": {"11": {"summary": "Meeting summary"}},
    }
    api = _DummyApi(payload=payload)
    monkeypatch.setattr(admin_app, "_api_service", lambda: api)
    anchored: list = []
    monkeypatch.setattr(admin_app, "_anchor", lambda *a, **k: anchored.append((a, k)) or True)

    handled = admin_app._maybe_handle_recall_query("meeting recap")

    assert handled
    assert st.session_state.chat_history
    entry = st.session_state.chat_history[-1]
    assert entry["role"] == "assistant"
    assert entry["content"].startswith("Here’s what the ledger currently recalls")
    assert entry["recall"]["slots"][0]["summary"] == "Meeting summary"
    assert anchored, "Expected recall anchoring"
    assert api.calls[0]["ledger_id"] == admin_app._ledger_id()
    assert api.calls[0]["mode"] == "body"


def test_recall_surfaces_http_error(monkeypatch):
    st.session_state.clear()
    st.session_state.entity = "tester"
    st.session_state.prime_schema = DEFAULT_PRIME_SCHEMA
    st.session_state.chat_history = []

    def failing_api():
        raise requests.RequestException("boom")

    monkeypatch.setattr(admin_app, "_api_service", lambda: SimpleNamespace(search=lambda *a, **k: failing_api()))

    errors: list[str] = []
    monkeypatch.setattr(st, "error", lambda message: errors.append(message))

    handled = admin_app._maybe_handle_recall_query("meeting recap")

    assert not handled
    assert errors and "boom" in errors[0]
    assert st.session_state.chat_history == []
