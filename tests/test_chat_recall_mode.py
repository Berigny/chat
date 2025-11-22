from types import SimpleNamespace

import streamlit as st

import chat_demo_app
from prime_schema import DEFAULT_PRIME_SCHEMA


class _MemoryStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def build_recall_response(self, *_, **kwargs):
        self.calls.append(kwargs)
        return "Here’s what the ledger currently recalls:\n• Sample memory"


class _ApiStub(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.metrics_calls: list[dict] = []

    def patch_metrics(self, entity, payload, *, ledger_id=None):
        self.metrics_calls.append({"entity": entity, "payload": payload, "ledger_id": ledger_id})


def _prepare_session_state() -> None:
    st.session_state.clear()
    st.session_state.entity = "tester"
    st.session_state.prime_schema = DEFAULT_PRIME_SCHEMA
    st.session_state.chat_history = []


def test_chat_recall_uses_body_mode_by_default(monkeypatch):
    _prepare_session_state()
    st.session_state.recall_mode = "body"
    memory_stub = _MemoryStub()
    monkeypatch.setattr(chat_demo_app, "MEMORY_SERVICE", memory_stub)
    monkeypatch.setattr(chat_demo_app, "API_SERVICE", _ApiStub())

    handled = chat_demo_app._maybe_handle_recall_query("what did we discuss yesterday?")

    assert handled
    assert memory_stub.calls
    assert memory_stub.calls[0]["mode"] == "body"
    assert st.session_state.recall_mode == "body"


def test_chat_recall_detects_body_mode_from_query(monkeypatch):
    _prepare_session_state()
    st.session_state.recall_mode = "body"
    memory_stub = _MemoryStub()
    monkeypatch.setattr(chat_demo_app, "MEMORY_SERVICE", memory_stub)
    monkeypatch.setattr(chat_demo_app, "API_SERVICE", _ApiStub())

    handled = chat_demo_app._maybe_handle_recall_query("quote the attachment body for Gödel's incompleteness note")

    assert handled
    assert memory_stub.calls
    assert memory_stub.calls[0]["mode"] == "body"
    assert st.session_state.recall_mode == "body"
