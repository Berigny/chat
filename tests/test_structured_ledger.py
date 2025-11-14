import time
import streamlit as st

import chat_demo_app
from prime_schema import DEFAULT_PRIME_SCHEMA


def test_anchor_persists_structured_views(monkeypatch):
    st.session_state.clear()
    st.session_state.entity = "demo"
    st.session_state.prime_schema = DEFAULT_PRIME_SCHEMA
    st.session_state.ledger_id = "ledger-alpha"
    st.session_state.chat_history = []
    st.session_state.last_anchor_error = None
    st.session_state.rolling_text = []
    st.session_state.last_anchor_ts = time.time()
    st.session_state.latest_structured_ledger = {}

    captured: dict[str, object] = {}
    persisted: dict[str, object] = {}

    class DummyPrimeService:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def ingest(self, entity, text, schema, *, ledger_id=None, **kwargs):
            record = {
                "entity": entity,
                "text": text,
                "schema": schema,
                "ledger_id": ledger_id,
                "kwargs": kwargs,
            }
            self.calls.append(record)
            captured.update(record)
            return {
                "structured": {
                    "slots": [{"prime": 2, "title": "Meeting"}],
                    "s1": [],
                    "s2": [],
                    "bodies": [],
                },
                "anchor": {"edges": [], "energy": 1.0},
            }

    dummy_prime = DummyPrimeService()

    def fake_write(_api_service, entity, structured, *, ledger_id=None):
        persisted.update(entity=entity, structured=structured, ledger_id=ledger_id)
        return structured

    monkeypatch.setattr(chat_demo_app, "API_SERVICE", object())
    monkeypatch.setattr(chat_demo_app, "PRIME_SERVICE", dummy_prime)
    monkeypatch.setattr(chat_demo_app, "write_structured_views", fake_write)
    monkeypatch.setattr(chat_demo_app, "_refresh_capabilities_block", lambda: "")

    result = chat_demo_app._anchor("Test entry", record_chat=False, notify=False)

    assert result is True
    assert captured["entity"] == "demo"
    assert captured["text"] == "Test entry"
    assert persisted["entity"] == "demo"
    assert persisted["ledger_id"] == "ledger-alpha"
    assert st.session_state.latest_structured_ledger["slots"][0]["prime"] == 2
