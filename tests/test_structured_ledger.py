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

    def fake_ingest(self, entity, payload, *, ledger_id=None):
        captured.update(entity=entity, payload=payload, ledger_id=ledger_id)
        return {
            "structured": {
                "slots": [{"prime": 2, "title": "Meeting"}],
                "s1": [],
                "s2": [],
                "bodies": [],
            }
        }

    monkeypatch.setattr(chat_demo_app, "API_SERVICE", type("Svc", (), {"ingest": fake_ingest})())
    monkeypatch.setattr(chat_demo_app, "PRIME_SERVICE", type("Prime", (), {"build_factors": lambda *_, **__: [{"prime": 2, "delta": 1}]})())
    monkeypatch.setattr(chat_demo_app, "_refresh_capabilities_block", lambda: "")

    result = chat_demo_app._anchor("Test entry", record_chat=False, notify=False)

    assert result is True
    assert captured["entity"] == "demo"
    assert captured["payload"]["text"] == "Test entry"
    assert st.session_state.latest_structured_ledger["slots"][0]["prime"] == 2
