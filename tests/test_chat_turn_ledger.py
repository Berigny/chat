import time

import streamlit as st

import chat_demo_app


def test_update_rolling_memory_writes_turn_entry(monkeypatch):
    st.session_state.clear()
    st.session_state.entity = "demo"
    st.session_state.ledger_id = "ledger-omega"
    st.session_state.chat_history = []
    st.session_state.rolling_text = []
    st.session_state.last_anchor_ts = time.time()

    recorded_entries: list[dict] = []

    class RecordingBackend:
        def write_ledger_entry(self, **kwargs):
            recorded_entries.append(kwargs)
            return {
                "entry_id": f"{kwargs['key_namespace']}:{kwargs['key_identifier']}",
                "state": {
                    "coordinates": dict(kwargs.get("coordinates") or {}),
                    "metadata": dict(kwargs.get("metadata") or {}),
                },
            }

    monkeypatch.setattr(chat_demo_app.PRIME_SERVICE, "backend_client", RecordingBackend())

    chat_demo_app._update_rolling_memory("User turn", "Bot reply")

    assert recorded_entries, "Chat turns should be written immediately"
    entry = recorded_entries[-1]
    assert entry["key_namespace"] == "ledger-omega"
    assert entry["key_identifier"] == "demo-chat-turn"
    assert entry["coordinates"].get("prime_2", 0) > 0
    assert entry["coordinates"].get("prime_3", 0) > 0
    assert entry["metadata"].get("user_text") == "User turn"
    assert entry["metadata"].get("bot_reply") == "Bot reply"
