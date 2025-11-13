from types import SimpleNamespace

import streamlit as st

import admin_app
from prime_schema import DEFAULT_PRIME_SCHEMA
from services.memory_service import MemoryService


class _DummyApi:
    def __init__(self, *, query_response=None, memories=None):
        self.query_response = query_response or {}
        self.memories = memories or []
        self.query_calls = []

    def query_ledger(self, entity, query, **kwargs):
        self.query_calls.append({"entity": entity, "query": query, **kwargs})
        return self.query_response

    def fetch_memories(self, entity, **kwargs):
        return self.memories


def test_recall_no_matches(monkeypatch):
    st.session_state.clear()
    st.session_state.entity = "tester"
    st.session_state.prime_schema = DEFAULT_PRIME_SCHEMA
    st.session_state.chat_history = []

    api = _DummyApi(memories=[{"text": "Met Priya about launch."}])
    monkeypatch.setattr(admin_app, "_api_service", lambda: api)

    handled = admin_app._maybe_handle_recall_query("definitions of God")
    assert handled
    entry = st.session_state.chat_history[-1]
    assert entry["content"] == "No stored memories matched “definitions of God” yet."
    assert entry["recent_memories"] == ["Met Priya about launch."]
    assert api.query_calls, "Query endpoint was not invoked"


def test_recall_structured_response(monkeypatch):
    st.session_state.clear()
    st.session_state.entity = "tester"
    st.session_state.prime_schema = DEFAULT_PRIME_SCHEMA
    st.session_state.chat_history = []

    query_payload = {
        "slots": [
            {"prime": 11, "summary": "Meeting summary", "score": 0.9},
            {"prime": 3, "title": "Action items", "tags": ["follow-up"], "score": 0.7},
        ],
        "memories": [{"text": "Meeting summary"}],
    }
    api = _DummyApi(query_response=query_payload)
    monkeypatch.setattr(admin_app, "_api_service", lambda: api)
    monkeypatch.setattr(admin_app, "_anchor", lambda *args, **kwargs: True)

    handled = admin_app._maybe_handle_recall_query("meeting recap")
    assert handled
    entry = st.session_state.chat_history[-1]
    assert entry["role"] == "assistant"
    assert "Meeting summary" in entry["content"]
    assert entry["recall"]["s2"], "Structured S2 entries missing"
    assert api.query_calls[0]["query"] == "meeting recap"


def test_recall_filters_transcript_memories():
    api = SimpleNamespace(
        fetch_memories=lambda *a, **k: [
            {
                "text": (
                    "You: Did we finalise the investor deck yet?\n"
                    "Assistant: Not yet, but we scheduled time tomorrow."
                ),
                "timestamp": 1_700_000_000_000,
            },
            {
                "text": "Met Priya to review the investor deck timeline and action items for launch.",
                "timestamp": 1_700_000_100_000,
            },
        ]
    )
    service = MemoryService(api, {2: 1.0})

    response = service.build_recall_response(
        "demo",
        "recall investor deck details",
        DEFAULT_PRIME_SCHEMA,
        limit=2,
    )

    assert response is not None
    assert "Met Priya to review the investor deck timeline" in response
    assert "You:" not in response

