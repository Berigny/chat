import time

import streamlit as st

import chat_demo_app
from prime_schema import DEFAULT_PRIME_SCHEMA


def test_anchor_writes_single_ledger_entry(monkeypatch):
    st.session_state.clear()
    st.session_state.entity = "demo"
    st.session_state.prime_schema = DEFAULT_PRIME_SCHEMA
    st.session_state.ledger_id = "ledger-alpha"
    st.session_state.chat_history = []
    st.session_state.last_anchor_error = None
    st.session_state.rolling_text = []
    st.session_state.last_anchor_ts = time.time()
    st.session_state.latest_structured_ledger = {}
    st.session_state.latest_structured_metrics = {}

    long_summary = " ".join(["Meeting"] * 80)
    long_follow_up = " ".join(["Follow-up"] * 70)
    backend_calls: list[dict[str, object]] = []

    class DummyBackendClient:
        def write_ledger_entry(self, **kwargs):
            backend_calls.append(kwargs)
            return {
                "entry_id": f"{kwargs['key_namespace']}:{kwargs['key_identifier']}",
                "key": {
                    "namespace": kwargs["key_namespace"],
                    "identifier": kwargs["key_identifier"],
                },
                "state": {
                    "metadata": dict(kwargs.get("metadata") or {}),
                    "coordinates": dict(kwargs.get("coordinates") or {}),
                },
            }

        def read_ledger_entry(self, *_args, **_kwargs):
            raise AssertionError("Backend read should not be called during anchor")

    class DummyPrimeService:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.backend_client = DummyBackendClient()

        def ingest(self, entity, text, schema, *, ledger_id=None, **kwargs):
            structured = {
                "slots": [{"prime": 2, "title": "Meeting"}],
                "s1": [],
                "s2": [
                    {"prime": 11, "summary": long_summary},
                    {"prime": 13, "summary": ""},
                ],
                "raw": {
                    "17": {"summary": None},
                    "19": {"summary": f"  {long_follow_up}  "},
                },
                "bodies": [],
            }
            ledger_entry = self.backend_client.write_ledger_entry(
                key_namespace="default",
                key_identifier="demo-structured",
                text=text.strip(),
                phase="ingest",
                entity=entity,
                metadata={
                    "entity": entity,
                    "text": text.strip(),
                    "ledger_id": ledger_id,
                    "factors": [{"prime": 2, "delta": 1}],
                    "structured": structured,
                    "bodies": [],
                    "source": "chat-demo",
                    "timestamp": time.time(),
                },
                coordinates={"prime_2": 1.0},
            )
            result = {
                "text": text.strip(),
                "factors": [{"prime": 2, "delta": 1}],
                "structured": structured,
                "ledger_entry": ledger_entry,
            }
            self.calls.append(result)
            return result

    dummy_prime = DummyPrimeService()

    monkeypatch.setattr(chat_demo_app, "BACKEND_CLIENT", DummyBackendClient())
    monkeypatch.setattr(chat_demo_app, "PRIME_SERVICE", dummy_prime)
    monkeypatch.setattr(chat_demo_app, "_refresh_capabilities_block", lambda: "")

    result = chat_demo_app._anchor("Test entry", record_chat=False, notify=False)

    assert result is True
    assert backend_calls and len(backend_calls) == 1
    assert st.session_state.last_structured_entry_id == "default:demo-structured"
    assert st.session_state.last_anchor_payload == dummy_prime.calls[-1]["ledger_entry"]
    assert st.session_state.latest_structured_ledger == {
        "11": {"summary": long_summary},
        "19": {"summary": long_follow_up},
    }
    assert st.session_state.latest_structured_metrics == {"prime_2": 1.0}


def test_persist_structured_views_from_backend_entry(monkeypatch):
    st.session_state.clear()
    st.session_state.ledger_id = "ledger-gamma"
    st.session_state.latest_structured_ledger = {}
    st.session_state.latest_structured_metrics = {}
    st.session_state.last_structured_entry_id = "default:demo-structured"

    structured_snapshot = {
        "slots": [{"prime": 11, "summary": "Snapshot"}],
        "s2": [{"prime": 11, "summary": "Snapshot"}],
    }

    class DummyBackendClient:
        def read_ledger_entry(self, entry_id):
            assert entry_id == "default:demo-structured"
            return {
                "state": {
                    "metadata": {"structured": structured_snapshot},
                }
            }

    class DummyAPIService:
        def fetch_ledger(self, entity, *, ledger_id=None):
            raise AssertionError("Legacy fetch should not be called when backend entry exists")

    monkeypatch.setattr(chat_demo_app, "BACKEND_CLIENT", DummyBackendClient())
    monkeypatch.setattr(chat_demo_app, "API_SERVICE", DummyAPIService())

    chat_demo_app._persist_structured_views_from_ledger("demo")

    assert st.session_state.latest_structured_ledger == {"11": {"summary": "Snapshot"}}


def test_flat_s2_map_drops_empty_summaries():
    structured = {
        "11": {"summary": "   "},
        "13": {"summary": None},
        "17": {"summary": "Agenda"},
        "raw": {
            "19": {"summary": "  Next steps"},
            "s2": [
                {"prime": 11, "summary": ""},
                {"prime": 13, "summary": None},
                {"prime": 19, "summary": "  "},
            ],
        },
        "s2": [
            {"prime": 11, "summary": ""},
            {"prime": 19, "summary": "Final"},
        ],
    }

    result = chat_demo_app._derive_flat_s2_map(structured)

    assert result == {
        "17": {"summary": "Agenda"},
        "19": {"summary": "Final"},
    }
