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
    st.session_state.latest_structured_metrics = {}

    captured: dict[str, object] = {}
    persisted: dict[str, object] = {}
    s2_calls: list[dict[str, object]] = []
    score_calls: list[dict[str, object]] = []

    long_summary = " ".join(["Meeting"] * 80)
    long_follow_up = " ".join(["Follow-up"] * 70)

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
                    "s2": [
                        {
                            "prime": 11,
                            "body_prime": 101,
                            "summary": long_summary,
                        },
                        {"prime": 13, "summary": ""},
                    ],
                    "raw": {
                        "17": {"summary": None},
                        "19": {"summary": f"  {long_follow_up}  "},
                    },
                    "bodies": [],
                },
                "anchor": {"edges": [], "energy": 1.0},
            }

    class DummyApiService:
        def __init__(self) -> None:
            self.patch_calls: list[dict[str, object]] = []

        def put_ledger_s2(self, entity, payload, *, ledger_id=None):
            call = {"entity": entity, "payload": payload, "ledger_id": ledger_id}
            s2_calls.append(call)
            return payload

        def patch_metrics(self, entity, payload, *, ledger_id=None):
            call = {"entity": entity, "payload": payload, "ledger_id": ledger_id}
            self.patch_calls.append(call)
            return {"ledger_integrity": 0.9}

    dummy_prime = DummyPrimeService()
    dummy_api = DummyApiService()

    def fake_write(_api_service, entity, structured, *, ledger_id=None):
        persisted.update(entity=entity, structured=structured, ledger_id=ledger_id)
        return structured

    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_score_post(url, params=None, json=None, headers=None, timeout=None):
        score_calls.append(
            {
                "url": url,
                "params": params,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return DummyResponse({"metrics": {"ledger_integrity": 0.91}})

    monkeypatch.setattr(chat_demo_app, "API_SERVICE", dummy_api)
    monkeypatch.setattr(chat_demo_app, "PRIME_SERVICE", dummy_prime)
    monkeypatch.setattr(chat_demo_app, "write_structured_views", fake_write)
    monkeypatch.setattr(chat_demo_app.requests, "post", fake_score_post)
    monkeypatch.setattr(chat_demo_app, "_refresh_capabilities_block", lambda: "")

    result = chat_demo_app._anchor("Test entry", record_chat=False, notify=False)

    assert result is True
    assert captured["entity"] == "demo"
    assert captured["text"] == "Test entry"
    assert persisted["entity"] == "demo"
    assert persisted["ledger_id"] == "ledger-alpha"
    assert persisted["structured"]["s2"] == []
    assert s2_calls == [
        {
            "entity": "demo",
            "payload": {
                "11": {"summary": long_summary},
                "19": {"summary": long_follow_up},
            },
            "ledger_id": "ledger-alpha",
        }
    ]
    assert score_calls == [
        {
            "url": f"{chat_demo_app.API.rstrip('/')}/score/s2",
            "params": {"entity": "demo"},
            "json": {
                "11": {"summary": long_summary},
                "19": {"summary": long_follow_up},
            },
            "headers": {
                "Content-Type": "application/json",
                "X-Ledger-ID": "ledger-alpha",
                **(
                    {"x-api-key": chat_demo_app.SETTINGS.api_key}
                    if chat_demo_app.SETTINGS.api_key
                    else {}
                ),
            },
            "timeout": 10,
        }
    ]
    assert dummy_api.patch_calls == [
        {
            "entity": "demo",
            "payload": {"ledger_integrity": 0.91},
            "ledger_id": "ledger-alpha",
        }
    ]
    assert st.session_state.latest_structured_ledger == {
        "11": {"summary": long_summary},
        "19": {"summary": long_follow_up},
    }
    assert st.session_state.latest_structured_metrics == {"ledger_integrity": 0.9}


def test_persist_structured_views_skips_short_summaries(monkeypatch):
    captured: dict[str, object] = {}
    patch_calls: list[dict[str, object]] = []
    put_calls: list[dict[str, object]] = []
    post_calls: list[dict[str, object]] = []

    class DummyApiService:
        def put_ledger_s2(self, entity, payload, *, ledger_id=None):
            put_calls.append({"entity": entity, "payload": payload, "ledger_id": ledger_id})

        def patch_metrics(self, entity, payload, *, ledger_id=None):
            patch_calls.append({"entity": entity, "payload": payload, "ledger_id": ledger_id})

    def fake_write(_api_service, entity, structured, *, ledger_id=None):
        captured.update(entity=entity, structured=structured, ledger_id=ledger_id)
        return structured

    def fake_post(*args, **kwargs):
        post_calls.append({"args": args, "kwargs": kwargs})
        raise AssertionError("requests.post should not be called for short summaries")

    short_structured = {
        "slots": [{"prime": 2, "title": "Short"}],
        "s2": [
            {"prime": 11, "summary": "Too short for scoring"},
            {"prime": 19, "summary": "Another short summary"},
        ],
    }

    monkeypatch.setattr(chat_demo_app, "API_SERVICE", DummyApiService())
    monkeypatch.setattr(chat_demo_app, "write_structured_views", fake_write)
    monkeypatch.setattr(chat_demo_app.requests, "post", fake_post)

    result = chat_demo_app._persist_structured_views("demo", short_structured, ledger_id="ledger-beta")

    assert captured["structured"]["s2"] == []
    assert result == {
        "s2": {
            "11": {"summary": "Too short for scoring"},
            "19": {"summary": "Another short summary"},
        }
    }
    assert post_calls == []
    assert patch_calls == []
    assert put_calls == []


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
