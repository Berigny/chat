import time
from types import SimpleNamespace

import streamlit as st

import chat_demo_app
from prime_schema import DEFAULT_PRIME_SCHEMA
from services.prompt_service import PromptService
from services.memory_service import MemoryService


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

    ledger_payload = {
        "slots": [
            {
                "prime": 2,
                "title": "Meeting with Alice",
                "tags": ["meeting", "alice"],
                "summary": "Met Alice to review the roadmap.",
                "body": ["We reviewed the upcoming roadmap items."],
                "score": 0.91,
                "timestamp": 1_700_000_000_000,
            },
            {
                "prime": 11,
                "title": "Tuesday 4pm",
                "summary": "Planning session scheduled for Tuesday at 4pm.",
                "body": ["Block calendar slots for the planning session."],
                "score": 0.73,
                "tags": ["calendar"],
                "timestamp": 1_700_000_100_000,
            },
        ]
    }

    body_calls: list = []

    def fake_put_body(entity, prime, body_text, ledger_id=None, metadata=None):
        body_calls.append((entity, prime, body_text, ledger_id, metadata))
        return {}

    monkeypatch.setattr(chat_demo_app.API_SERVICE, "put_ledger_body", fake_put_body)

    s1_calls: list = []
    s2_calls: list = []

    monkeypatch.setattr(
        chat_demo_app.API_SERVICE,
        "put_ledger_s1",
        lambda entity, payload, ledger_id=None: s1_calls.append((entity, payload, ledger_id)) or {},
    )
    monkeypatch.setattr(
        chat_demo_app.API_SERVICE,
        "put_ledger_s2",
        lambda entity, payload, ledger_id=None: s2_calls.append((entity, payload, ledger_id)) or {},
    )
    structured_updates: list = []
    monkeypatch.setattr(
        chat_demo_app.MEMORY_SERVICE,
        "update_structured_ledger",
        lambda entity, payload, ledger_id=None: structured_updates.append((entity, payload, ledger_id)),
    )
    monkeypatch.setattr(chat_demo_app, "_refresh_capabilities_block", lambda: "")

    minted_sequence = [29, 31]

    def fake_ingest(entity, text, schema, *, ledger_id=None, **kwargs):
        bodies = []
        s1_slots = []
        s2_slots = []
        slots_with_primes = []
        for idx, slot in enumerate(ledger_payload["slots"]):
            prime = slot["prime"]
            minted_prime = minted_sequence[idx]
            chunk = slot["body"][0]
            metadata = {"index": 0, "source_prime": prime}
            fake_put_body(entity, minted_prime, chunk, ledger_id=ledger_id, metadata=metadata)
            slots_with_primes.append({**slot, "body_prime": minted_prime})
            bodies.append({"prime": minted_prime, "body": chunk, "metadata": metadata, "key": f"{prime}:0"})
            if prime in {2, 3, 5, 7}:
                s1_slots.append(
                    {
                        "prime": prime,
                        "value": 1,
                        "title": slot.get("title"),
                        "tags": slot.get("tags"),
                        "body_prime": minted_prime,
                        "metadata": {"tier": "S1"},
                    }
                )
            elif prime in {11, 13, 17, 19}:
                s2_slots.append(
                    {
                        "prime": prime,
                        "summary": slot.get("summary"),
                        "body_prime": minted_prime,
                        "metadata": {"tier": "S2"},
                    }
                )
        return {
            "structured": {
                "slots": slots_with_primes,
                "s1": s1_slots,
                "s2": s2_slots,
                "bodies": bodies,
            },
            "payload": {
                "s1": s1_slots,
                "s2": s2_slots,
                "bodies": [{"prime": entry["prime"], "metadata": entry.get("metadata", {})} for entry in bodies],
            },
            "response": {},
        }

    monkeypatch.setattr(chat_demo_app.PRIME_SERVICE, "ingest", fake_ingest)

    result = chat_demo_app._anchor("Test entry", record_chat=False, notify=False)

    assert result is True
    assert s1_calls, "Expected S1 persistence call"
    assert s2_calls, "Expected S2 persistence call"
    assert body_calls, "Expected body persistence calls"

    s1_payload = s1_calls[0][1]
    assert s1_payload["slots"][0]["value"] == 1
    assert s1_payload["slots"][0]["body_prime"] == 29

    s2_payload = s2_calls[0][1]
    assert any(slot.get("body_prime") for slot in s2_payload["slots"])

    body_entry = body_calls[0]
    assert body_entry[2] == "We reviewed the upcoming roadmap items."
    assert body_entry[-1]["index"] == 0

    assert structured_updates and structured_updates[0][0] == "demo"
    assert st.session_state.latest_structured_ledger["slots"], "Structured ledger cache not stored"


def test_memory_service_uses_structured_slots():
    api = SimpleNamespace(
        fetch_memories=lambda *a, **k: [
            {
                "text": "You: fallback conversation",
                "timestamp": 1,
            }
        ]
    )
    service = MemoryService(api, {2: 1.0})
    service.update_structured_ledger(
        "demo",
        {
            "slots": [
                {
                    "prime": 2,
                    "title": "Meeting",
                    "summary": "Met with product team.",
                    "tags": ["meeting"],
                    "score": 0.9,
                    "body": ["Discussed roadmap."],
                    "timestamp": 1_700_000_000_000,
                },
                {
                    "prime": 11,
                    "summary": "Scheduled follow-up for Tuesday.",
                    "score": 0.8,
                    "timestamp": 1_700_000_100_000,
                },
            ]
        },
        ledger_id="alpha",
    )

    results = service.select_context(
        "demo",
        "What did we plan?",
        DEFAULT_PRIME_SCHEMA,
        ledger_id="alpha",
        limit=2,
    )

    assert len(results) == 2
    assert results[0]["_structured_text"] == "Met with product team."
    assert results[0]["prime"] == 2
    assert results[1]["prime"] == 11


def test_prompt_service_renders_structured_sections():
    structured_entries = [
        {
            "prime": 2,
            "title": "Meeting",
            "summary": "Met with Alice.",
            "tags": ("meeting", "alice"),
            "body": ["We reviewed the launch plan."],
            "timestamp": 1_700_000_000_000,
        },
        {
            "prime": 11,
            "title": "Tuesday",
            "summary": "Follow-up scheduled.",
            "tags": (),
            "body": ["Calendar invites sent."],
            "timestamp": 1_700_000_050_000,
        },
    ]

    class StubMemoryService:
        def select_context(self, *a, **k):
            return structured_entries

        def structured_context(self, *a, **k):
            return {}

        def render_context_block(self, *a, **k):
            return ""

    prompt_service = PromptService(memory_service=StubMemoryService())

    prompt = prompt_service.build_augmented_prompt(
        entity="demo",
        question="What did we discuss?",
        schema=DEFAULT_PRIME_SCHEMA,
        chat_history=[],
    )

    assert "--- Ledger S1 Slots ---" in prompt
    assert "--- Ledger S2 Summaries ---" in prompt
    assert "--- Ledger Bodies ---" in prompt
    assert "Met with Alice." in prompt
    assert "Calendar invites sent." in prompt
