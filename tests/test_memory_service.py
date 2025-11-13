import pytest

from services.memory_service import MOBIUS_REFRESH_INTERVAL, MemoryService


BERIGNY_EXCERPT = """Light for the million Religion of God and not of man\nFirst Lecture (Delivered before the Truth-seekers Free Debating society, on May 28, 1863, by Dr Berigny) God; Considered scientifically, morally and philosophically\nAll weak minds move with the atmosphere of public opinion, yet the discourse insists that knowledge alone restores religious harmony."""


@pytest.mark.parametrize("recall_prefix", ["Here’s", "Here's"])
def test_prepare_fallback_entries_filters_transcripts(recall_prefix: str) -> None:
    service = MemoryService(api_service=None, prime_weights={})
    raw_entries = [
        {
            "timestamp": 10,
            "text": f"{recall_prefix} what the ledger currently recalls: [2025-11-13 08:31] what do you recall about god? [2025-11-13 08:10] You: any quotes from God you might have from last hour?",
        },
        {
            "timestamp": 5,
            "text": BERIGNY_EXCERPT,
        },
    ]

    fallback = service._prepare_fallback_entries(raw_entries, limit=2)

    assert len(fallback) == 1
    assert "First Lecture" in fallback[0]["_sanitized_text"]


def test_prepare_fallback_entries_deduplicates_matching_memories() -> None:
    service = MemoryService(api_service=None, prime_weights={})
    raw_entries = [
        {"timestamp": 12, "text": BERIGNY_EXCERPT},
        {"timestamp": 11, "text": BERIGNY_EXCERPT},
        {"timestamp": 10, "text": BERIGNY_EXCERPT},
    ]

    fallback = service._prepare_fallback_entries(raw_entries, limit=5)

    assert len(fallback) == 1
    assert fallback[0]["timestamp"] == 12


class DummyAssemblyApi:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_assembly(self, *_, **__):
        self.calls += 1
        return {"summaries": [], "bodies": [], "claims": []}


def test_mobius_refresh_waits_for_rotation() -> None:
    api = DummyAssemblyApi()
    service = MemoryService(api_service=api, prime_weights={})

    triggered = service.maybe_refresh_mobius_alignment("demo", ledger_id="alpha")
    assert not triggered
    assert api.calls == 0


def test_mobius_refresh_triggers_after_interval() -> None:
    api = DummyAssemblyApi()
    service = MemoryService(api_service=api, prime_weights={})

    service.note_mobius_rotation("demo", ledger_id="alpha", timestamp=1.0)

    triggered = service.maybe_refresh_mobius_alignment(
        "demo", ledger_id="alpha", now=MOBIUS_REFRESH_INTERVAL + 1
    )

    assert triggered
    assert api.calls == 1
