import re
from datetime import datetime, timedelta

from services.memory_resolver import MemoryResolver


def _extract_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]{3,}", (text or "").lower())


def _noop_strip(text: str) -> str:
    return (text or "").strip()


def test_resolver_prioritises_definition_sentences():
    resolver = MemoryResolver(_extract_tokens, _noop_strip)
    now = datetime.utcnow()
    entries = [
        {
            "timestamp": int(now.timestamp() * 1000),
            "text": "God is defined as the living memory that binds justice and mercy.",
            "meta": {"source": "ledger"},
        },
        {
            "timestamp": int((now - timedelta(hours=3)).timestamp() * 1000),
            "text": "We reviewed general progress on the project roadmap.",
        },
    ]

    history = [("You", "Let us keep the definition of God handy."), ("Bot", "Will do.")]

    resolution = resolver.resolve("What definitions of God exist?", history, entries)

    assert resolution.summary is not None
    assert "God is defined as the living memory" in resolution.summary
    assert "god" in resolution.matched_terms


def test_resolver_returns_context_when_keywords_fail():
    resolver = MemoryResolver(_extract_tokens, _noop_strip)
    now = datetime.utcnow()
    entries = [
        {
            "timestamp": int((now - timedelta(hours=1)).timestamp() * 1000),
            "text": "We outlined the deployment checklist for the investor demo.",
        },
        {
            "timestamp": int((now - timedelta(hours=2)).timestamp() * 1000),
            "text": "Clarified that memory retrieval will summarise definitions before quoting.",
        },
    ]

    history = [("You", "remind me of the deployment checklist"), ("Bot", "Sure")]  # filler question

    resolution = resolver.resolve("what did we talk about last few days", history, entries)

    assert resolution.summary is not None
    assert "deployment checklist" in resolution.summary.lower()
    assert "did" not in resolution.focus_terms
