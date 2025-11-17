"""Tests for :mod:`services.api_helpers`."""

from __future__ import annotations

from types import SimpleNamespace

from services import api_helpers


def test_build_headers_includes_ledger() -> None:
    session_state = {
        api_helpers.ADMIN_CONFIG_KEY: {"api_key": "secret"},
        "ledger_id": "demo",
    }
    headers = api_helpers.build_headers(session_state)
    assert headers["x-api-key"] == "secret"
    assert headers["X-Ledger-ID"] == "demo"


def test_refresh_ledgers_uses_requester() -> None:
    payload = {"ledgers": [{"ledger_id": "alpha", "path": "/ledgers/alpha"}]}

    class DummyRequester:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict, int]] = []

        def get(self, url: str, *, headers: dict, timeout: int):
            self.calls.append((url, headers, timeout))
            return SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)

    requester = DummyRequester()
    session_state = {
        api_helpers.ADMIN_CONFIG_KEY: {"api_url": "https://api.example", "api_key": "k"},
        "ledger_id": "alpha",
    }
    ledgers, error = api_helpers.refresh_ledgers(session_state, requester=requester)
    assert error is None
    assert ledgers == [{"ledger_id": "alpha", "path": "/ledgers/alpha"}]
    assert requester.calls[0][0] == "https://api.example/admin/ledgers"


def test_create_or_switch_ledger_posts_payload() -> None:
    class DummyRequester:
        def __init__(self) -> None:
            self.payloads: list[dict] = []

        def post(self, url: str, *, json: dict, headers: dict, timeout: int):
            self.payloads.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            return SimpleNamespace(raise_for_status=lambda: None)

    requester = DummyRequester()
    session_state = {
        api_helpers.ADMIN_CONFIG_KEY: {"api_url": "https://api.example", "api_key": "k"},
        "ledger_id": "alpha",
    }
    ok, error = api_helpers.create_or_switch_ledger(
        session_state,
        "beta",
        requester=requester,
    )
    assert ok is True
    assert error is None
    posted = requester.payloads[0]
    assert posted["url"].endswith("/admin/ledgers")
    assert posted["json"] == {"ledger_id": "beta"}
    assert session_state["ledger_id"] == "beta"
