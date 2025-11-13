import requests
import streamlit as st
import pytest

import admin_app
from prime_schema import DEFAULT_PRIME_SCHEMA


class DummyPrimeService:
    def build_factors(self, *_args, **_kwargs):
        return []


class FailingApiService:
    def ingest(self, *_args, **_kwargs):
        raise requests.HTTPError("boom")


@pytest.fixture(autouse=True)
def clear_state():
    st.session_state.clear()
    st.session_state.entity = "tester"
    st.session_state.prime_schema = DEFAULT_PRIME_SCHEMA
    st.session_state.chat_history = []
    st.session_state.quote_safe = True
    yield
    st.session_state.clear()


def test_anchor_failure(monkeypatch):
    errors: list[str] = []
    toasts: list[tuple[str, str | None]] = []

    st.session_state["__prime_service__"] = DummyPrimeService()
    monkeypatch.setattr(admin_app, "_api_service", lambda: FailingApiService())
    monkeypatch.setattr(st, "error", lambda msg: errors.append(msg))
    monkeypatch.setattr(st, "toast", lambda msg, icon=None: toasts.append((msg, icon)))

    ok = admin_app._anchor("hello world", record_chat=True)
    assert not ok
    assert errors
    assert st.session_state.chat_history == []
    assert any(icon == "❌" for _, icon in toasts)
