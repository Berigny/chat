import requests
import streamlit as st
import pytest

import admin_app
from prime_schema import DEFAULT_PRIME_SCHEMA


class FailingPrimeService:
    def anchor(self, *_, **__):
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

    st.session_state["__prime_service__"] = FailingPrimeService()
    monkeypatch.setattr(st, "error", lambda msg: errors.append(msg))
    monkeypatch.setattr(st, "toast", lambda msg, icon=None: toasts.append((msg, icon)))

    ok = admin_app._anchor("hello world", record_chat=True)
    assert not ok
    assert errors
    assert st.session_state.chat_history == []
    assert any(icon == "❌" for _, icon in toasts)

