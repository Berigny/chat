"""Helpers for choosing between OpenAI and Gemini providers in the UI."""

from __future__ import annotations

from typing import List

import streamlit as st

OPENAI_PROVIDER = "OpenAI (GPT-3.5)"
GENAI_PROVIDER = "Gemini Flash"


def _provider_options(openai_ready: bool, gemini_ready: bool) -> List[str]:
    options: List[str] = []
    if gemini_ready:
        options.append(GENAI_PROVIDER)
    if openai_ready:
        options.append(OPENAI_PROVIDER)
    if not options:
        options = [GENAI_PROVIDER]
    return options


def init_llm_provider(*, openai_ready: bool, gemini_ready: bool) -> None:
    """Ensure the session tracks which provider is active."""

    if "llm_provider" in st.session_state:
        return
    options = _provider_options(openai_ready, gemini_ready)
    st.session_state.llm_provider = options[0]


def render_llm_selector(*, openai_ready: bool, gemini_ready: bool) -> None:
    """Render the sidebar control for choosing the chat LLM."""

    options = _provider_options(openai_ready, gemini_ready)
    current = st.session_state.get("llm_provider")
    if current not in options:
        current = options[0]
    selection = st.sidebar.selectbox(
        "Response model",
        options,
        index=options.index(current),
        help="Switch between Gemini and OpenAI for chat replies.",
    )
    if selection != current:
        st.session_state.llm_provider = selection
        st.toast(f"LLM switched to {selection}", icon="🤖")


def use_openai_provider() -> bool:
    """Return True when the session is configured to use OpenAI."""

    return st.session_state.get("llm_provider") == OPENAI_PROVIDER


__all__ = [
    "GENAI_PROVIDER",
    "OPENAI_PROVIDER",
    "init_llm_provider",
    "render_llm_selector",
    "use_openai_provider",
]
