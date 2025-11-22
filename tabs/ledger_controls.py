"""Sidebar ledger controls."""

from __future__ import annotations

import streamlit as st

from services.api_helpers import create_or_switch_ledger, get_ledger_id, refresh_ledgers


def render_tab(session_state) -> None:
    sidebar = st.sidebar
    sidebar.subheader("Ledger routing")

    if sidebar.button("Refresh ledgers", key="refresh_ledgers_btn"):
        _refresh_state(session_state)

    if session_state.get("ledger_refresh_error"):
        sidebar.warning(
            f"Ledger list unavailable: {session_state.ledger_refresh_error}"
        )

    available = [
        entry["ledger_id"]
        for entry in session_state.get("ledgers", [])
        if entry.get("ledger_id")
    ]
    current = get_ledger_id(session_state)
    if current and current not in available:
        available = [current] + [opt for opt in available if opt != current]

    if available:
        idx = available.index(current) if current in available else 0
        selection = sidebar.selectbox(
            "Active ledger",
            available,
            index=idx,
            help="All API calls include X-Ledger-ID so traffic routes to the selected remote ledger.",
        )
        if selection != current:
            session_state.ledger_id = selection
            st.toast(f"Routing to ledger '{selection}'", icon="🔀")
    else:
        sidebar.info("No ledgers yet. Create one below to start routing traffic.")

    with sidebar.form("create_ledger_form", clear_on_submit=True):
        new_ledger = st.text_input("Create or switch ledger", placeholder="team-alpha")
        submitted = st.form_submit_button("Create / Switch")
        if submitted:
            ledger_id = new_ledger.strip()
            if ledger_id:
                ok, error = create_or_switch_ledger(session_state, ledger_id)
                if ok:
                    st.toast(f"Ledger '{ledger_id}' ready", icon="📂")
                    _refresh_state(session_state, silent=True)
                elif error:
                    st.error("Unable to create or switch ledger; backend declined the request.")
                    st.caption(str(error))
            else:
                st.error("Ledger ID cannot be blank.")

    if session_state.get("ledgers"):
        sidebar.caption("Active ledger paths:")
        for entry in session_state["ledgers"]:
            ledger_id = entry.get("ledger_id")
            path = entry.get("path") or "—"
            sidebar.caption(f"• {ledger_id}: {path}")
    sidebar.caption(
        "To archive a ledger, stop routing to it and remove it via the admin API."
    )


def _refresh_state(session_state, *, silent: bool = False) -> None:
    ledgers, error = refresh_ledgers(session_state)
    if error:
        session_state.ledger_refresh_error = error
        if not silent:
            st.error("Ledger list unavailable; backend not exposing ledger management.")
            st.caption(str(error))
        return
    session_state.ledger_refresh_error = None
    session_state.ledgers = ledgers
    if not session_state.get("ledger_id") and ledgers:
        session_state.ledger_id = ledgers[0]["ledger_id"]
