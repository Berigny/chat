"""About tab renderer."""

from __future__ import annotations

import streamlit as st


def render_tab() -> None:
    """Render the static about content."""

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(
            """
            <div class="about-col about-col-left">
                <h2 class="about-heading" style="font-size: 01.2rem; font-weight: 400">DualSubstrate ledger demo</h2>
                <p class="about-text">To test this DualSubstrate ledger demo speak or type. Everything anchors to the prime-based ledger. Tip: type /q or “what did I say at 7 pm” and I’ll quote you word-for-word from the prime-ledger. Anything else = normal chat.</p>
                <hr>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_right:
        st.markdown(
            """
            <div class="about-col about-col-right">
                <h2 class="metrics-heading" style="font-size: 1.25rem; font-weight: 400">How it works</h2>
                <p class="metrics-paragraph">Every exchange lands in a discrete prime ledger while the continuous substrate keeps the chat UX feeling modern. Use the neighboring tabs to inspect traversal paths, inference status, search diagnostics, and ledger routing.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
