"""Chat tab renderer."""

from __future__ import annotations

import html
from typing import Sequence

import streamlit as st


def render_tab(chat_history: Sequence[tuple[str, str]] | None, pending_attachments: Sequence[dict] | None) -> None:
    """Render the chat stream for the primary tab."""

    if pending_attachments:
        for attachment in pending_attachments:
            preview = (attachment.get("text") or "").strip()
            summary = preview[:200].replace("\n", " ")
            if len(preview) > 200:
                summary += "…"
            st.info(f"Attachment ready: {attachment.get('name', 'file')} – {summary}")

    history = list(chat_history or [])
    recent_history = list(reversed(history[-20:]))
    if recent_history:
        entries = [
            f"<div class='chat-entry'><strong>{html.escape(role)}:</strong> {html.escape(content)}</div>"
            for role, content in recent_history
        ]
        stream_html = "<hr>".join(entries)
    else:
        stream_html = "<div class='chat-entry'>No chat history yet.</div>"
    st.markdown(f"<div class='chat-stream'>{stream_html}</div>", unsafe_allow_html=True)
    st.markdown("<hr class='full-divider'>", unsafe_allow_html=True)
