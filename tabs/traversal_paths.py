"""Traversal paths tab renderer."""

from __future__ import annotations

from typing import Mapping, Sequence

import streamlit as st

from services.api_helpers import fetch_traversal_paths, get_ledger_id


def render_tab(session_state) -> None:
    entity = session_state.get("entity")
    if not entity:
        st.info("Select an entity to view traversal paths.")
        return

    payload = fetch_traversal_paths(
        session_state,
        entity=entity,
        ledger_id=get_ledger_id(session_state),
        limit=10,
    )
    if not payload.get("supported", True):
        st.info("Traversal endpoint unavailable on this backend.")
        return

    paths = payload.get("paths") if isinstance(payload.get("paths"), Sequence) else []
    if not paths:
        message = payload.get("message") if isinstance(payload.get("message"), str) else None
        st.info(message or "No traversal paths returned yet.")
        return

    for idx, path in enumerate(paths[:10], start=1):
        if not isinstance(path, Mapping):
            continue
        nodes = path.get("nodes") if isinstance(path.get("nodes"), Sequence) else []
        labels: list[str] = []
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            label = node.get("label") if isinstance(node.get("label"), str) else None
            if not label and isinstance(node.get("prime"), int):
                label = f"Prime {node['prime']}"
            if not label and isinstance(node.get("note"), str):
                label = node["note"]
            weight = node.get("weight") if isinstance(node.get("weight"), (int, float)) else None
            if weight is not None and label:
                labels.append(f"{label} ({weight:.2f})")
            elif label:
                labels.append(label)
        if not labels:
            labels.append("(no nodes)")
        score = path.get("score") if isinstance(path.get("score"), (int, float)) else None
        header = f"Path {idx}: {' → '.join(labels)}"
        if score is not None:
            header = f"{header} — score {score:.2f}"
        st.markdown(f"**{header}**")
        metadata = path.get("metadata") if isinstance(path.get("metadata"), Mapping) else {}
        if metadata:
            meta_rows = [
                f"{key}: {value}"
                for key, value in metadata.items()
                if isinstance(value, (str, int, float))
            ]
            if meta_rows:
                st.caption("; ".join(meta_rows[:6]))
        st.divider()
