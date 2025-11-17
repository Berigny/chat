"""Reusable Streamlit UI primitives."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import streamlit as st


def sanitize_json_payload(payload: object) -> Mapping[str, Any] | Sequence[Any] | list[Any]:
    """Return a Streamlit-friendly JSON payload."""

    if isinstance(payload, (Mapping, list, tuple)):
        return payload  # type: ignore[return-value]
    if payload is None:
        return {}
    if isinstance(payload, str):
        cleaned = payload.strip()
        if cleaned.startswith("{") or cleaned.startswith("["):
            try:
                import json

                return json.loads(cleaned)
            except Exception:
                return {"text": cleaned}
        return {"text": cleaned}
    return {"value": payload}


def prepare_metric_rows(metrics: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> list[tuple[str, str]]:
    """Normalize metric entries to ``(label, value)`` rows."""

    rows: list[tuple[str, str]] = []
    if isinstance(metrics, Mapping):
        items = metrics.items()
    else:
        items = metrics or []
    for label, value in items:
        if not label:
            continue
        if isinstance(value, (int, float)):
            rows.append((str(label), f"{value:.2f}"))
        else:
            rows.append((str(label), str(value)))
    return rows


def render_json_viewer(
    title: str,
    payload: object,
    *,
    expanded: bool = False,
    st_module=st,
) -> None:
    """Render a collapsible JSON viewer with consistent styling."""

    cleaned = sanitize_json_payload(payload)
    with st_module.expander(title, expanded=expanded):
        st_module.json(cleaned, expanded=expanded)


def render_metrics_card(
    title: str,
    metrics: Mapping[str, Any] | Iterable[tuple[str, Any]],
    *,
    st_module=st,
) -> None:
    """Render a titled metric group."""

    rows = prepare_metric_rows(metrics)
    if not rows:
        return
    st_module.markdown(f"#### {title}")
    columns = st_module.columns(len(rows))
    for column, (label, value) in zip(columns, rows):
        column.metric(label, value)


def render_code_block(
    title: str,
    content: str,
    *,
    language: str = "text",
    st_module=st,
) -> None:
    """Render a titled code block."""

    st_module.caption(title)
    st_module.code(content, language=language)


def toggle_group(
    label: str,
    options: Sequence[str],
    *,
    key: str,
    default: str | None = None,
    help_text: str | None = None,
    st_module=st,
) -> str:
    """Render a segmented toggle and return the selected option."""

    if default and default in options:
        index = options.index(default)
    else:
        index = 0
    return st_module.radio(
        label,
        options,
        index=index,
        help=help_text,
        key=key,
        horizontal=True,
    )
