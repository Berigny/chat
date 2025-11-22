"""Ledger debug panel utilities."""
from __future__ import annotations

from typing import Any, Callable, Mapping

import streamlit as st

from services.api import requests


LedgerRefresher = Callable[[bool], None]


def _summarize_requests_error(exc: requests.HTTPError) -> str:
    """Return a concise message for HTTP failures."""

    if exc.response is None:
        return str(exc)
    status = getattr(exc.response, "status_code", None)
    reason = getattr(exc.response, "reason", None)
    detail = getattr(exc.response, "text", "") or ""
    summary = f"{status} {reason}".strip()
    if detail and detail not in summary:
        return f"{summary} – {detail.strip()}" if summary else detail.strip()
    return summary or str(exc)


def _extract_entry_count(metrics: Mapping[str, Any] | None) -> Any:
    """Best-effort extraction of an entry count from varied payloads."""

    if not isinstance(metrics, Mapping):
        return None
    for key in ("entry_count", "entries", "total_entries", "count", "total"):
        if key in metrics:
            return metrics.get(key)
    return None


def _render_search_results(results: list[Mapping[str, Any]]) -> None:
    st.markdown("**Top entries**")
    for entry in results[:10]:
        key = entry.get("key") if isinstance(entry, Mapping) else {}
        state = entry.get("state") if isinstance(entry, Mapping) else {}
        metadata = state.get("metadata") if isinstance(state, Mapping) else {}
        text = ""
        if isinstance(metadata, Mapping):
            text = metadata.get("text") or metadata.get("full_text") or ""
        snippet = (text[:240] + "…") if isinstance(text, str) and len(text) > 240 else text

        st.markdown(
            f"- **Key**: `{(key or {}).get('namespace')}` / `{(key or {}).get('identifier')}`  "
            f" | **phase**: `{(state or {}).get('phase')}`  "
            f" | **created_at**: `{entry.get('created_at') if isinstance(entry, Mapping) else ''}`\n\n"
            f"  _{snippet}_"
        )


def render_ledger_debug_panel(
    *,
    api_service: Any,
    ledger_management_enabled: bool,
    refresh_ledgers: LedgerRefresher,
) -> None:
    """Render ledger-level metrics and search debugging tools."""

    st.markdown("### Ledger debug")

    if not ledger_management_enabled:
        st.info("Ledger management is disabled for this deployment.")
        return

    if st.button("🔄 Refresh ledger list", key="ledger_debug_refresh"):
        refresh_ledgers(silent=True)

    ledgers = st.session_state.get("ledgers") or []
    if not ledgers:
        st.warning("No ledgers reported by backend. Try refreshing or check /admin/ledgers.")
        return

    overview_rows: list[dict[str, Any]] = []
    for ledger in ledgers:
        ledger_id = None
        if isinstance(ledger, Mapping):
            ledger_id = ledger.get("ledger_id") or ledger.get("id")
        ledger_id = ledger_id or str(ledger)

        entry_count = None
        metrics_error = None
        try:
            metrics = api_service.get_ledger_metrics(ledger_id)
            entry_count = _extract_entry_count(metrics)
        except requests.HTTPError as exc:  # type: ignore[name-defined]
            metrics_error = _summarize_requests_error(exc)
        except Exception as exc:  # pragma: no cover
            metrics_error = str(exc)

        overview_rows.append(
            {
                "ledger_id": ledger_id,
                "entry_count": entry_count,
                "metrics_error": metrics_error,
            }
        )

    st.write("**Ledger overview**")
    st.table(overview_rows)

    ledger_ids = [row["ledger_id"] for row in overview_rows if row.get("ledger_id")]
    current = st.session_state.get("ledger_id")
    default_index = ledger_ids.index(current) if current in ledger_ids else 0 if ledger_ids else 0

    selected_ledger = st.selectbox(
        "Inspect ledger",
        ledger_ids,
        index=default_index,
        key="ledger_debug_selected",
    )

    st.info(f"Current active ledger in session: `{current}`")

    if not selected_ledger:
        return

    st.markdown("#### Selected ledger metrics")
    try:
        metrics = api_service.get_ledger_metrics(selected_ledger)
        st.json(metrics)
    except Exception as exc:  # pragma: no cover
        st.error(f"Failed to load metrics for {selected_ledger}: {exc}")
        return

    st.markdown("#### Debug search")
    debug_query = st.text_input(
        "Search query for this ledger",
        value="",
        placeholder="e.g. God, eight equations, favourite colour",
        key="ledger_debug_query",
    )
    run_search = st.button("Run debug search", key="ledger_debug_run")

    if run_search:
        try:
            search_result = api_service.debug_search(selected_ledger, debug_query or "")
        except Exception as exc:  # pragma: no cover
            st.error(f"Search failed: {exc}")
            return

        st.write("Raw search response:")
        st.json(search_result)

        results = []
        if isinstance(search_result, Mapping):
            results = search_result.get("results") or search_result.get("entries") or []
        if not isinstance(results, list) or not results:
            st.warning("No entries returned for this query.")
            return

        _render_search_results([entry for entry in results if isinstance(entry, Mapping)])
