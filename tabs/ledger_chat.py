"""Ledger chat tab for the admin console."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import requests
import streamlit as st

from prime_schema import DEFAULT_PRIME_SCHEMA, schema_block
from services.api_helpers import (
    get_api_service,
    get_ledger_id,
    ingest_text,
    search_probe,
)
from ui_components import render_json_viewer, toggle_group

_SAFE_FACTORS = [{"prime": 5, "delta": 1}, {"prime": 19, "delta": 1}]


def _derive_flat_s2_map(structured: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    flat: dict[str, dict[str, str]] = {}
    if not isinstance(structured, Mapping):
        return flat
    for entry in structured.get("s2", []) or []:
        if not isinstance(entry, Mapping):
            continue
        summary = entry.get("summary")
        prime = entry.get("prime")
        if not isinstance(prime, int):
            continue
        if isinstance(summary, str) and summary.strip():
            flat[str(prime)] = {"summary": summary.strip()}
    raw_map = structured.get("raw") if isinstance(structured, Mapping) else None
    if isinstance(raw_map, Mapping):
        for key, value in raw_map.items():
            if not isinstance(key, str) or key in flat:
                continue
            summary_value = value.get("summary") if isinstance(value, Mapping) else None
            if isinstance(summary_value, str) and summary_value.strip():
                flat[key] = {"summary": summary_value.strip()}
    return flat


def render_tab(session_state) -> None:
    """Render the chat tab."""

    recall_mode = toggle_group(
        "Recall mode",
        ["s1", "s2"],
        key="recall_mode_selector",
        default=session_state.get("recall_mode", "s1"),
        help_text="Choose which ledger substrate powers the recall probe.",
    )
    session_state.recall_mode = recall_mode

    render_enrichment_report(session_state.get("last_enrichment_report"))
    render_history(session_state)

    prompt = st.chat_input("Ask the ledger")
    if prompt:
        submit_user_message(session_state, prompt.strip())


def anchor_message(
    session_state,
    text: str,
    *,
    record_chat: bool = True,
    notify: bool = True,
    factors_override: Iterable[Mapping[str, int]] | None = None,
) -> bool:
    entity = session_state.get("entity")
    if not entity:
        st.error("No entity configured.")
        return False

    schema = session_state.get("prime_schema") or DEFAULT_PRIME_SCHEMA
    ledger_id = get_ledger_id(session_state)
    overrides = _coerce_override_factors(factors_override)

    try:
        ingest_result = ingest_text(
            session_state,
            entity=entity,
            text=text,
            schema=schema,
            ledger_id=ledger_id,
            metadata={"source": "admin_app"},
            factors_override=overrides,
        )
    except requests.RequestException as exc:
        st.error(f"Anchor failed: {exc}")
        session_state.last_anchor_status = "error"
        session_state.last_anchor_payload = None
        if notify:
            st.toast("Anchor failed", icon="❌")
        return False

    if record_chat:
        session_state.chat_history.append({"role": "user", "content": text})
    session_state.last_anchor_status = "ok"
    flow_errors = ingest_result.get("flow_errors") if isinstance(ingest_result, Mapping) else None
    ledger_entry = ingest_result.get("ledger_entry") if isinstance(ingest_result, Mapping) else None
    session_state.last_anchor_payload = ledger_entry
    if flow_errors:
        st.error("Anchor blocked: " + "; ".join(flow_errors))
        session_state.last_anchor_status = "error"
        if notify:
            st.toast("Anchor blocked", icon="⚠️")
        return False

    structured = ingest_result.get("structured") if isinstance(ingest_result, Mapping) else None
    structured_payload = structured if isinstance(structured, Mapping) else {}
    if isinstance(ledger_entry, Mapping):
        entry_metadata = ledger_entry.get("state", {}).get("metadata", {})
        if isinstance(entry_metadata, Mapping):
            structured_candidate = entry_metadata.get("structured")
            if isinstance(structured_candidate, Mapping):
                structured_payload = structured_candidate
        entry_id = ledger_entry.get("entry_id") or ledger_entry.get("id")
        if entry_id:
            session_state.last_structured_entry_id = entry_id
    if structured_payload:
        session_state.latest_structured_ledger = _derive_flat_s2_map(structured_payload)
    if notify:
        st.toast("Anchored", icon="✅")
    return True


def submit_user_message(session_state, text: str) -> None:
    success = anchor_message(session_state, text, record_chat=False)
    entry = {"role": "user", "content": text}
    if not success:
        entry["failed"] = True
        st.toast("Anchor failed", icon="❌")
    session_state.chat_history.append(entry)
    maybe_handle_recall_query(session_state, text)


def maybe_handle_recall_query(session_state, question: str) -> bool:
    entity = session_state.get("entity")
    if not entity:
        return False
    ledger_id = get_ledger_id(session_state)
    mode = session_state.get("recall_mode") or "s1"
    try:
        payload = search_probe(
            session_state,
            entity=entity,
            question=question,
            ledger_id=ledger_id,
            mode=mode,
            limit=5,
        )
    except requests.RequestException as exc:
        st.error(f"Recall failed: {exc}")
        return False

    payload = payload if isinstance(payload, Mapping) else {}
    response_text = payload.get("response") if isinstance(payload, dict) else None
    if isinstance(response_text, str):
        response_text = response_text.strip()
    if not response_text:
        return False

    session_state.chat_history.append(
        {"role": "assistant", "content": response_text, "recall": payload}
    )
    anchor_message(
        session_state,
        response_text,
        record_chat=False,
        notify=False,
        factors_override=_SAFE_FACTORS,
    )
    session_state.last_prompt = _build_lawful_augmentation_prompt(session_state, question, payload)
    return True


def render_enrichment_report(report: Mapping[str, Any] | None) -> None:
    if not report:
        return
    enriched = report.get("enriched", 0)
    total = report.get("total", 0)
    st.markdown("### Enrichment ethics review")
    st.caption(f"Processed {enriched}/{total} memories during the latest run.")

    reports = report.get("reports") if isinstance(report, Mapping) else None
    if not isinstance(reports, Sequence) or not reports:
        st.info("No enrichment artefacts recorded yet.")
        return

    for idx, entry in enumerate(reports, start=1):
        if not isinstance(entry, Mapping):
            continue
        st.markdown(f"**Report {idx}**")
        render_json_viewer("Payload", entry)


def render_history(session_state) -> None:
    for message in session_state.get("chat_history", []):
        role = message.get("role", "assistant")
        with st.chat_message(role):
            prefix = "❗ " if message.get("failed") else ""
            st.write(f"{prefix}{message.get('content', '')}")
            footer = message.get("footer")
            if footer:
                st.caption(footer)

            recall = message.get("recall") or {}
            if recall:
                _render_recall_snippets(recall)

            recent_memories = message.get("recent_memories") or []
            if recent_memories:
                st.caption("Latest ledger entries:")
                for text in recent_memories:
                    st.caption(f"• {text}")


def _render_recall_snippets(payload: Mapping[str, Any]) -> None:
    s1_slots = payload.get("s1") or []
    if s1_slots:
        st.caption("Structured S1 slots:")
        for slot in s1_slots:
            title = slot.get("title") or "(untitled)"
            tags = slot.get("tags") or []
            if tags:
                tag_str = ", ".join(str(tag) for tag in tags[:3])
                st.caption(f"• {title} [tags: {tag_str}]")
            else:
                st.caption(f"• {title}")

    s2_slots = _normalize_slot_entries(payload.get("s2"))
    if s2_slots:
        st.caption("Structured S2 summaries:")
        for slot in s2_slots:
            summary = slot.get("summary")
            if summary:
                st.caption(f"• {summary}")

    body_entries = payload.get("bodies") or []
    if body_entries:
        st.caption("Ledger body excerpts:")
        for entry in body_entries[:5]:
            chunk = entry.get("body") if isinstance(entry, Mapping) else None
            if isinstance(chunk, str) and chunk.strip():
                st.caption(f"• {chunk.strip()}")


def _normalize_slot_entries(slots: object) -> List[Dict[str, object]]:
    if isinstance(slots, Mapping):
        return [value for value in slots.values() if isinstance(value, Mapping)]
    if isinstance(slots, Sequence):
        return [value for value in slots if isinstance(value, Mapping)]
    return []


def _render_slot_snippets(slots: object) -> List[str]:
    snippets: List[str] = []
    for slot in _normalize_slot_entries(slots):
        summary = slot.get("summary")
        if isinstance(summary, str) and summary.strip():
            snippets.append(summary.strip())
            continue
        body = slot.get("body") if isinstance(slot.get("body"), list) else None
        if body:
            for chunk in body:
                if isinstance(chunk, str) and chunk.strip():
                    snippets.append(chunk.strip())
    return snippets


def _build_lawful_augmentation_prompt(
    session_state,
    question: str,
    payload: Mapping[str, object],
) -> str:
    block = schema_block(session_state.get("prime_schema"))
    snippets: List[str] = []
    snippets.extend(_render_slot_snippets(payload.get("s2", [])))
    if not snippets:
        snippets.extend(_render_slot_snippets(payload.get("slots", [])))
    if not snippets:
        for entry in payload.get("bodies", []):
            chunk = entry.get("body") if isinstance(entry, Mapping) else None
            if isinstance(chunk, str) and chunk.strip():
                snippets.append(chunk.strip())
    normalized_snippets = [item.strip() for item in snippets if isinstance(item, str) and item.strip()]
    snippet_block = "\n".join(normalized_snippets) if normalized_snippets else "(no snippets)"
    return (
        "Only quote snippets provided below. Do not invent memories.\n"
        "If no snippet matches, say so explicitly.\n"
        f"{block}\n"
        f"Question: {question}\n"
        f"Snippets:\n{snippet_block}"
    )


def _coerce_override_factors(entries: Iterable[Mapping[str, int]] | None) -> List[Dict[str, int]]:
    factors: List[Dict[str, int]] = []
    if entries is None:
        return factors
    for entry in entries:
        prime = entry.get("prime")
        delta = entry.get("delta", 1)
        if prime is None:
            continue
        try:
            factors.append({"prime": int(prime), "delta": int(delta)})
        except (TypeError, ValueError):
            continue
    return factors
