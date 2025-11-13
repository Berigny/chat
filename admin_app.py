from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import streamlit as st

from intent_map import intent_primes, route_topic
from prime_schema import DEFAULT_PRIME_SCHEMA, fetch_schema, schema_block
from prime_tagger import tag_modifiers
from services.api import ApiService
from services.prime_service import create_prime_service


API_URL = os.getenv("DUALSUBSTRATE_API", "https://dualsubstrate-commercial.fly.dev")
DEFAULT_ENTITY = os.getenv("DEFAULT_ENTITY", "demo_user")
DEFAULT_LEDGER_ID = os.getenv("DEFAULT_LEDGER_ID", "default")

S1_PRIMES = {2, 3, 5, 7}
S2_PRIMES = {11, 13, 17, 19}


def _api_key() -> str | None:
    secrets = getattr(st, "secrets", {}) or {}
    return secrets.get("DUALSUBSTRATE_API_KEY") or os.getenv("DUALSUBSTRATE_API_KEY")


def _ledger_id() -> str | None:
    if hasattr(st, "session_state"):
        ledger_id = st.session_state.get("ledger_id")
    else:
        ledger_id = None
    return ledger_id or DEFAULT_LEDGER_ID


def _api_service() -> ApiService:
    key = "__api_service__"
    service = st.session_state.get(key) if hasattr(st, "session_state") else None
    if service is None:
        service = ApiService(API_URL, _api_key())
        if hasattr(st, "session_state"):
            st.session_state[key] = service
    return service


def _prime_service() -> "PrimeService":
    key = "__prime_service__"
    service = st.session_state.get(key) if hasattr(st, "session_state") else None
    if service is None:
        service = create_prime_service(_api_service(), min(DEFAULT_PRIME_SCHEMA))
        if hasattr(st, "session_state"):
            st.session_state[key] = service
    return service


def _headers(*, include_ledger: bool = True) -> Dict[str, str]:
    key = _api_key()
    headers = {"x-api-key": key} if key else {}
    if include_ledger:
        ledger_id = _ledger_id()
        if ledger_id:
            headers["X-Ledger-ID"] = ledger_id
    return headers


def _coerce_ledger_records(payload) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    source = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("ledgers"), list):
            source = payload.get("ledgers")  # type: ignore[assignment]
        else:
            source = [
                {"ledger_id": key, "path": value}
                for key, value in payload.items()
                if isinstance(key, str)
            ]
    if not isinstance(source, list):
        return records
    for item in source:
        if isinstance(item, str):
            records.append({"ledger_id": item})
            continue
        if not isinstance(item, dict):
            continue
        ledger_id = item.get("ledger_id") or item.get("id") or item.get("name")
        if not ledger_id:
            continue
        records.append({"ledger_id": str(ledger_id), "path": item.get("path") or item.get("base_path")})
    return records


def _refresh_ledgers(*, silent: bool = False) -> None:
    try:
        resp = requests.get(
            f"{API_URL}/admin/ledgers",
            headers=_headers(include_ledger=False),
            timeout=5,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        st.session_state.ledger_refresh_error = str(exc)
        if not silent:
            st.error(f"Failed to load ledgers: {exc}")
        return
    except ValueError:
        payload = []

    st.session_state.ledger_refresh_error = None
    st.session_state.ledgers = _coerce_ledger_records(payload) or st.session_state.get("ledgers", [])
    if not st.session_state.get("ledger_id"):
        st.session_state.ledger_id = st.session_state.ledgers[0]["ledger_id"] if st.session_state.ledgers else DEFAULT_LEDGER_ID


def _create_or_switch_ledger(ledger_id: str) -> bool:
    if not ledger_id:
        st.error("Ledger ID cannot be blank.")
        return False
    payload = {"ledger_id": ledger_id}
    try:
        resp = requests.post(
            f"{API_URL}/admin/ledgers",
            json=payload,
            headers=_headers(include_ledger=False),
            timeout=5,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Failed to create or switch ledger: {exc}")
        return False

    st.session_state.ledger_id = ledger_id
    st.toast(f"Ledger '{ledger_id}' ready", icon="📂")
    return True


def _init_session() -> None:
    if "entity" not in st.session_state:
        st.session_state.entity = DEFAULT_ENTITY
    if "prime_schema" not in st.session_state:
        st.session_state.prime_schema = fetch_schema(st.session_state.entity)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "quote_safe" not in st.session_state:
        st.session_state.quote_safe = True
    if "last_anchor_status" not in st.session_state:
        st.session_state.last_anchor_status = None
    if "ledger_id" not in st.session_state:
        st.session_state.ledger_id = DEFAULT_LEDGER_ID
    if "ledgers" not in st.session_state:
        st.session_state.ledgers = []
    if "ledger_refresh_error" not in st.session_state:
        st.session_state.ledger_refresh_error = None


def _load_ledger() -> Dict[str, Dict]:
    entity = st.session_state.get("entity")
    if not entity:
        return {}
    try:
        payload = _api_service().fetch_ledger(entity, ledger_id=_ledger_id())
    except requests.RequestException:
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_override_factors(entries: Optional[Iterable], *, default_delta: int = 1) -> List[Dict[str, int]]:
    factors: List[Dict[str, int]] = []
    if entries is None:
        return factors

    for entry in entries:
        if isinstance(entry, dict):
            prime = entry.get("prime", entry.get("p"))
            delta = entry.get("delta", entry.get("d", default_delta))
        else:
            prime = entry
            delta = default_delta
        if prime is None:
            continue
        try:
            factors.append({"prime": int(prime), "delta": int(delta)})
        except (TypeError, ValueError):
            continue

    return factors


def _anchor(
    text: str,
    *,
    record_chat: bool = True,
    notify: bool = True,
    factors_override: Optional[Iterable] = None,
) -> bool:
    entity = st.session_state.get("entity")
    if not entity:
        st.error("No entity configured.")
        return False

    schema = st.session_state.get("prime_schema") or DEFAULT_PRIME_SCHEMA

    modifiers = tag_modifiers(text, schema)
    override = (
        _coerce_override_factors(factors_override)
        if factors_override is not None
        else None
    )

    prime_service = _prime_service()
    try:
        prime_service.anchor(
            entity,
            text,
            schema,
            ledger_id=_ledger_id(),
            factors_override=override,
            modifiers=modifiers or None,
        )
    except requests.RequestException as exc:
        st.error(f"Anchor failed: {exc}")
        st.session_state.last_anchor_status = "error"
        if notify:
            st.toast("Anchor failed", icon="❌")
        return False

    if record_chat:
        st.session_state.chat_history.append({"role": "user", "content": text})
    st.session_state.last_anchor_status = "ok"
    if notify:
        st.toast("Anchored", icon="✅")
    return True


def _reset_entity_factors() -> bool:
    entity = st.session_state.get("entity")
    if not entity:
        return False
    schema = st.session_state.get("prime_schema") or DEFAULT_PRIME_SCHEMA
    try:
        payload = _api_service().fetch_ledger(entity, ledger_id=_ledger_id())
    except requests.RequestException as exc:
        st.error(f"Could not fetch ledger: {exc}")
        return False

    factors = payload.get("factors") if isinstance(payload, dict) else []
    for entry in factors:
        prime = entry.get("prime")
        value = entry.get("value")
        if not isinstance(prime, int) or not value:
            continue
        delta = -abs(int(value))
        try:
            _prime_service().anchor(
                entity,
                f"[reset] prime {prime}",
                schema,
                ledger_id=_ledger_id(),
                factors_override=[{"prime": prime, "delta": delta}],
            )
        except requests.RequestException as exc:
            st.error(f"Reset failed for prime {prime}: {exc}")
            return False
    return True


def _run_enrichment(limit: int = 50, reset_first: bool = True) -> None:
    entity = st.session_state.get("entity")
    if not entity:
        return
    if reset_first:
        if not _reset_entity_factors():
            st.error("Reset failed; enrichment skipped.")
            return

    try:
        memories = _api_service().fetch_memories(
            entity,
            ledger_id=_ledger_id(),
            limit=max(1, min(limit, 100)),
        )
    except requests.RequestException as exc:
        st.error(f"Failed to load memories: {exc}")
        return

    schema = st.session_state.get("prime_schema") or DEFAULT_PRIME_SCHEMA
    enriched = 0
    failures: List[str] = []
    prime_service = _prime_service()
    for entry in memories:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        modifiers = tag_modifiers(text, schema)
        try:
            prime_service.anchor(
                entity,
                text,
                schema,
                ledger_id=_ledger_id(),
                modifiers=modifiers or None,
            )
        except requests.RequestException as exc:
            stamp = entry.get("timestamp")
            label = str(stamp) if stamp else "unknown"
            failures.append(f"{label}: {exc}")
            continue
        enriched += 1
    if failures:
        st.warning(
            "Enrichment completed with failures. Retry details: "
            + "; ".join(failures)
        )
    st.success(f"Enrichment finished: {enriched}/{len(memories)} entries.")


def _latest_user_transcript() -> Optional[str]:
    for message in reversed(st.session_state.chat_history):
        if message.get("role") == "user" and not message.get("failed"):
            return message.get("content")
    return None


def _normalize_slot(slot: Dict) -> Optional[Dict[str, object]]:
    if not isinstance(slot, dict):
        return None
    prime = slot.get("prime")
    if not isinstance(prime, int):
        return None
    title = slot.get("title") or slot.get("name") or slot.get("label")
    summary = slot.get("summary") or slot.get("synopsis") or slot.get("description")
    tags = slot.get("tags") or []
    body_source = slot.get("body") or slot.get("chunks") or slot.get("body_chunks")
    if isinstance(tags, (list, tuple)):
        tag_list = [tag for tag in tags if isinstance(tag, str)]
    else:
        tag_list = []
    if isinstance(body_source, (list, tuple)):
        bodies = [chunk for chunk in body_source if isinstance(chunk, str) and chunk.strip()]
    else:
        bodies = []
    score = slot.get("score")
    if isinstance(score, (int, float)):
        score_value = float(score)
    else:
        score_value = 0.0
    timestamp = slot.get("timestamp")
    if isinstance(timestamp, (int, float)):
        timestamp_value: Optional[int] = int(timestamp)
    else:
        timestamp_value = None
    normalized = {
        "prime": prime,
        "title": str(title).strip() if isinstance(title, str) else None,
        "summary": str(summary).strip() if isinstance(summary, str) else None,
        "tags": tag_list,
        "body": bodies,
        "score": score_value,
        "timestamp": timestamp_value,
        "raw": slot,
    }
    return normalized


def _normalize_query_payload(payload: Dict | None) -> Dict[str, object]:
    payload = payload or {}
    raw_slots = payload.get("slots") if isinstance(payload, dict) else None
    if not isinstance(raw_slots, list):
        raw_slots = []
    normalized_slots = [slot for slot in (_normalize_slot(item) for item in raw_slots) if slot]
    normalized_slots.sort(key=lambda item: (item.get("score", 0.0), item.get("timestamp") or 0), reverse=True)

    s1_slots: List[Dict[str, object]] = []
    s2_slots: List[Dict[str, object]] = []
    body_entries: List[Dict[str, object]] = []
    for slot in normalized_slots:
        prime = slot.get("prime")
        if isinstance(prime, int) and prime in S1_PRIMES and (slot.get("title") or slot.get("tags")):
            s1_slots.append({
                "prime": prime,
                "title": slot.get("title"),
                "tags": slot.get("tags"),
                "score": slot.get("score", 0.0),
            })
        if isinstance(prime, int) and prime in S2_PRIMES and (slot.get("summary") or slot.get("body")):
            s2_slots.append({
                "prime": prime,
                "summary": slot.get("summary"),
                "score": slot.get("score", 0.0),
            })
        if slot.get("body"):
            for idx, chunk in enumerate(slot["body"]):
                body_entries.append(
                    {
                        "prime": prime,
                        "body": chunk,
                        "metadata": {
                            "index": idx,
                            "title": slot.get("title"),
                            "summary": slot.get("summary"),
                            "tags": slot.get("tags"),
                            "score": slot.get("score", 0.0),
                        },
                    }
                )

    memories_raw = payload.get("memories") if isinstance(payload, dict) else None
    memories: List[Dict[str, object]] = []
    if isinstance(memories_raw, list):
        for entry in memories_raw:
            if isinstance(entry, dict):
                memories.append(entry)

    lawful_prompt = payload.get("lawful_prompt") if isinstance(payload, dict) else None
    if not isinstance(lawful_prompt, str):
        lawful_prompt = None

    return {
        "slots": normalized_slots,
        "s1": s1_slots,
        "s2": s2_slots,
        "bodies": body_entries,
        "memories": memories,
        "lawful_prompt": lawful_prompt,
    }


def _render_slot_snippets(slots: Sequence[Dict[str, object]]) -> List[str]:
    snippets: List[str] = []
    for slot in slots:
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


def _compose_recall_summary(question: str, payload: Dict[str, object]) -> Tuple[str, str, Dict[str, object]]:
    slots: Sequence[Dict[str, object]] = payload.get("slots", []) if isinstance(payload, dict) else []
    s2_slots: Sequence[Dict[str, object]] = payload.get("s2", []) if isinstance(payload, dict) else []
    bodies: Sequence[Dict[str, object]] = payload.get("bodies", []) if isinstance(payload, dict) else []

    summary_sources: List[str] = []
    for entry in s2_slots:
        summary = entry.get("summary")
        if isinstance(summary, str) and summary.strip():
            summary_sources.append(summary.strip())
    if not summary_sources:
        for slot in slots:
            snippet = slot.get("summary") or slot.get("title")
            if isinstance(snippet, str) and snippet.strip():
                summary_sources.append(snippet.strip())
    if not summary_sources:
        for entry in bodies:
            chunk = entry.get("body")
            if isinstance(chunk, str) and chunk.strip():
                summary_sources.append(chunk.strip())

    summary_text = " ".join(summary_sources).strip()
    if not summary_text:
        summary_text = "I couldn’t find enough grounded material to answer that yet."

    prime_set = sorted({slot.get("prime") for slot in slots if isinstance(slot.get("prime"), int)})
    metadata_parts: List[str] = []
    if prime_set:
        metadata_parts.append("primes " + ", ".join(str(prime) for prime in prime_set[:6]))
    slot_count = len(slots)
    if slot_count:
        metadata_parts.append(f"{slot_count} structured slot{'s' if slot_count != 1 else ''}")
    body_count = len(bodies)
    if body_count:
        metadata_parts.append(f"{body_count} body chunk{'s' if body_count != 1 else ''}")

    signature = hashlib.sha1(f"{question}|{summary_text}".encode("utf-8")).hexdigest()[:12]
    metadata_parts.append(f"sig {signature}")
    footer = "Sources: " + "; ".join(metadata_parts)

    return summary_text, footer, payload


def _build_lawful_augmentation_prompt(question: str, payload: Dict[str, object]) -> str:
    block = schema_block(st.session_state.get("prime_schema"))
    snippets: List[str] = []
    snippets.extend(_render_slot_snippets(payload.get("s2", [])))
    if not snippets:
        snippets.extend(_render_slot_snippets(payload.get("slots", [])))
    if not snippets:
        for entry in payload.get("bodies", []):
            chunk = entry.get("body") if isinstance(entry, dict) else None
            if isinstance(chunk, str) and chunk.strip():
                snippets.append(chunk.strip())
    normalized_snippets = []
    for item in snippets:
        if isinstance(item, str) and item.strip():
            normalized_snippets.append(item.strip())
    snippet_block = "\n".join(normalized_snippets) if normalized_snippets else "(no snippets)"
    return (
        "Only quote snippets provided below. Do not invent memories.\n"
        "If no snippet matches, say so explicitly.\n"
        f"{block}\n"
        f"Question: {question}\n"
        f"Snippets:\n{snippet_block}"
    )


def _maybe_handle_recall_query(question: str) -> bool:
    topic = route_topic(question)
    required, preferred, modifiers = intent_primes(question)
    tag_modifiers(question, st.session_state.get("prime_schema"))
    entity = st.session_state.get("entity")
    if not entity:
        return False
    try:
        query_payload = _api_service().query_ledger(
            entity,
            question,
            ledger_id=_ledger_id(),
            limit=5,
            topic=topic,
            required=list(required),
            preferred=list(preferred),
            modifiers=list(modifiers),
        )
    except requests.RequestException as exc:
        st.error(f"Recall failed: {exc}")
        return False

    normalized = _normalize_query_payload(query_payload if isinstance(query_payload, dict) else {})
    if not normalized.get("slots") and not normalized.get("memories"):
        message = f"No stored memories matched “{question}” yet."
        recent_snippets: List[str] = []
        try:
            recent = _api_service().fetch_memories(entity, ledger_id=_ledger_id(), limit=3)
        except requests.RequestException:
            recent = []
        for entry in recent:
            text = (entry.get("text") or "") if isinstance(entry, dict) else ""
            if isinstance(text, str):
                snippet = text.strip()
                if snippet:
                    recent_snippets.append(snippet[:240])
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": message,
                "recent_memories": recent_snippets,
            }
        )
        return True

    summary, footer, enriched = _compose_recall_summary(question, normalized)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": summary, "footer": footer, "recall": enriched}
    )

    required_primes = list(required)
    safe_primes = sorted(set(required_primes + [5, 19]))
    safe_factors = [{"prime": prime, "delta": 1} for prime in safe_primes]
    _anchor(summary, record_chat=False, notify=False, factors_override=safe_factors)
    st.session_state.last_prompt = _build_lawful_augmentation_prompt(question, enriched)
    return True


def _submit_user_message(text: str) -> None:
    success = _anchor(text, record_chat=False)
    entry = {"role": "user", "content": text}
    if not success:
        entry["failed"] = True
        st.toast("Anchor failed", icon="❌")
    st.session_state.chat_history.append(entry)
    _maybe_handle_recall_query(text)


def _render_history() -> None:
    for message in st.session_state.chat_history:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            prefix = "❗ " if message.get("failed") else ""
            st.write(f"{prefix}{message.get('content', '')}")
            footer = message.get("footer")
            if footer:
                st.caption(footer)

            recall = message.get("recall") or {}
            if recall:
                s1_slots = recall.get("s1") or []
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

                s2_slots = recall.get("s2") or []
                if s2_slots:
                    st.caption("Structured S2 summaries:")
                    for slot in s2_slots:
                        summary = slot.get("summary")
                        if summary:
                            st.caption(f"• {summary}")

                body_entries = recall.get("bodies") or []
                if body_entries:
                    st.caption("Ledger body excerpts:")
                    for entry in body_entries[:5]:
                        chunk = entry.get("body")
                        if isinstance(chunk, str) and chunk.strip():
                            st.caption(f"• {chunk.strip()}")

            recent_memories = message.get("recent_memories") or []
            if recent_memories:
                st.caption("Latest ledger entries:")
                for text in recent_memories:
                    st.caption(f"• {text}")


def _render_ledger_controls() -> None:
    st.sidebar.subheader("Ledger routing")
    if st.sidebar.button("Refresh ledgers", key="refresh_ledgers_btn"):
        _refresh_ledgers()
    if st.session_state.get("ledger_refresh_error"):
        st.sidebar.warning(f"Ledger list unavailable: {st.session_state.ledger_refresh_error}")

    available = [entry["ledger_id"] for entry in st.session_state.get("ledgers", []) if entry.get("ledger_id")]
    current = st.session_state.get("ledger_id") or DEFAULT_LEDGER_ID
    if current and current not in available:
        available = [current] + [opt for opt in available if opt != current]

    if available:
        idx = available.index(current) if current in available else 0
        selection = st.sidebar.selectbox(
            "Active ledger",
            available,
            index=idx,
            help="All API calls include X-Ledger-ID so traffic routes to the selected remote ledger.",
        )
        if selection != current:
            st.session_state.ledger_id = selection
            st.toast(f"Routing to ledger '{selection}'", icon="🔀")
    else:
        st.sidebar.info("No ledgers yet. Create one below to start routing traffic.")

    with st.sidebar.form("create_ledger_form", clear_on_submit=True):
        new_ledger = st.text_input("Create or switch ledger", placeholder="team-alpha")
        submitted = st.form_submit_button("Create / Switch")
        if submitted:
            ledger_id = new_ledger.strip()
            if ledger_id and _create_or_switch_ledger(ledger_id):
                _refresh_ledgers(silent=True)

    if st.session_state.get("ledgers"):
        st.sidebar.caption("Active ledger paths:")
        for entry in st.session_state["ledgers"]:
            ledger_id = entry.get("ledger_id")
            path = entry.get("path") or "—"
            st.sidebar.caption(f"• {ledger_id}: {path}")
    st.sidebar.caption("To archive a ledger, stop routing to it and remove it via the admin API.")


def main() -> None:
    st.set_page_config(page_title="Prime Ledger", layout="wide")
    _init_session()
    if not st.session_state.get("ledgers"):
        _refresh_ledgers(silent=True)

    st.sidebar.title("Controls")
    _render_ledger_controls()
    st.session_state.quote_safe = st.sidebar.toggle("Quote-safe", value=True)

    if st.sidebar.button("Test Anchor"):
        if _anchor("Hello world", record_chat=False, notify=False):
            st.toast("Test anchor succeeded", icon="✅")
        else:
            st.toast("Test anchor failed", icon="❌")

    ledger_payload = json.dumps(_load_ledger(), indent=2)
    st.sidebar.download_button(
        "Download ledger JSON",
        data=ledger_payload,
        file_name="ledger.json",
        mime="application/json",
    )

    st.title("Ledger Recall Assistant")
    _render_history()

    user_input = st.chat_input("Ask the ledger")
    if user_input:
        _submit_user_message(user_input.strip())


if __name__ == "__main__":
    main()
