from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import requests
import streamlit as st

from composer import compose_summary
from intent_map import intent_primes, route_topic
from ledger_store import open_kv, query_shards
from prime_schema import DEFAULT_PRIME_SCHEMA, fetch_schema, schema_block
from prime_tagger import tag_modifiers
from services.api import ApiService
from services.prime_service import create_prime_service


API_URL = os.getenv("DUALSUBSTRATE_API", "https://dualsubstrate-commercial.fly.dev")
DEFAULT_ENTITY = os.getenv("DEFAULT_ENTITY", "demo_user")
LEDGER_PATH = os.getenv("LEDGER_PATH", "ledger_db")
DEFAULT_LEDGER_ID = os.getenv("DEFAULT_LEDGER_ID", "default")


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
    kv = open_kv(LEDGER_PATH)
    data: Dict[str, Dict] = {}
    for key, value in kv.items():
        try:
            data[key] = json.loads(value)
        except json.JSONDecodeError:
            continue
    kv.close()
    return data


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


def _write_tombstone(topic: str, primes: Iterable[int], summary: str) -> None:
    kv = open_kv(LEDGER_PATH)
    key = f"tombstone::{topic}::{datetime.utcnow().date().isoformat()}"
    record = {
        "topic": topic,
        "primes": list(primes),
        "summary": summary,
        "tombstone": True,
    }
    kv.put(key, json.dumps(record))
    kv.close()


def _build_lawful_augmentation_prompt(question: str, shards: List[Dict]) -> str:
    block = schema_block(st.session_state.get("prime_schema"))
    snippet_lines: List[str] = []
    for shard in shards:
        for snippet in shard.get("snippets", []):
            text = snippet.get("text", "")
            if text:
                snippet_lines.append(text)
    snippets = "\n".join(snippet_lines) or "(no snippets)"
    return (
        "Only quote snippets provided below. Do not invent memories.\n"
        "If no snippet matches, say so explicitly.\n"
        f"{block}\n"
        f"Question: {question}\n"
        f"Snippets:\n{snippets}"
    )


def _maybe_handle_recall_query(question: str) -> bool:
    topic = route_topic(question)
    required, preferred, modifiers = intent_primes(question)
    tag_modifiers(question, st.session_state.get("prime_schema"))
    shards = query_shards(topic, required, preferred, modifiers, limit=5)
    if not shards:
        message = f"No stored memories matched “{question}” yet."
        st.session_state.chat_history.append({"role": "assistant", "content": message})
        _write_tombstone(topic, required, message)
        return True

    reply = compose_summary(shards, question)
    if "\n\n" in reply:
        summary, footer = reply.rsplit("\n\n", 1)
    else:
        summary, footer = reply, ""
    st.session_state.chat_history.append(
        {"role": "assistant", "content": summary, "footer": footer, "shards": shards}
    )

    required_primes = list(required)
    safe_primes = sorted(set(required_primes + [5, 19]))
    safe_factors = [{"prime": prime, "delta": 1} for prime in safe_primes]
    _anchor(summary, record_chat=False, notify=False, factors_override=safe_factors)
    st.session_state.last_prompt = _build_lawful_augmentation_prompt(question, shards)
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

            shards = message.get("shards") or []
            for shard in shards:
                for snippet in shard.get("snippets", []):
                    text = snippet.get("text", "")
                    modifiers = snippet.get("modifiers") or []
                    if not text:
                        continue
                    if modifiers:
                        modifier_str = ", ".join(str(mod) for mod in modifiers)
                        st.caption(f"• {text} (modifiers: {modifier_str})")
                    else:
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
            help="All API calls include X-Ledger-ID so traffic routes to the chosen RocksDB.",
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
    st.sidebar.caption("To archive a ledger, stop routing to it and remove the directory from LEDGER_ROOT.")


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
