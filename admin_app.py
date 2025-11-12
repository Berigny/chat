from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import requests
import streamlit as st

from composer import compose_summary
from flow_safe import sequence as flow_sequence
from intent_map import intent_primes, route_topic
from ledger_store import open_kv, query_shards
from prime_schema import DEFAULT_PRIME_SCHEMA, fetch_schema, schema_block
from prime_tagger import tag_modifiers, tag_primes


API_URL = os.getenv("DUALSUBSTRATE_API", "https://dualsubstrate-commercial.fly.dev")
DEFAULT_ENTITY = os.getenv("DEFAULT_ENTITY", "demo_user")
LEDGER_PATH = os.getenv("LEDGER_PATH", "ledger_db")


def _headers() -> Dict[str, str]:
    secrets = getattr(st, "secrets", {}) or {}
    key = secrets.get("DUALSUBSTRATE_API_KEY") or os.getenv("DUALSUBSTRATE_API_KEY")
    return {"x-api-key": key} if key else {}


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


def _is_flow_safe_sequence(seq: Iterable) -> bool:
    items = list(seq or [])
    if not items or len(items) % 2:
        return False
    for idx in range(0, len(items), 2):
        first, second = items[idx], items[idx + 1]
        if not isinstance(first, dict) or not isinstance(second, dict):
            return False
        prime_a = first.get("prime", first.get("p"))
        prime_b = second.get("prime", second.get("p"))
        delta_a = first.get("delta", first.get("d", 1))
        delta_b = second.get("delta", second.get("d", 1))
        if prime_a != prime_b or int(delta_a) != int(delta_b):
            return False
    return True


def _flow_safe_sequence(entries: Optional[Iterable], *, default_delta: int = 1) -> List[Dict[str, int]]:
    safe: List[Dict[str, int]] = []
    if entries is None:
        return flow_sequence([min(DEFAULT_PRIME_SCHEMA)], delta=default_delta)

    for entry in entries:
        if isinstance(entry, dict):
            prime = entry.get("prime", entry.get("p"))
            delta = entry.get("delta", entry.get("d", default_delta))
        else:
            prime = entry
            delta = default_delta
        if prime is None:
            continue
        safe.extend(flow_sequence([int(prime)], delta=int(delta)))

    return safe or flow_sequence([min(DEFAULT_PRIME_SCHEMA)], delta=default_delta)


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

    if factors_override is None:
        primes = tag_primes(text, schema)
        safe_seq = flow_sequence(primes)
    else:
        override = list(factors_override)
        if override and isinstance(override[0], dict):
            if _is_flow_safe_sequence(override):
                safe_seq = [
                    {"prime": int(item.get("prime", item.get("p"))), "delta": int(item.get("delta", item.get("d", 1)))}
                    for item in override
                    if item.get("prime", item.get("p")) is not None
                ]
            else:
                safe_seq = _flow_safe_sequence(override)
        else:
            safe_seq = _flow_safe_sequence(override)

    payload = {"entity": entity, "text": text, "factors": safe_seq}
    if modifiers:
        payload["modifiers"] = modifiers
    try:
        resp = requests.post(
            f"{API_URL}/anchor",
            json=payload,
            headers=_headers(),
            timeout=5,
        )
        resp.raise_for_status()
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
    try:
        resp = requests.get(
            f"{API_URL}/ledger",
            params={"entity": entity},
            headers=_headers(),
            timeout=5,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Could not fetch ledger: {exc}")
        return False

    data = resp.json()
    factors = data.get("factors") or []
    for entry in factors:
        prime = entry.get("prime")
        value = entry.get("value")
        if not isinstance(prime, int) or not value:
            continue
        delta = -abs(int(value))
        seq = flow_sequence([prime], delta=delta)
        try:
            post = requests.post(
                f"{API_URL}/anchor",
                json={"entity": entity, "factors": seq},
                headers=_headers(),
                timeout=5,
            )
            post.raise_for_status()
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
        resp = requests.get(
            f"{API_URL}/memories",
            params={"entity": entity, "limit": max(1, min(limit, 100))},
            headers=_headers(),
            timeout=5,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Failed to load memories: {exc}")
        return

    try:
        memories = resp.json()
    except ValueError:
        memories = []

    schema = st.session_state.get("prime_schema") or DEFAULT_PRIME_SCHEMA
    enriched = 0
    failures: List[str] = []
    for entry in memories:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        primes = tag_primes(text, schema)
        modifiers = tag_modifiers(text, schema)
        safe_seq = flow_sequence(primes)
        payload = {"entity": entity, "text": text, "factors": safe_seq}
        if modifiers:
            payload["modifiers"] = modifiers
        try:
            post = requests.post(
                f"{API_URL}/anchor",
                json=payload,
                headers=_headers(),
                timeout=5,
            )
            post.raise_for_status()
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
    safe_seq = flow_sequence(safe_primes)
    _anchor(summary, record_chat=False, notify=False, factors_override=safe_seq)
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


def main() -> None:
    st.set_page_config(page_title="Prime Ledger", layout="wide")
    _init_session()

    st.sidebar.title("Controls")
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
