from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests
import streamlit as st

from prime_schema import DEFAULT_PRIME_SCHEMA, fetch_schema, schema_block
from prime_tagger import tag_modifiers
from services.api import ApiService
from services.api_service import EnrichmentHelper
from services.ethics_service import EthicsService
from services.prime_service import PrimeService
from services.memory_service import MemoryService
from services.migration_cli import run_ledger_migration
from services.structured_writer import write_structured_views


API_URL = os.getenv("DUALSUBSTRATE_API", "https://dualsubstrate-commercial.fly.dev")
DEFAULT_ENTITY = os.getenv("DEFAULT_ENTITY", "demo_user")
DEFAULT_LEDGER_ID = os.getenv("DEFAULT_LEDGER_ID", "default")
PRIME_WEIGHTS = {
    2: 1.5,
    3: 1.5,
    5: 1.5,
    7: 1.5,
    11: 1.0,
    13: 1.0,
    17: 1.0,
    19: 1.0,
}

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


def _memory_service() -> MemoryService:
    key = "__memory_service__"
    service = st.session_state.get(key) if hasattr(st, "session_state") else None
    if service is None:
        service = MemoryService(_api_service(), PRIME_WEIGHTS)
        if hasattr(st, "session_state"):
            st.session_state[key] = service
    return service


def _prime_service() -> "PrimeService":
    key = "__prime_service__"
    service = st.session_state.get(key) if hasattr(st, "session_state") else None
    if service is None:
        service = PrimeService(_api_service(), min(DEFAULT_PRIME_SCHEMA))
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
    _memory_service().clear_entity_cache(ledger_id=ledger_id)
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
    if "last_anchor_payload" not in st.session_state:
        st.session_state.last_anchor_payload = None
    if "ledger_id" not in st.session_state:
        st.session_state.ledger_id = DEFAULT_LEDGER_ID
    if "ledgers" not in st.session_state:
        st.session_state.ledgers = []
    if "ledger_refresh_error" not in st.session_state:
        st.session_state.ledger_refresh_error = None
    if "last_enrichment_report" not in st.session_state:
        st.session_state.last_enrichment_report = None


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
        ingest_result = prime_service.ingest(
            entity,
            text,
            schema,
            ledger_id=_ledger_id(),
            factors_override=override,
            llm_extractor=None,
            metadata={"source": "admin_app"},
        )
    except requests.RequestException as exc:
        st.error(f"Anchor failed: {exc}")
        st.session_state.last_anchor_status = "error"
        st.session_state.last_anchor_payload = None
        if notify:
            st.toast("Anchor failed", icon="❌")
        return False

    if record_chat:
        st.session_state.chat_history.append({"role": "user", "content": text})
    st.session_state.last_anchor_status = "ok"
    flow_errors = (
        ingest_result.get("flow_errors")
        if isinstance(ingest_result, dict)
        else None
    )
    st.session_state.last_anchor_payload = (
        ingest_result.get("anchor") if isinstance(ingest_result, dict) else None
    )
    if flow_errors:
        message = "; ".join(flow_errors)
        st.error(f"Anchor blocked: {message}")
        st.session_state.last_anchor_status = "error"
        if notify:
            st.toast("Anchor blocked", icon="⚠️")
        return False
    structured = ingest_result.get("structured") if isinstance(ingest_result, dict) else {}
    if structured:
        persisted = write_structured_views(
            _api_service(),
            entity,
            structured,
            ledger_id=_ledger_id(),
        )
        st.session_state.latest_structured_ledger = persisted
    if notify:
        st.toast("Anchored", icon="✅")
    return True


def _backfill_body_primes() -> None:
    entity = st.session_state.get("entity")
    if not entity:
        st.warning("Select an entity first.")
        return

    ledger_id = _ledger_id()
    sidebar = st.sidebar
    with sidebar.spinner("Running ledger migration…"):
        result = run_ledger_migration(
            entity,
            ledger_id=ledger_id,
            extra_args=["--backfill-bodies"],
        )

    st.session_state.last_migration_result = result.asdict()
    if result.ok:
        sidebar.success(f"Backfill completed for {entity}.")
    else:
        sidebar.error(
            f"Backfill failed for {entity} (exit code {result.returncode})."
        )

    if result.stdout.strip():
        sidebar.text_area(
            "Migration stdout",
            result.stdout,
            height=200,
            key="__migration_stdout__",
        )
    if result.stderr.strip():
        sidebar.code(result.stderr, language="text")


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


def _run_enrichment(limit: int = 50, reset_first: bool = True) -> dict | None:
    entity = st.session_state.get("entity")
    if not entity:
        st.warning("No active entity selected for enrichment.")
        return None

    ledger_id = _ledger_id()
    window = max(1, min(int(limit or 0), 200))

    if reset_first and not _reset_entity_factors():
        st.error("Reset failed; enrichment skipped.")
        return None

    api_service = _api_service()

    try:
        memories = api_service.fetch_memories(
            entity,
            ledger_id=ledger_id,
            limit=window,
        )
    except requests.RequestException as exc:
        st.error(f"Failed to load memories: {exc}")
        return None

    if not isinstance(memories, list) or not memories:
        st.info("No memories available for enrichment.")
        summary = {"enriched": 0, "total": 0, "failures": [], "reports": []}
        st.session_state.last_enrichment_report = summary
        return summary

    schema = st.session_state.get("prime_schema") or DEFAULT_PRIME_SCHEMA
    prime_service = _prime_service()
    helper = EnrichmentHelper(api_service, prime_service)
    ethics_service = EthicsService(schema=schema)

    enrichment_supported = api_service.supports_enrich()
    if not enrichment_supported:
        st.warning(
            "Remote enrichment endpoint is unavailable; ledger bodies will be stored without structured updates."
        )

    summary: dict[str, Any] = {
        "enriched": 0,
        "total": len(memories),
        "failures": [],
        "reports": [],
        "enrichment_supported": enrichment_supported,
    }

    ledger_snapshot: Dict[str, Any] | None = _load_ledger()

    for entry in memories:
        if not isinstance(entry, dict):
            continue
        text = (entry.get("text") or "").strip()
        if not text:
            continue

        ref_prime = None
        for key in ("prime", "body_prime", "prime_ref"):
            value = entry.get(key)
            if isinstance(value, int):
                ref_prime = value
                break
        if ref_prime is None:
            factors = entry.get("factors") if isinstance(entry.get("factors"), list) else []
            for factor in factors:
                if isinstance(factor, dict) and isinstance(factor.get("prime"), int):
                    ref_prime = factor["prime"]
                    break
        if ref_prime is None:
            ref_prime = prime_service.fallback_prime

        try:
            factor_deltas = prime_service.build_factors(
                text,
                schema,
                factors_override=None,
                llm_extractor=None,
            )
            result = helper.submit(
                entity,
                ref_prime=ref_prime,
                deltas=factor_deltas,
                body_chunks=[text],
                metadata={"source": "enrichment"},
                ledger_id=ledger_id,
                schema=schema,
            )
        except requests.RequestException as exc:
            stamp = entry.get("timestamp")
            label = str(stamp) if stamp else "unknown"
            summary["failures"].append(f"{label}: {exc}")
            continue

        flow_errors = (
            result.get("flow_errors") if isinstance(result, dict) else None
        )
        if flow_errors:
            summary["failures"].append(
                f"ref {ref_prime}: {'; '.join(flow_errors)}"
            )
            continue

        if not result.get("enrichment_supported", True):
            summary["enrichment_supported"] = False
            message = (
                "Enrichment endpoint unavailable; stored bodies without remote enrichment."
            )
            if message not in summary["failures"]:
                summary["failures"].append(message)
            result["text"] = text
            summary["reports"].append(result)
            continue

        summary["enriched"] += 1
        response_payload = result.get("response") if isinstance(result, dict) else {}
        structured = response_payload.get("structured") if isinstance(response_payload, dict) else None
        if structured:
            try:
                write_structured_views(
                    api_service,
                    entity,
                    structured,
                    ledger_id=ledger_id,
                )
            except requests.RequestException as exc:
                summary["failures"].append(f"Structured persist failed: {exc}")

        try:
            ledger_snapshot = api_service.fetch_ledger(entity, ledger_id=ledger_id)
        except requests.RequestException:
            pass

        ethics = ethics_service.evaluate(
            ledger_snapshot if isinstance(ledger_snapshot, dict) else {},
            deltas=result.get("deltas"),
            minted_bodies=result.get("bodies"),
        )
        result["ethics"] = ethics.asdict()
        result["text"] = text
        summary["reports"].append(result)

    if summary["failures"]:
        st.warning(
            "Enrichment completed with warnings. Details: "
            + "; ".join(summary["failures"])
        )
    st.success(f"Enrichment finished: {summary['enriched']}/{summary['total']} entries.")

    st.session_state.last_enrichment_report = summary
    _load_ledger()
    return summary


def _render_enrichment_report(report: Dict[str, Any] | None) -> None:
    if not report:
        return

    enriched = report.get("enriched", 0)
    total = report.get("total", 0)
    st.markdown("### Enrichment ethics review")
    st.caption(f"Processed {enriched}/{total} memories during the latest run.")

    reports = report.get("reports") if isinstance(report, dict) else None
    if not isinstance(reports, list) or not reports:
        st.info("No enrichment artefacts recorded yet.")
        return

    for idx, entry in enumerate(reports, start=1):
        if not isinstance(entry, dict):
            continue
        ref_prime = entry.get("ref_prime")
        header = f"Memory {idx}"
        if isinstance(ref_prime, int):
            header += f" • ref prime {ref_prime}"
        st.markdown(f"**{header}**")

        ethics = entry.get("ethics") if isinstance(entry.get("ethics"), dict) else {}
        lawfulness = float(ethics.get("lawfulness", 0.0))
        evidence = float(ethics.get("evidence", 0.0))
        non_harm = float(ethics.get("non_harm", 0.0))
        coherence = float(ethics.get("coherence", 0.0))
        cols = st.columns(4)
        cols[0].metric("Lawfulness", f"{lawfulness:.2f}")
        cols[1].metric("Evidence", f"{evidence:.2f}")
        cols[2].metric("Non-harm", f"{non_harm:.2f}")
        cols[3].metric("Coherence", f"{coherence:.2f}")

        notes = ethics.get("notes") if isinstance(ethics.get("notes"), list) else []
        if notes:
            for note in notes[:4]:
                st.caption(f"• {note}")
        bodies = entry.get("bodies") if isinstance(entry.get("bodies"), list) else []
        if bodies:
            snippets = []
            for body_entry in bodies[:2]:
                text = body_entry.get("body") if isinstance(body_entry, dict) else None
                if isinstance(text, str) and text.strip():
                    snippets.append(text.strip()[:160])
            if snippets:
                st.caption("Minted evidence:")
                for snippet in snippets:
                    st.caption(f"› {snippet}")


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
    entity = st.session_state.get("entity")
    if not entity:
        return False
    try:
        payload = _api_service().search(
            entity,
            question,
            ledger_id=_ledger_id(),
            mode="recall",
            limit=5,
        )
    except requests.RequestException as exc:
        st.error(f"Recall failed: {exc}")
        return False

    payload = payload if isinstance(payload, dict) else {}
    response_text = payload.get("response") if isinstance(payload, dict) else None
    if isinstance(response_text, str):
        response_text = response_text.strip()
    if not response_text:
        return False

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response_text, "recall": payload}
    )

    safe_factors = [{"prime": prime, "delta": 1} for prime in (5, 19)]
    _anchor(response_text, record_chat=False, notify=False, factors_override=safe_factors)
    st.session_state.last_prompt = _build_lawful_augmentation_prompt(question, payload)
    return True


def _submit_user_message(text: str) -> None:
    success = _anchor(text, record_chat=False)
    entry = {"role": "user", "content": text}
    if not success:
        entry["failed"] = True
        st.toast("Anchor failed", icon="❌")
    st.session_state.chat_history.append(entry)
    _maybe_handle_recall_query(text)


def _render_traversal_panel(entity: str | None) -> None:
    if not entity:
        st.info("Select an entity to view traversal paths.")
        return
    payload = _memory_service().traversal_paths(
        entity,
        ledger_id=_ledger_id(),
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
            meta_rows = [f"{key}: {value}" for key, value in metadata.items() if isinstance(value, (str, int, float))]
            if meta_rows:
                st.caption("; ".join(meta_rows[:6]))
        st.divider()


def _render_inference_panel(entity: str | None) -> None:
    if not entity:
        st.info("Select an entity to view inference status.")
        return
    payload = _memory_service().fetch_inference_state(
        entity,
        ledger_id=_ledger_id(),
        include_history=True,
        limit=10,
    )
    if not payload.get("supported", True):
        st.info("Inference state endpoint unavailable on this backend.")
        return
    status = payload.get("status") if isinstance(payload.get("status"), str) else None
    if status:
        st.markdown(f"**State:** {status}")
    active = payload.get("active") if isinstance(payload.get("active"), Mapping) else None
    if active:
        st.markdown(f"**Active:** {_format_inference_row_admin(active)}")
    queue = payload.get("queue") if isinstance(payload.get("queue"), Sequence) else []
    if queue:
        st.subheader("Queue")
        for entry in queue[:10]:
            summary = _format_inference_row_admin(entry)
            if summary:
                st.caption(f"• {summary}")
    history = payload.get("history") if isinstance(payload.get("history"), Sequence) else []
    if history:
        st.subheader("Recent Completions")
        for entry in history[:10]:
            summary = _format_inference_row_admin(entry)
            if summary:
                st.caption(f"• {summary}")
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    if metrics:
        metric_rows = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_rows.append(f"{key}: {value:.2f}")
        if metric_rows:
            st.subheader("Metrics")
            for row in metric_rows[:10]:
                st.caption(row)
    message = payload.get("message") if isinstance(payload.get("message"), str) else None
    if message and not (queue or history or active):
        st.info(message)


def _format_inference_row_admin(entry: Mapping[str, Any]) -> str:
    if not isinstance(entry, Mapping):
        return ""
    label = entry.get("label") if isinstance(entry.get("label"), str) else None
    if not label and isinstance(entry.get("prime"), int):
        label = f"Prime {entry['prime']}"
    status = entry.get("status") if isinstance(entry.get("status"), str) else None
    score = entry.get("score") if isinstance(entry.get("score"), (int, float)) else None
    note = entry.get("note") if isinstance(entry.get("note"), str) else None
    timestamp = entry.get("timestamp") if isinstance(entry.get("timestamp"), (int, float)) else None
    ts_label = None
    if timestamp is not None:
        try:
            ts_label = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M")
        except (ValueError, OverflowError, OSError):
            ts_label = None
    parts = [part for part in (label, status, ts_label) if part]
    summary = " | ".join(parts) if parts else "(entry)"
    if score is not None:
        summary = f"{summary} — score {score:.2f}"
    if note:
        summary = f"{summary} — {note}"
    return summary


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

                s2_slots = _normalize_slot_entries(recall.get("s2"))
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

    if st.sidebar.button("Backfill body primes"):
        _backfill_body_primes()

    with st.sidebar.expander("Enrichment workflow", expanded=False):
        enrichment_limit = st.number_input(
            "Replay memories",
            min_value=1,
            max_value=200,
            value=25,
            step=1,
            key="enrichment_limit_input",
        )
        reset_before = st.checkbox(
            "Reset discrete primes first",
            value=True,
            key="enrichment_reset_checkbox",
        )
        if st.button("Run enrichment", key="enrichment_run_btn"):
            with st.spinner("Running enrichment cycle…"):
                report = _run_enrichment(limit=int(enrichment_limit), reset_first=reset_before)
            if report is not None:
                st.session_state.last_enrichment_report = report

    ledger_payload = json.dumps(_load_ledger(), indent=2)
    st.sidebar.download_button(
        "Download ledger JSON",
        data=ledger_payload,
        file_name="ledger.json",
        mime="application/json",
    )

    st.title("Ledger Recall Assistant")
    memory_service = _memory_service()
    traversal_supported = memory_service.supports_traverse()
    inference_supported = memory_service.supports_inference_state()

    tab_labels = ["Chat"]
    if traversal_supported:
        tab_labels.append("Traversal Paths")
    if inference_supported:
        tab_labels.append("Inference Status")

    tabs = st.tabs(tab_labels)
    tab_index = 0
    chat_tab = tabs[tab_index]
    tab_index += 1
    traversal_tab = tabs[tab_index] if traversal_supported else None
    if traversal_supported:
        tab_index += 1
    inference_tab = tabs[tab_index] if inference_supported else None

    with chat_tab:
        _render_enrichment_report(st.session_state.get("last_enrichment_report"))
        _render_history()

        user_input = st.chat_input("Ask the ledger")
        if user_input:
            _submit_user_message(user_input.strip())

    entity = st.session_state.get("entity")
    if traversal_supported and traversal_tab is not None:
        with traversal_tab:
            _render_traversal_panel(entity)
    if inference_supported and inference_tab is not None:
        with inference_tab:
            _render_inference_panel(entity)


if __name__ == "__main__":
    main()
