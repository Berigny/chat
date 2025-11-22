"""Connectivity and search diagnostics tab."""

from __future__ import annotations

import json
import socket
import re
from typing import Any, Callable, Mapping, Sequence

import streamlit as st

from services.api import requests
from backend_client import BackendAPIClient, build_action_request


GetEntityFn = Callable[[], str | None]
PromotionCallback = Callable[[str, str | None, Mapping[str, Any]], Mapping[str, Any]]
ResetRecallModeCallback = Callable[[], None]
LedgerTracker = Callable[[str | None, str | None, Mapping[str, Any] | None], None]
AutoRecordLookup = Callable[[str | None, str | None], Mapping[str, Any] | None]

def _normalize_namespace(ledger_id: str | None) -> str:
    raw = (ledger_id or "default").strip().lower()
    slug = re.sub(r"[^a-z0-9_-]", "-", raw)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "default"


def _entity_slug(entity: str | None, default_entity: str) -> str:
    raw = (entity or default_entity or "entity").strip().lower()
    slug = re.sub(r"[^a-z0-9_-]", "-", raw)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "entity"


def _parse_json_field(label: str, raw: str, *, fallback: Any) -> tuple[Any, str | None]:
    if not raw:
        return fallback, None
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return fallback, f"{label}: {exc}"


def render_tab(
    *,
    api_base: str,
    settings: Any,
    backend_client: BackendAPIClient,
    default_entity: str,
    get_entity: GetEntityFn,
    clean_attachment_header: Callable[[str | None], str],
    apply_backdoor_promotion: PromotionCallback,
    promotion_result_ok: Callable[[Mapping[str, Any] | None], bool],
    reset_recall_mode: ResetRecallModeCallback | None = None,
    update_auto_promotion_tracker: LedgerTracker,
    get_auto_promotion_record: AutoRecordLookup,
) -> None:
    """Render connectivity, search diagnostics, and ledger payload tools."""

    st.subheader("Connectivity debug")
    host_url = api_base.rstrip("/")
    host_name = host_url.replace("https://", "").replace("http://", "").split("/")[0]
    st.write(f"Target base URL: `{host_url}`")
    try:
        resolved_ip = socket.gethostbyname(host_name)
        st.success(f"DNS resolved `{host_name}` → `{resolved_ip}`")
    except Exception as exc:  # pragma: no cover - platform specific
        st.error(f"DNS resolution failed for `{host_name}`: {exc}")

    headers: dict[str, str] = {}
    if settings.api_key:
        headers["x-api-key"] = settings.api_key

    health_url = f"{host_url}/health"
    st.write(f"Checking `{health_url}`")
    try:
        health_resp = requests.get(health_url, headers=headers, timeout=5)
        st.write(f"/health → HTTP {health_resp.status_code}")
        try:
            st.json(health_resp.json())
        except Exception:  # pragma: no cover - text fallback
            st.code(health_resp.text or "<empty response>", language="text")
    except Exception as exc:  # pragma: no cover - network dependent
        st.error(f"/health error: {exc}")

    debug_url = f"{host_url}/docs"
    st.write(f"Trying GET `{debug_url}`")
    try:
        response = requests.get(debug_url, headers=headers, timeout=10)
        st.write(f"/docs → HTTP {response.status_code}")
        preview = response.text[:500]
        if len(response.text) > 500:
            preview += "…"
        st.code(preview or "<empty response>", language="text")
    except Exception as exc:  # pragma: no cover - network dependent
        st.error(f"/docs error: {exc}")

    st.caption("Trigger `/admin/reindex` once per ledger to rebuild the token-prime index.")
    if st.button("Build search index", key="build_search_index"):
        entity = get_entity() or default_entity
        try:
            payload = backend_client.reindex_ledger(entity)
        except requests.RequestException as exc:  # pragma: no cover - network dependent
            st.error(f"Index build error: {exc}")
        else:
            st.toast("Reindex triggered – only needed once after the first anchor.", icon="🔍")
            st.json(payload or {"status": "ok"})

    entity = get_entity() or default_entity
    ledger_id = st.session_state.get("ledger_id")

    st.divider()
    st.write("### Search diagnostics")
    st.caption("Probe the token-prime `/search` endpoint without leaving the debug tab.")
    promotion_record = get_auto_promotion_record(entity, ledger_id)
    if promotion_record is None:
        st.caption("Auto-promotion now calls `/coherence/evaluate` and `/ethics/evaluate` on load.")
    elif promotion_record.get("ok"):
        st.success("Governance diagnostics succeeded – coherence and ethics are reachable.")
    else:
        error_text = promotion_record.get("error")
        if error_text:
            st.warning(f"Auto-promotion still failing: {error_text}")
        else:
            st.warning("Auto-promotion pending – inspect response details below.")
    if promotion_record and promotion_record.get("result"):
        with st.expander("Auto-promotion response", expanded=not promotion_record.get("ok")):
            st.json(promotion_record["result"])

    col_probe, col_use_latest = st.columns([3, 1])
    default_query = st.session_state.get(
        "search_probe_query",
        "do you have any quotes about God?",
    )
    with col_probe:
        probe_query_raw = st.text_input(
            "Probe query",
            value=default_query,
            key="search_probe_query",
        )
        probe_query = clean_attachment_header(probe_query_raw)
    with col_use_latest:
        st.metric("Ledger", st.session_state.get("ledger_id") or "default")

    latest_preview = st.session_state.get("search_probe_latest_preview")
    if latest_preview:
        st.caption("Latest anchor snippet (verbatim)")
        st.code(latest_preview, language="text")

    probe_limit = st.number_input(
        "Result limit",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        key="search_probe_limit",
    )

    def _run_search_probe(query: str) -> None:
        cleaned_query = clean_attachment_header(query)
        if not cleaned_query:
            st.warning("Enter a probe query first.")
            return
        try:
            payload = backend_client.search_memories(
                entity=entity,
                query=cleaned_query,
                limit=int(probe_limit),
            )
        except requests.RequestException as exc:
            st.error(f"Search call failed: {exc}")
            return

        st.info("Search request succeeded.")
        st.session_state["search_probe_last_payload"] = payload if isinstance(payload, Mapping) else {}
        st.session_state["search_probe_last_query"] = cleaned_query
        st.session_state["search_probe_last_response"] = {
            "status": 200,
            "reason": "OK",
            "elapsed": None,
            "headers": {},
            "url": f"{host_url}/search",
        }
        results = payload.get("results") if isinstance(payload, Mapping) else None
        if isinstance(results, Sequence):
            st.caption(f"{len(results)} result(s)")
            for idx, row in enumerate(results, start=1):
                if not isinstance(row, Mapping):
                    continue
                entry_id = row.get("entry_id") or f"result-{idx}"
                score = row.get("score")
                st.write(f"**{entry_id}** — score {score if score is not None else 'n/a'}")
                snippet = row.get("snippet")
                if isinstance(snippet, str) and snippet.strip():
                    st.code(snippet.strip(), language="text")
                entry_payload = row.get("entry")
                if isinstance(entry_payload, Mapping):
                    with st.expander(f"Entry payload: {entry_id}", expanded=False):
                        st.json(entry_payload)
            if not results:
                st.info("no memories yet")
        else:
            st.info("no memories yet")

    if st.button("Probe /search endpoint", key="probe_search_endpoint"):
        _run_search_probe(probe_query)

    last_response_meta = st.session_state.get("search_probe_last_response")
    if isinstance(last_response_meta, Mapping):
        with st.expander("Last /search HTTP details", expanded=False):
            status = last_response_meta.get("status")
            reason = last_response_meta.get("reason")
            elapsed = last_response_meta.get("elapsed")
            url = last_response_meta.get("url")
            if status:
                st.write(f"HTTP {status} ({reason or 'n/a'})")
            if getattr(elapsed, "total_seconds", None):
                st.caption(f"Elapsed: {elapsed.total_seconds():.3f}s")
            if url:
                st.caption(f"URL: {url}")
            headers_meta = last_response_meta.get("headers")
            if isinstance(headers_meta, Mapping):
                st.json(dict(headers_meta))

    st.divider()
    st.write("### 🧪 Governance diagnostics")
    st.caption(
        "Send an ActionRequest payload to `/coherence/evaluate` and `/ethics/evaluate`. "
        "This mirrors the automatic back-door promotion logic."
    )
    payload_preview = build_action_request(
        actor=entity or default_entity,
        action="auto-promote",
        key_namespace=_normalize_namespace(ledger_id),
        key_identifier=f"{_entity_slug(entity, default_entity)}-structured",
        parameters={"tier": 3.0},
    )
    st.json(payload_preview)
    if st.button("Run governance diagnostics", key="governance_diag_btn"):
        try:
            result = apply_backdoor_promotion(entity, ledger_id)
        except Exception as exc:  # pragma: no cover - network dependent
            st.error(f"Promotion failed: {exc}")
        else:
            success = promotion_result_ok(result)
            if success:
                st.success("Coherence and ethics checks passed.")
                if reset_recall_mode:
                    reset_recall_mode()
            else:
                st.warning("Diagnostics returned warnings – inspect the payload below.")
            st.json(result)
            update_auto_promotion_tracker(entity, ledger_id, result=result)

    st.divider()
    st.write("### Ledger entry writer (`/ledger/write`)")
    namespace_default = _normalize_namespace(ledger_id)
    identifier_default = f"{_entity_slug(entity, default_entity)}-structured"
    writer_ns = st.text_input("Namespace", value=namespace_default, key="ledger_writer_ns")
    writer_identifier = st.text_input(
        "Identifier",
        value=identifier_default,
        key="ledger_writer_identifier",
    )
    phase_value = st.text_input("Phase label", value="structured-ledger", key="ledger_writer_phase")
    coordinates_raw = st.text_area(
        "Coordinates JSON",
        value=json.dumps({"signal": 1.0}, indent=2),
        key="ledger_writer_coordinates",
        height=120,
    )
    metadata_raw = st.text_area(
        "Metadata JSON",
        value=json.dumps(
            {
                "entity": entity,
                "ledger_id": ledger_id,
                "notes": "connectivity writer demo",
            },
            indent=2,
        ),
        key="ledger_writer_metadata",
        height=180,
    )
    if st.button("Write ledger entry", key="ledger_write_btn"):
        coords, coords_error = _parse_json_field("coordinates", coordinates_raw, fallback={})
        metadata, metadata_error = _parse_json_field("metadata", metadata_raw, fallback={})
        for error in (coords_error, metadata_error):
            if error:
                st.error(error)
                return
        entry = {
            "key": {"namespace": writer_ns or namespace_default, "identifier": writer_identifier or identifier_default},
            "state": {
                "coordinates": coords,
                "phase": phase_value or "structured-ledger",
                "metadata": metadata,
            },
        }
        write_headers = dict(headers)
        write_headers.setdefault("Content-Type", "application/json")
        try:
            resp = requests.post(
                f"{host_url}/ledger/write",
                headers=write_headers,
                json=entry,
                timeout=10,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            st.error(f"Ledger write failed: {exc}")
        else:
            st.write(f"HTTP status: {resp.status_code}")
            try:
                st.json(resp.json())
            except Exception:
                st.code(resp.text or "<empty response>", language="json")

    st.divider()
    st.write("### Ledger entry reader (`/ledger/read/{namespace:identifier}`)")
    entry_id_default = f"{writer_ns or namespace_default}:{writer_identifier or identifier_default}"
    entry_id = st.text_input(
        "Ledger entry path",
        value=entry_id_default,
        key="ledger_reader_entry_id",
    )
    if st.button("Read ledger entry", key="ledger_read_btn"):
        target = (entry_id or entry_id_default).strip()
        if not target or ":" not in target:
            st.error("Entry path must be in the form `namespace:identifier`.")
        else:
            try:
                resp = requests.get(
                    f"{host_url}/ledger/read/{target}",
                    headers=headers,
                    timeout=10,
                )
            except Exception as exc:  # pragma: no cover - network dependent
                st.error(f"Ledger read failed: {exc}")
            else:
                st.write(f"HTTP status: {resp.status_code}")
                try:
                    st.json(resp.json())
                except Exception:
                    st.code(resp.text or "<empty response>", language="json")
