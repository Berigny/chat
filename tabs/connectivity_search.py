"""Connectivity and search diagnostics tab."""

from __future__ import annotations

import json
import socket
from typing import Any, Callable, Mapping, Sequence

import streamlit as st

from services.api import requests


GetEntityFn = Callable[[], str | None]
PromotionCallback = Callable[[str, str | None, Mapping[str, Any]], Mapping[str, Any]]
BooleanCallback = Callable[[], None]
LedgerTracker = Callable[[str | None, str | None, Mapping[str, Any] | None], None]
AutoRecordLookup = Callable[[str | None, str | None], Mapping[str, Any] | None]
MetricsPatchCallback = Callable[[str, Mapping[str, Any], str | None], None]
LawfulnessPatchCallback = Callable[[str, int, str | None], None]


def render_tab(
    *,
    api_base: str,
    settings: Any,
    api_service: Any,
    default_entity: str,
    get_entity: GetEntityFn,
    clean_attachment_header: Callable[[str | None], str],
    apply_backdoor_promotion: PromotionCallback,
    promotion_result_ok: Callable[[Mapping[str, Any] | None], bool],
    ensure_slots_recall_mode: BooleanCallback,
    update_auto_promotion_tracker: LedgerTracker,
    get_auto_promotion_record: AutoRecordLookup,
    recommended_s2_metrics: Mapping[str, Any],
    safe_promotion_metrics: Mapping[str, Any],
    derive_flat_s2_map: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    s2_prime_keys: Sequence[str],
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

    debug_url = f"{host_url}/docs"
    st.write(f"Trying GET `{debug_url}`")
    headers: dict[str, str] = {}
    if settings.api_key:
        headers["x-api-key"] = settings.api_key
    try:
        response = requests.get(debug_url, headers=headers, timeout=10)
        st.write(f"Status: {response.status_code}")
        preview = response.text[:500]
        if len(response.text) > 500:
            preview += "…"
        st.code(preview or "<empty response>", language="text")
    except Exception as exc:  # pragma: no cover - network dependent
        st.error(f"HTTP error: {exc}")

    st.caption("Build the search index once after the first anchor to enable retrieval debugging.")
    if st.button("Build search index", key="build_search_index"):
        entity = get_entity() or default_entity
        post_headers = dict(headers)
        try:
            resp = requests.post(
                f"{api_base.rstrip('/')}/search/index",
                params={"entity": entity},
                headers=post_headers,
                timeout=30,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            st.error(f"Index build error: {exc}")
        else:
            st.write(f"HTTP status: {resp.status_code}")
            try:
                payload = resp.json()
            except Exception:
                payload = resp.text
            if isinstance(payload, (dict, list)):
                st.json(payload)
            else:
                st.code(str(payload) or "<empty response>", language="json")
            if resp.status_code == 404:
                st.info("/search/index is not enabled on this deployment yet.")
            st.toast(
                "Search index build triggered – only needed once after the first anchor.",
                icon="🔍",
            )

    entity = get_entity() or default_entity
    ledger_id = st.session_state.get("ledger_id")

    st.divider()
    st.write("### Search diagnostics")
    st.caption("Probe the recall/search endpoints without leaving the debug tab.")
    promotion_record = get_auto_promotion_record(entity, ledger_id)
    if promotion_record is None:
        st.caption("Auto-promotion attempts to raise lawfulness to tier 3 on load.")
    elif promotion_record.get("ok"):
        st.success("Default auto-promotion applied – S2 gates should be open.")
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
    mode_options = ["auto (engine default)", "recall", "slots", "s1", "body"]
    selected_mode = st.selectbox(
        "Search mode override",
        mode_options,
        index=0,
        key="search_probe_mode",
        help="Force a specific /search mode while debugging recall.",
    )
    mode_value = None if selected_mode.startswith("auto") else selected_mode
    probe_limit = st.number_input(
        "Result limit",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="search_probe_limit",
    )

    def _run_search_probe(query: str, mode_override: str | None = None) -> None:
        cleaned_query = clean_attachment_header(query)
        if not cleaned_query:
            st.warning("Enter a probe query first.")
            return
        try:
            payload = api_service.search(
                entity,
                cleaned_query,
                ledger_id=ledger_id,
                mode=mode_override if mode_override is not None else mode_value,
                limit=int(probe_limit),
            )
        except requests.RequestException as exc:
            st.error(f"Search call failed: {exc}")
        else:
            st.info("Search request succeeded.")
            st.session_state["search_probe_last_payload"] = payload
            st.session_state["search_probe_last_query"] = cleaned_query
            st.session_state["search_probe_last_mode"] = (
                mode_override if mode_override is not None else mode_value
            )
            response_text = payload.get("response") if isinstance(payload, Mapping) else None
            if response_text:
                st.caption("Response text")
                st.code(response_text)
            slots = payload.get("slots") if isinstance(payload, Mapping) else None
            if isinstance(slots, list) and slots:
                st.caption("Slots returned")
                st.json(slots)
            st.json(payload or {})

    if st.button("Probe /search endpoint", key="probe_search_endpoint"):
        _run_search_probe(probe_query)

    def _should_offer_body_retry(payload: Mapping[str, Any] | None) -> bool:
        if not isinstance(payload, Mapping):
            return False
        results = payload.get("results")
        if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
            if len(results) == 0:
                return True
        total = payload.get("total")
        if isinstance(total, (int, float)) and total == 0:
            return True
        return False

    last_payload = st.session_state.get("search_probe_last_payload")
    last_query = st.session_state.get("search_probe_last_query")
    if _should_offer_body_retry(last_payload):
        st.warning(
            "The last probe returned zero results. Body mode inspects the raw memory body "
            "so you can confirm whether a full-text hit exists even when snippet/slot "
            "indexes look empty."
        )
        st.caption(
            "Use this fallback when you expect stored text to match but structured search "
            "comes back blank. Body mode may surface the entry even if slots or recalls "
            "are missing."
        )
        if st.button("Retry in body mode", key="retry_search_body_mode"):
            _run_search_probe(last_query or probe_query, mode_override="body")

    st.divider()
    st.write("### 🧪 TEST ONLY – Entity promotion back-door")
    st.caption(
        "This bypasses real governance. Use it to unlock S2 search/writes "
        "while the scoring pipeline is being wired."
    )

    hdr = {"Content-Type": "application/json"}
    if settings.api_key:
        hdr["x-api-key"] = settings.api_key

    new_tier = st.slider("Lawfulness tier", 0, 3, 1, help="0=none, 3=S2 unlocked")
    if st.button("Apply tier", key="apply_lawfulness"):
        try:
            resp = requests.patch(
                f"{api_base.rstrip('/')}/ledger/lawfulness",
                params={"entity": entity, "ledger_id": ledger_id} if ledger_id else {"entity": entity},
                headers=hdr,
                json={"value": new_tier},
                timeout=10,
            )
            st.write(f"Lawfulness → {new_tier} : HTTP {resp.status_code}")
            if resp.status_code != 200:
                st.json(resp.json() if resp.content else resp.text)
        except Exception as exc:
            st.error(f"Lawfulness patch failed: {exc}")

    st.caption("Apply the recommended S2 metrics in one click or replay the back-door bundle.")
    if st.button("Promote to S2 tier", key="promote_s2"):
        try:
            api_service.patch_metrics(
                entity,
                dict(recommended_s2_metrics),
                ledger_id=ledger_id,
            )
            api_service.patch_lawfulness(
                entity,
                3,
                ledger_id=ledger_id,
            )
        except requests.RequestException as exc:
            st.error(f"S2 promotion failed: {exc}")
        else:
            ensure_slots_recall_mode()
            st.toast("S2 recall unlocked – slots search enabled.", icon="🚀")
            st.success("Promotion succeeded – recall mode updated.")

    safe_metrics = dict(safe_promotion_metrics)
    if st.button("🔓 Promote entity to S2 tier (test back-door)", key="promote_s2_backdoor"):
        try:
            result = apply_backdoor_promotion(
                entity,
                ledger_id,
                metrics_payload=safe_metrics,
            )
        except Exception as exc:
            st.error(f"Promotion failed: {exc}")
        else:
            success = promotion_result_ok(result)
            if success:
                st.success("Entity promoted – S2 writes & search unlocked.")
                ensure_slots_recall_mode()
            else:
                st.warning("Promotion attempted – inspect HTTP details below.")
            st.json(result)
            update_auto_promotion_tracker(entity, ledger_id, result=result)

    st.divider()
    st.write("### /traverse debug")
    start_node = st.number_input(
        "Start node",
        min_value=0,
        max_value=7,
        value=2,
        step=1,
        format="%d",
    )
    traverse_depth = st.number_input(
        "Traversal depth",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        format="%d",
    )

    if st.button("Debug /traverse", key="debug_traverse"):
        params = {
            "start": int(start_node),
            "depth": int(traverse_depth),
        }
        try:
            resp = requests.post(
                f"{api_base.rstrip('/')}/traverse",
                params=params,
                headers=headers,
                timeout=15,
            )
            st.write(f"Status: {resp.status_code}")
            st.code(resp.text[:2000] or "<empty response>", language="json")
        except Exception as exc:  # pragma: no cover - network dependent
            st.error(f"Traversal call error: {exc}")

    st.divider()
    st.write("### /ledger/s2 debug")
    debug_entity = get_entity() or default_entity
    debug_payload = {
        "11": {"summary": "Test summary"},
        "13": {"summary": "Another facet"},
    }
    st.caption("Send a simple S2 payload to inspect raw HTTP responses from the engine.")
    st.json(debug_payload)
    debug_headers: dict[str, str] = {}
    if settings.api_key:
        debug_headers["x-api-key"] = settings.api_key
    if ledger_id:
        debug_headers["X-Ledger-ID"] = ledger_id
    if st.button("Debug /ledger/s2", key="s2_endpoint_debug"):
        try:
            resp = requests.put(
                f"{api_base.rstrip('/')}/ledger/s2",
                params={"entity": debug_entity},
                json=debug_payload,
                headers=debug_headers,
                timeout=10,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            st.error(f"S2 call error: {exc}")
        else:
            st.write(f"HTTP status: {resp.status_code}")
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            st.json(detail if detail else {})

    st.divider()
    st.write("### /ledger/metrics debug")
    metrics_payload = {"ΔE": -1.0, "ΔDrift": -0.5, "ΔRetention": 0.8, "K": 0.0}
    st.caption("Replay the recommended metrics payload to surface validation errors.")
    st.json(metrics_payload)
    if st.button("Debug /ledger/metrics", key="metrics_endpoint_debug"):
        try:
            resp = requests.patch(
                f"{api_base.rstrip('/')}/ledger/metrics",
                params={"entity": debug_entity},
                json=metrics_payload,
                headers=debug_headers,
                timeout=10,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            st.error(f"Metrics call error: {exc}")
        else:
            st.write(f"HTTP status: {resp.status_code}")
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            st.json(detail if detail else {})

    st.divider()
    st.write("### /ledger/s2 payload debug")

    latest = st.session_state.get("latest_structured_ledger", {})
    s2_only = derive_flat_s2_map(latest)

    st.caption("What the UI would send *right now* (after coercion):")
    st.json(s2_only)

    hdr = {"Content-Type": "application/json"}
    if settings.api_key:
        hdr["x-api-key"] = settings.api_key

    if st.button("Copy as cURL", key="copy_s2_curl"):
        host = api_base.rstrip("/")
        entity_name = get_entity() or default_entity
        body = json.dumps(s2_only, sort_keys=True, separators=(",", ":"))
        hdr_str = " ".join(f'-H "{k}: {v}"' for k, v in hdr.items())
        curl = (
            f"curl -X PUT {hdr_str} -d '{body}' "
            f'"{host}/ledger/s2?entity={entity_name}"'
        )
        st.code(curl, language="bash")
        st.toast(
            "cURL copied to clipboard area – paste into Fly console to replay",
            icon="📋",
        )

    edited = st.text_area(
        "Edit payload (danger zone)",
        value=json.dumps(s2_only, indent=2),
        key="s2_live_edit",
        height=300,
    )
    if st.button("Send edited payload", key="send_s2_edit"):
        try:
            edited_map = json.loads(edited)
            assert isinstance(edited_map, dict)
            assert all(k in s2_prime_keys for k in edited_map)
        except Exception as exc:  # pragma: no cover - user input heavy
            st.error(f"Invalid shape: {exc}")
        else:
            pruned_map = derive_flat_s2_map(edited_map)
            resp = requests.put(
                f"{api_base.rstrip('/')}/ledger/s2?entity={get_entity() or default_entity}",
                headers=hdr,
                json=pruned_map,
                timeout=10,
            )
            st.write(f"HTTP status: {resp.status_code}")
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            st.json(detail)
