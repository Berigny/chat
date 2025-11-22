from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

import requests
import streamlit as st

from prime_schema import DEFAULT_PRIME_SCHEMA, fetch_schema
from services.api_service import EnrichmentHelper
from services.ethics_service import EthicsService
from services.migration_cli import run_ledger_migration
from services.structured_writer import write_structured_views
from services.api_helpers import (
    ADMIN_CONFIG_KEY,
    create_or_switch_ledger,
    fetch_inference_state,
    fetch_traversal_paths,
    get_api_service,
    get_ledger_id,
    get_memory_service,
    get_prime_service,
    ingest_text,
    load_ledger_snapshot,
    refresh_ledgers,
    reset_entity_factors,
    search_probe,
)
from tabs import inference_status, ledger_chat as ledger_chat_tab, ledger_controls, traversal_paths


API_URL = os.getenv("DUALSUBSTRATE_API", "https://dualsubstrate-commercial.fly.dev")
DEFAULT_ENTITY = os.getenv("DEFAULT_ENTITY", "Demo_dev")
DEFAULT_LEDGER_ID = os.getenv("DEFAULT_LEDGER_ID", "default")
ENABLE_ADVANCED_PROBES = os.getenv("ENABLE_ADVANCED_PROBES", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
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
    if not hasattr(st, "session_state"):
        return DEFAULT_LEDGER_ID
    return get_ledger_id(st.session_state, DEFAULT_LEDGER_ID)


def _sync_ledgers(*, silent: bool = False) -> None:
    ledgers, error = refresh_ledgers(st.session_state)
    if error:
        st.session_state.ledger_refresh_error = error
        if not silent:
            st.error(f"Failed to load ledgers: {error}")
        return
    st.session_state.ledger_refresh_error = None
    st.session_state.ledgers = ledgers
    if not st.session_state.get("ledger_id") and ledgers:
        st.session_state.ledger_id = ledgers[0]["ledger_id"]


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
    if "recall_mode" not in st.session_state:
        st.session_state.recall_mode = "s1"
    if ADMIN_CONFIG_KEY not in st.session_state:
        st.session_state[ADMIN_CONFIG_KEY] = {
            "api_url": API_URL,
            "api_key": _api_key(),
            "prime_weights": PRIME_WEIGHTS,
            "fallback_prime": min(DEFAULT_PRIME_SCHEMA),
        }


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


def _run_enrichment(limit: int = 50, reset_first: bool = True) -> dict | None:
    entity = st.session_state.get("entity")
    if not entity:
        st.warning("No active entity selected for enrichment.")
        return None

    ledger_id = _ledger_id()
    window = max(1, min(int(limit or 0), 200))

    if reset_first and not reset_entity_factors(
        st.session_state,
        entity=entity,
        schema=schema,
        ledger_id=ledger_id,
    ):
        st.error("Reset failed; enrichment skipped.")
        return None

    api_service = get_api_service(st.session_state)

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
    prime_service = get_prime_service(st.session_state)
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

    ledger_snapshot: Dict[str, Any] | None = load_ledger_snapshot(
        st.session_state,
        entity=entity,
        ledger_id=ledger_id,
    )

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


def main() -> None:
    st.set_page_config(page_title="Prime Ledger", layout="wide")
    _init_session()
    if not st.session_state.get("ledgers"):
        _sync_ledgers(silent=True)

    st.sidebar.title("Controls")
    ledger_controls.render_tab(st.session_state)
    st.session_state.quote_safe = st.sidebar.toggle("Quote-safe", value=True)

    if st.sidebar.button("Test Anchor"):
        if ledger_chat_tab.anchor_message(
            st.session_state,
            "Hello world",
            record_chat=False,
            notify=False,
        ):
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

    entity = st.session_state.get("entity")
    ledger_snapshot = (
        load_ledger_snapshot(
            st.session_state,
            entity=entity,
            ledger_id=get_ledger_id(st.session_state),
        )
        if entity
        else {}
    )
    st.sidebar.download_button(
        "Download ledger JSON",
        data=json.dumps(ledger_snapshot, indent=2),
        file_name="ledger.json",
        mime="application/json",
    )

    st.title("Ledger Recall Assistant")
    memory_service = get_memory_service(st.session_state)
    traversal_supported = ENABLE_ADVANCED_PROBES and memory_service.supports_traverse()
    inference_supported = ENABLE_ADVANCED_PROBES and memory_service.supports_inference_state()

    tab_labels = ["Chat"]
    if traversal_supported:
        tab_labels.append("Traversal Paths")
    if inference_supported:
        tab_labels.append("Inference Status")

    tabs = st.tabs(tab_labels)
    chat_tab = tabs[0]
    tab_offset = 1

    with chat_tab:
        ledger_chat_tab.render_tab(st.session_state)

    if traversal_supported:
        with tabs[tab_offset]:
            traversal_paths.render_tab(st.session_state)
        tab_offset += 1
    if inference_supported:
        with tabs[tab_offset]:
            inference_status.render_tab(st.session_state)


if __name__ == "__main__":
    main()
