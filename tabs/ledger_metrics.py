"""Ledger routing and metrics tab."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import streamlit as st

from services.api import requests
from tabs.metrics_editor import render_entity_metrics_panel


RefreshLedgers = Callable[[bool], None]
LedgerSwitcher = Callable[[str, bool], bool]
ValidateLedger = Callable[[str], tuple[bool, str | None]]
SimpleCallback = Callable[[], None]
RenderLedgerState = Callable[[Mapping[str, Any]], None]
ExecuteEnrichment = Callable[[str, int], Mapping[str, Any]]
RenderEnrichmentPanel = Callable[[Mapping[str, Any] | None], None]


def _render_ledger_selector(
    *,
    add_option_label: str,
    refresh_ledgers: RefreshLedgers,
    create_or_switch_ledger: LedgerSwitcher,
    validate_ledger_name: ValidateLedger,
) -> None:
    st.subheader("Ledger routing")
    if st.button("Refresh ledgers", key="refresh_ledgers_btn"):
        refresh_ledgers(silent=False)
    raw_ledgers = st.session_state.get("ledgers", [])
    ledger_options: list[str] = []
    for entry in raw_ledgers:
        lid = entry.get("ledger_id")
        if lid and lid not in ledger_options:
            ledger_options.append(lid)
    active_ledger = st.session_state.get("ledger_id")
    if active_ledger and active_ledger not in ledger_options:
        ledger_options.insert(0, active_ledger)
    available_options = list(ledger_options)
    available_options.append(add_option_label)
    initial_index = available_options.index(active_ledger) if active_ledger in available_options else 0
    selection = st.selectbox(
        "Active ledger",
        available_options,
        index=initial_index,
        help="All API calls send X-Ledger-ID so memories stay scoped per tenant.",
    )
    if selection == add_option_label:
        st.caption("Rules: 3-32 chars, lowercase letters/digits, hyphens allowed in the middle.")
        new_ledger = st.text_input("New ledger ID", placeholder="team-alpha", key="new_ledger_id")
        if st.button("Create ledger", key="create_ledger_btn"):
            valid, error = validate_ledger_name(new_ledger)
            if not valid:
                st.error(error)
            elif create_or_switch_ledger(new_ledger):
                refresh_ledgers(silent=True)
    elif selection and selection != active_ledger:
        if create_or_switch_ledger(selection):
            refresh_ledgers(silent=True)

    if st.session_state.get("ledgers"):
        st.caption("Ledger directories:")
        for entry in st.session_state["ledgers"]:
            ledger_id = entry.get("ledger_id")
            path = entry.get("path") or "—"
            st.caption(f"• {ledger_id}: {path}")
    else:
        st.info("No ledgers detected yet — choose “Add new ledger…” to create one.")


def render_tab(
    *,
    tokens_saved: str,
    ledger_integrity: float,
    durability_hours: float,
    add_ledger_option: str,
    refresh_ledgers: RefreshLedgers,
    create_or_switch_ledger: LedgerSwitcher,
    validate_ledger_name: ValidateLedger,
    load_ledger: SimpleCallback,
    render_ledger_state: RenderLedgerState,
    get_entity: Callable[[], str | None],
    memory_service: Any,
    perform_lattice_rotation_fn: Callable[..., Mapping[str, Any]],
    trigger_rerun: SimpleCallback,
    api_service: Any,
    execute_enrichment: ExecuteEnrichment,
    refresh_capabilities_block: SimpleCallback,
    render_enrichment_panel: RenderEnrichmentPanel,
) -> None:
    """Render ledger controls, metrics, and lattice tools."""

    _render_ledger_selector(
        add_option_label=add_ledger_option,
        refresh_ledgers=refresh_ledgers,
        create_or_switch_ledger=create_or_switch_ledger,
        validate_ledger_name=validate_ledger_name,
    )

    col_left, col_right = st.columns(2)
    with col_left:
    st.markdown(
        """
        <div class="prime-ledger-block">
            <h2 class="prime-heading" style="font-size: 1.2rem; font-weight: 400">Prime-Ledger Snapshot</h2>
            <p class="prime-text">A live, word-perfect copy of everything you’ve anchored - sealed in primes, mathematically identical forever.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load ledger", key="load_ledger_tab"):
            load_ledger()
        if st.session_state.get("ledger_state"):
            render_ledger_state(st.session_state.ledger_state)
    with col_right:
        st.markdown(
            """
            <div class="about-col about-col-right">
                <h2 class="metrics-heading" style="font-size: 1.25rem; font-weight: 400">Metrics</h2>
                <p class="metrics-paragraph">Tokens Saved = words you never had to re-compute; Integrity = % of anchors that were unique (100 % = zero duplicates); Durability = hours your speech has survived restarts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Tokens Saved", tokens_saved)
        with metric_cols[1]:
            st.metric("Integrity %", f"{ledger_integrity*100:.1f} %")
        with metric_cols[2]:
            st.metric("Durability h", f"{durability_hours:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

    render_entity_metrics_panel(
        api_service,
        entity=get_entity(),
        ledger_id=st.session_state.get("ledger_id"),
    )

    st.markdown("### Möbius lattice rotation")
    if st.button("♾️ Möbius Transform", help="Reproject the exponent lattice"):
        entity = get_entity()
        if not entity:
            st.warning("No active entity.")
        else:
            ledger_id = st.session_state.get("ledger_id")
            try:
                data = perform_lattice_rotation_fn(
                    api_service,
                    entity,
                    ledger_id=ledger_id,
                    axis=(0.0, 0.0, 1.0),
                    angle=1.0472,
                )
                st.success(
                    f"Rotated lattice. Δenergy = {data.get('energy_cycles')}, "
                    f"checksum {data.get('original_checksum')} → {data.get('rotated_checksum')}."
                )
                load_ledger()
                memory_service.note_mobius_rotation(entity, ledger_id=ledger_id)
                memory_service.realign_with_ledger(
                    entity,
                    ledger_id=ledger_id,
                )
                if st.session_state.get("ledger_state"):
                    st.caption("Updated ledger snapshot after Möbius transform:")
                    render_ledger_state(st.session_state.ledger_state)
                trigger_rerun()
            except requests.RequestException as exc:  # type: ignore[name-defined]
                st.error(f"Möbius rotation failed: {exc}")

    st.markdown("### Enrichment")
    if st.button("Initiate Enrichment", help="Replay stored transcripts with richer prime coverage"):
        with st.spinner("Enriching memories…"):
            entity = get_entity()
            if not entity:
                st.warning("No active entity.")
            else:
                summary = execute_enrichment(entity, limit=50)
                st.session_state.latest_enrichment_report = summary
                if summary.get("error"):
                    st.error(summary["error"])
                elif summary.get("message") and not summary.get("enriched"):
                    st.info(summary["message"])
                else:
                    st.success(
                        f"Enriched {summary.get('enriched', 0)}/{summary.get('total', 0)} memories."
                    )
                    failures = summary.get("failures")
                    if failures:
                        st.warning("Some entries failed: " + "; ".join(failures))
                refresh_capabilities_block()
    if st.session_state.get("latest_enrichment_report"):
        render_enrichment_panel(st.session_state.latest_enrichment_report)
