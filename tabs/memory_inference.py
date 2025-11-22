"""Memory and inference tab renderer."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import streamlit as st

try:
    from tests.rocksdb_probe import run_probe as run_rocksdb_probe
except Exception:  # pragma: no cover - probe is optional in prod builds
    run_rocksdb_probe = None


RenderCallback = Callable[[str | None], None]


def _render_section(title: str, callback: RenderCallback, entity: str | None) -> None:
    with st.expander(title, expanded=True):
        callback(entity)


def render_tab(
    *,
    entity: str | None,
    traversal_supported: bool,
    inference_supported: bool,
    render_traversal_callback: RenderCallback,
    render_inference_callback: RenderCallback,
    inference_snapshot: Mapping[str, Any] | None,
    advanced_probes_enabled: bool,
    enable_rocksdb_probe: bool,
) -> None:
    """Render traversal, inference, and telemetry panels."""

    if not advanced_probes_enabled:
        st.button(
            "Advanced probes disabled",
            disabled=True,
            help="Traversal and inference probes are turned off on this demo backend",
        )
        st.caption(
            "Traversal, inference, and telemetry panels are unavailable on this demo backend."
        )
        return

    if traversal_supported:
        _render_section("Traversal snapshot", render_traversal_callback, entity)
    else:
        st.info("Traversal probes are disabled on this demo backend; no paths to display.")

    st.divider()

    if inference_supported:
        _render_section("Inference status", render_inference_callback, entity)
    else:
        st.info("Inference state endpoint unavailable on this deployment.")

    telemetry_snapshot = inference_snapshot or {}
    telemetry_state = telemetry_snapshot.get("inference_state")
    telemetry_traverse = telemetry_snapshot.get("inference_traverse")
    telemetry_memories = telemetry_snapshot.get("inference_memories")
    telemetry_retrieve = telemetry_snapshot.get("inference_retrieve")
    telemetry_supported = telemetry_snapshot.get("inference_supported")
    telemetry_errors = telemetry_snapshot.get("inference_errors") or []

    telemetry_payloads = [
        ("State", telemetry_state),
        ("Traverse", telemetry_traverse),
        ("Memories", telemetry_memories),
        ("Retrieve", telemetry_retrieve),
    ]
    telemetry_any = any(payload is not None for _, payload in telemetry_payloads)

    if telemetry_any:
        st.divider()
        st.markdown("#### Inference telemetry")
        for label, payload in telemetry_payloads:
            if payload is None:
                continue
            expanded = label == "State"
            with st.expander(label, expanded=expanded):
                if isinstance(payload, (list, dict)):
                    st.json(payload)
                else:
                    st.write(payload)
    elif telemetry_supported is False:
        st.caption("Inference telemetry endpoints are not available on this deployment.")
    elif telemetry_errors:
        st.warning("Inference telemetry unavailable: " + "; ".join(telemetry_errors))

    if enable_rocksdb_probe:
        _render_rocksdb_probe(entity)
    else:
        st.divider()
        st.info("RocksDB filesystem probe is disabled in this hosted demo.")


def _render_rocksdb_probe(entity: str | None) -> None:
    st.divider()
    st.markdown("#### RocksDB probe")
    st.caption(
        "Inspect the ledger's raw key/value pairs and compare them with "
        "`MemoryService.memory_lookup()` output to verify prime traversal patterns."
    )
    st.warning(
        "This probe queries the live RocksDB path configured via ROCKSDB_DATA_PATH."
        " Run it sparingly in production environments."
    )
    if run_rocksdb_probe is None:
        st.info("RocksDB probe unavailable (install rocksdict to enable it).")
        return

    default_entity = entity or "Demo_dev"
    with st.form("rocksdb_probe_form", clear_on_submit=False):
        probe_entity = st.text_input("Entity ID", value=default_entity, key="rocksdb_probe_entity")
        probe_prompt = st.text_area(
            "Synthetic prompt", value="kangaroo neon laser", key="rocksdb_probe_prompt"
        )
        probe_pattern = st.text_input(
            "Prime pattern", value="2*3*5*7", key="rocksdb_probe_pattern"
        )
        probe_top_n = st.number_input(
            "Result count", min_value=1, max_value=100, value=20, step=1, key="rocksdb_probe_topn"
        )
        submitted = st.form_submit_button("Run RocksDB probe")

    if submitted:
        sanitized_entity = (probe_entity or "").strip()
        if not sanitized_entity:
            st.error("Entity is required before running the probe.")
        else:
            with st.spinner("Walking RocksDB keyspace…"):
                try:
                    result = run_rocksdb_probe(
                        sanitized_entity,
                        probe_prompt,
                        probe_pattern,
                        int(probe_top_n),
                    )
                except PermissionError:
                    st.info("RocksDB filesystem probe is disabled in this hosted environment.")
                except FileNotFoundError as exc:
                    st.error(f"RocksDB path not found: {exc}")
                except Exception as exc:  # pragma: no cover - surface probe errors
                    st.error(f"RocksDB probe failed: {exc}")
                else:
                    st.session_state["rocksdb_probe_result"] = result
                    st.success("RocksDB probe completed. Compare the hits with the sidebar ledger view.")

    existing = st.session_state.get("rocksdb_probe_result")
    if existing:
        with st.expander("Latest RocksDB probe results", expanded=False):
            st.json(existing)
