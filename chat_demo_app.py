import audioop
import base64
import hashlib
import html
import io
import json
import logging
from datetime import datetime
import mimetypes
import os
import re
import time
import wave

from pathlib import Path

from typing import Any, Dict, List, Mapping, Sequence
import streamlit as st
try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None
try:
    from openai import OpenAI
    try:
        from openai import BadRequestError as OpenAIClientBadRequest
    except ImportError:
        OpenAIClientBadRequest = None
except ModuleNotFoundError:
    OpenAI = None
    OpenAIClientBadRequest = None
try:
    from pypdf import PdfReader
except ModuleNotFoundError:
    PdfReader = None

from api_client import DualSubstrateV2Client as ApiClient
from app_settings import DEFAULT_METRIC_FLOORS, load_settings
from agent_selector import (
    init_llm_provider,
    render_llm_selector,
    use_openai_provider,
)
from services.api import ApiService, requests
from services.api_service import EnrichmentHelper
from services.ethics_service import EthicsService
from services.ledger_traversal import LedgerTraversalService
from services.memory_service import (
    MemoryService,
    derive_time_filters,
    estimate_quote_count,
    is_recall_query,
    strip_ledger_noise,
)
from services.prompt_service import (
    LEDGER_SNIPPET_LIMIT,
    TRANSLATOR_SYSTEM_PROMPT,
    build_synthesis_prompt,
    build_traversal_intent_prompt,
    create_prompt_service,
)
from services.prime_service import PrimeService
from services.structured_writer import write_structured_views
from services.ledger_tasks import (
    fetch_metrics_snapshot,
    perform_lattice_rotation,
    reset_discrete_ledger,
)
from tabs import about, chat as chat_tab, connectivity_search, ledger_metrics, memory_inference
import ui_coherence
import ui_ethics
from prime_pipeline import (
    call_factor_extraction_llm,
    normalize_override_factors,
)

SETTINGS = load_settings()
os.environ.setdefault("ROCKSDB_DATA_PATH", SETTINGS.rocksdb_data_path)
API = SETTINGS.api_base
DEFAULT_ENTITY = SETTINGS.default_entity
DEFAULT_LEDGER_ID = SETTINGS.default_ledger_id
ADD_LEDGER_OPTION = "➕ Add new ledger…"

GENAI_KEY = SETTINGS.genai_api_key
if genai and GENAI_KEY:
    genai.configure(api_key=GENAI_KEY)

OPENAI_API_KEY = SETTINGS.openai_api_key
ASSET_DIR = Path(__file__).parent
API_CLIENT = ApiClient(API, api_key=SETTINGS.api_key)

_RERUN_FN = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)


def _openai_ready() -> bool:
    return bool(OpenAI and OPENAI_API_KEY)


def _gemini_ready() -> bool:
    return bool(genai and GENAI_KEY)


def _select_llm_provider(prefer_openai: bool) -> str | None:
    openai_ready = _openai_ready()
    gemini_ready = _gemini_ready()
    if prefer_openai and openai_ready:
        return "openai"
    if not prefer_openai and gemini_ready:
        return "gemini"
    if openai_ready:
        return "openai"
    if gemini_ready:
        return "gemini"
    return None


def _get_openai_client():
    if not _openai_ready():
        return None
    client = st.session_state.get("_openai_client")
    if client:
        return client
    client = OpenAI(api_key=OPENAI_API_KEY)
    st.session_state["_openai_client"] = client
    return client


def _build_gemini_model(*, json_mode: bool = False):
    if not _gemini_ready():
        return None
    generation_config = {"response_mime_type": "application/json"} if json_mode else None
    return genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)


def _llm_factor_extractor(text: str, schema: dict[int, dict]) -> list[dict]:
    if not (genai and GENAI_KEY):
        return []
    return call_factor_extraction_llm(text, schema, genai_module=genai)


def _secret(key: str) -> str | None:
    """Retrieve a Streamlit secret defensively."""

    try:
        value = st.secrets.get(key)
    except Exception:
        return None
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _ethics_service() -> EthicsService:
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    return EthicsService(schema=schema)


def _clean_attachment_header(text: str | None) -> str:
    """Remove attachment headers from probe/debug queries."""

    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return re.sub(r"^\[Attachment:[^\]]+\]\s*", "", text).strip()


METRIC_FLOORS = {**DEFAULT_METRIC_FLOORS, **SETTINGS.metric_floors}
API_SERVICE = ApiService(API, SETTINGS.api_key)

PLACEHOLDER_METRICS_GUARD = {
    "dE": -1.0,
    "dDrift": -1.0,
    "dRetention": 1.0,
    "K": 0.0,
}

RECOMMENDED_S2_METRICS = {
    **METRIC_FLOORS,
    "dRetention": 1.0,
    "dE": 1.0,
    "dDrift": 0.0,
    "K": 0.5,
}

SAFE_PROMOTION_METRICS = {
    "dE": -1.0,
    "dDrift": -1.0,
    "dRetention": 0.8,
    "K": 0.0,
}

SEARCH_INDEX_REFRESH_INTERVAL = 90.0
_ATTACHMENT_QUERY_HINTS = ("attachment", "attachments", "pdf", "document", "file", "upload", "chunk")
_TOPIC_TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")


def _get_entity() -> str | None:
    return st.session_state.get("entity")


_S2_PRIME_KEYS = {"11", "13", "17", "19"}
_S2_MIN_WORDS = 120


def _derive_flat_s2_map(structured: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    """Return a sanitized map of S2 entries keyed by allowed prime IDs."""

    result: dict[str, dict[str, str]] = {}
    if not isinstance(structured, Mapping):
        return result

    def record(prime_key: str, entry: Mapping[str, Any]) -> None:
        if not isinstance(entry, Mapping):
            return
        raw_summary = entry.get("summary")
        if isinstance(raw_summary, str):
            summary = raw_summary.strip()
        elif raw_summary is None:
            summary = ""
        else:
            summary = str(raw_summary).strip()
        if summary and prime_key not in result:
            result[prime_key] = {"summary": summary}

    def merge(candidate: object) -> None:
        if isinstance(candidate, Mapping):
            for key in _S2_PRIME_KEYS:
                value = candidate.get(key)
                if isinstance(value, Mapping):
                    record(key, value)
        elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
            for item in candidate:
                if not isinstance(item, Mapping):
                    continue
                prime = item.get("prime")
                if isinstance(prime, int):
                    prime_key = str(prime)
                elif isinstance(prime, str) and prime.isdigit():
                    prime_key = prime
                else:
                    continue
                if prime_key in _S2_PRIME_KEYS:
                    record(prime_key, item)

    merge(structured)
    merge(structured.get("s2"))

    raw_candidate = structured.get("raw")
    merge(raw_candidate)
    if isinstance(raw_candidate, Mapping):
        merge(raw_candidate.get("s2"))

    return result


def _reset_discrete_state() -> bool:
    entity = _get_entity()
    if not entity:
        return False
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    ok, error = reset_discrete_ledger(
        API_SERVICE,
        PRIME_SERVICE,
        entity,
        ledger_id=st.session_state.get("ledger_id"),
        schema=schema,
    )
    if error:
        st.warning(error)
    return ok


def _refresh_ledgers(*, silent: bool = False) -> None:
    try:
        st.session_state.ledgers = API_SERVICE.list_ledgers()
        st.session_state.ledger_refresh_error = None
    except requests.RequestException as exc:
        st.session_state.ledgers = []
        st.session_state.ledger_refresh_error = str(exc)
        if not silent:
            st.error(f"Failed to load ledger list: {exc}")
        return
    if not st.session_state.ledgers:
        st.session_state.ledger_id = DEFAULT_LEDGER_ID
        return
    if not st.session_state.get("ledger_id"):
        st.session_state.ledger_id = st.session_state.ledgers[0]["ledger_id"]


def _create_or_switch_ledger(ledger_id: str, *, notify: bool = True) -> bool:
    ledger_id = (ledger_id or "").strip()
    if not ledger_id:
        if notify:
            st.error("Ledger ID cannot be blank.")
        return False
    try:
        API_SERVICE.create_ledger(ledger_id)
    except requests.RequestException as exc:
        if notify:
            st.error(f"Could not create/switch ledger: {exc}")
        return False

    st.session_state.ledger_id = ledger_id
    MEMORY_SERVICE.clear_entity_cache(entity=_get_entity(), ledger_id=ledger_id)
    if notify:
        st.toast(f"Ledger ready: {ledger_id}", icon="📚")
    return True


def _validate_ledger_name(candidate: str) -> tuple[bool, str | None]:
    ledger_id = (candidate or "").strip()
    if not ledger_id:
        return False, "Ledger ID cannot be empty."
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{1,30})[a-z0-9]", ledger_id):
        return False, "Use 3-32 lowercase letters or numbers; hyphens allowed inside only."
    return True, None


def _ensure_ledger_bootstrap() -> None:
    if "ledger_id" not in st.session_state:
        st.session_state.ledger_id = DEFAULT_LEDGER_ID
    if "ledgers" not in st.session_state:
        st.session_state.ledgers = []
    if "ledger_refresh_error" not in st.session_state:
        st.session_state.ledger_refresh_error = None
    if "latest_enrichment_report" not in st.session_state:
        st.session_state.latest_enrichment_report = None

    if not st.session_state.get("ledgers"):
        _refresh_ledgers(silent=True)
    active = st.session_state.get("ledger_id") or DEFAULT_LEDGER_ID
    if active:
        _create_or_switch_ledger(active, notify=False)


def _auto_promotion_key(entity: str | None, ledger_id: str | None) -> str | None:
    if not entity:
        return None
    normalized_ledger = ledger_id or ""
    return f"{entity}::{normalized_ledger}"


def _summarize_http_response(response):
    summary: dict[str, Any] = {}
    if response is None:
        return summary
    status = getattr(response, "status_code", None)
    if status is not None:
        summary["status"] = status
    detail: Any | None = None
    try:
        detail = response.json()
    except Exception:
        text = (getattr(response, "text", "") or "").strip()
        if text:
            detail = text
    if detail is not None:
        summary["detail"] = detail
    return summary


def _response_ok(summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(summary, Mapping):
        return False
    status = summary.get("status")
    return isinstance(status, int) and 200 <= status < 300


def _promotion_result_ok(result: Mapping[str, Any] | None) -> bool:
    if not isinstance(result, Mapping):
        return False
    return _response_ok(result.get("lawfulness")) and _response_ok(result.get("metrics"))


def _reset_recall_mode() -> None:
    """Return the recall mode selector to the default 'all' state."""

    if st.session_state.get("recall_mode") != "all":
        st.session_state.recall_mode = "all"


def _apply_latest_anchor_to_probe() -> None:
    """Populate the search probe input with the most recent anchor snippet."""

    latest = st.session_state.get("search_probe_latest_preview")
    if latest:
        st.session_state["search_probe_query"] = latest


def _search_index_state(entity: str, ledger_id: str | None) -> dict[str, Any]:
    """Return the dirty/refresh tracking state for a search index."""

    tracker = st.session_state.setdefault("search_index_state", {})
    key = f"{entity}::{ledger_id or ''}"
    state = tracker.get(key)
    if not isinstance(state, dict):
        state = {"dirty": True, "last_build": 0.0}
    tracker[key] = state
    return state


def _mark_search_index_dirty(entity: str | None, ledger_id: str | None) -> None:
    if not entity:
        return
    state = _search_index_state(entity, ledger_id)
    state["dirty"] = True
    state["last_dirty"] = time.time()


def _maybe_refresh_search_index(entity: str | None, ledger_id: str | None, *, force: bool = False) -> None:
    if not entity:
        return
    state = _search_index_state(entity, ledger_id)
    dirty = bool(state.get("dirty", True))
    now = time.time()
    last_build = float(state.get("last_build") or 0.0)
    if not (force or dirty):
        return
    if not force and not dirty and now - last_build < SEARCH_INDEX_REFRESH_INTERVAL:
        return
    params = {"entity": entity}
    if ledger_id:
        params["ledger_id"] = ledger_id
    headers = {}
    if SETTINGS.api_key:
        headers["x-api-key"] = SETTINGS.api_key
    try:
        response = requests.post(
            f"{API.rstrip('/')}/search/index",
            params=params,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning(
            "Search index refresh failed for %s/%s: %s",
            entity,
            ledger_id or "default",
            exc,
        )
        state["last_error"] = str(exc)
        return
    state["last_build"] = now
    state["dirty"] = False
    state["last_error"] = None
    try:
        state["last_status"] = response.json()
    except ValueError:
        state["last_status"] = {"status": response.text[:120] if response.text else "ok"}


def _store_attachment_preview(name: str, chunks: Sequence[str]) -> None:
    if not chunks:
        return
    snippets: list[str] = []
    for chunk in chunks:
        excerpt = (chunk or "").strip()
        if not excerpt:
            continue
        normalized = re.sub(r"\s+", " ", excerpt)
        snippets.append(normalized[:500])
        if len(snippets) >= 5:
            break
    if not snippets:
        return

    entry = {
        "name": name,
        "captured": time.time(),
        "snippets": snippets,
    }
    bucket = st.session_state.setdefault("recent_attachments", [])
    bucket.append(entry)
    st.session_state.recent_attachments = bucket[-3:]


def _references_attachment_query(query: str) -> bool:
    normalized = (query or "").lower()
    return any(hint in normalized for hint in _ATTACHMENT_QUERY_HINTS)


def _topic_terms_from_query(text: str) -> list[str]:
    normalized = (text or "").lower()
    return [token for token in _TOPIC_TOKEN_PATTERN.findall(normalized) if token not in {"the", "and", "about", "from"}]


def _attachment_quote_fallback(query: str) -> str | None:
    attachment_mode = _references_attachment_query(query)
    terms = _topic_terms_from_query(query)
    terms_set = set(terms)
    attachments = st.session_state.get("recent_attachments") or []
    if not attachments:
        return None
    prioritized: list[tuple[str, list[str]]] = []
    for entry in reversed(attachments):
        snippets = entry.get("snippets") or []
        if not snippets:
            continue
        if attachment_mode:
            prioritized.append((entry.get("name") or "attachment", snippets))
            continue
        matched = []
        for snippet in snippets:
            lowered = snippet.lower()
            if not terms_set or any(term in lowered for term in terms_set):
                matched.append(snippet)
        if matched:
            prioritized.append((entry.get("name") or "attachment", matched))
    if not prioritized:
        return None
    name, snippets = prioritized[0]
    lines = [f"Here are excerpts from {name}:"]
    for snippet in snippets[:4]:
        preview = snippet.strip()
        if not preview:
            continue
        if len(preview) > 240:
            preview = f"{preview[:240]}…"
        lines.append(f"- {preview}")
    if len(lines) == 1:
        return None
    return "\n".join(lines)


def _apply_backdoor_promotion(
    entity: str,
    ledger_id: str | None,
    *,
    metrics_payload: Mapping[str, Any],
) -> dict[str, Any]:
    base_url = API.rstrip("/")
    params = {"entity": entity}
    if ledger_id:
        params["ledger_id"] = ledger_id
    headers = {"Content-Type": "application/json"}
    if SETTINGS.api_key:
        headers["x-api-key"] = SETTINGS.api_key
    lawfulness_resp = requests.patch(
        f"{base_url}/ledger/lawfulness",
        params=params,
        headers=headers,
        json={"value": 3},
        timeout=10,
    )
    metrics_resp = requests.patch(
        f"{base_url}/ledger/metrics",
        params=params,
        headers=headers,
        json=dict(metrics_payload or {}),
        timeout=10,
    )
    return {
        "lawfulness": _summarize_http_response(lawfulness_resp),
        "metrics": _summarize_http_response(metrics_resp),
    }


def _update_auto_promotion_tracker(
    entity: str | None,
    ledger_id: str | None,
    *,
    result: Mapping[str, Any] | None,
    error: str | None = None,
):
    key = _auto_promotion_key(entity, ledger_id)
    if not key:
        return
    if "auto_promotion_tracker" not in st.session_state:
        st.session_state.auto_promotion_tracker = {}
    tracker = st.session_state.auto_promotion_tracker
    ok = _promotion_result_ok(result)
    tracker[key] = {
        "ok": ok,
        "result": result,
        "error": error if error else (None if ok else "Non-2xx promotion response"),
        "updated": time.time(),
    }


def _get_auto_promotion_record(entity: str | None, ledger_id: str | None) -> Mapping[str, Any] | None:
    key = _auto_promotion_key(entity, ledger_id)
    if not key:
        return None
    tracker = st.session_state.get("auto_promotion_tracker") or {}
    record = tracker.get(key)
    return record


def _auto_promote_entity_if_needed() -> None:
    entity = _get_entity() or DEFAULT_ENTITY
    ledger_id = st.session_state.get("ledger_id")
    record = _get_auto_promotion_record(entity, ledger_id)
    if record and record.get("ok"):
        return
    if not entity:
        return
    try:
        result = _apply_backdoor_promotion(
            entity,
            ledger_id,
            metrics_payload=SAFE_PROMOTION_METRICS,
        )
    except requests.RequestException as exc:
        _update_auto_promotion_tracker(entity, ledger_id, result=None, error=str(exc))
        return
    _update_auto_promotion_tracker(entity, ledger_id, result=result)
    if _promotion_result_ok(result):
        _reset_recall_mode()


def _fetch_prime_schema(entity: str | None) -> dict[int, dict]:
    target = entity or DEFAULT_ENTITY
    try:
        schema = API_SERVICE.fetch_prime_schema(target, ledger_id=st.session_state.get("ledger_id"))
        if schema:
            return schema
    except requests.RequestException as exc:
        print("[SCHEMA] network fail, using baked defaults:", exc)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print("[SCHEMA] unexpected failure, using baked defaults:", exc)
    return DEFAULT_PRIME_SCHEMA.copy()  # guaranteed fallback


DEFAULT_PRIME_SCHEMA = {
    2: {"name": "Novelty", "tier": "S!", "mnemonic": "spark"},
    3: {"name": "Uniqueness", "tier": "S!", "mnemonic": "spec"},
    5: {"name": "Connection", "tier": "S!", "mnemonic": "stitch"},
    7: {"name": "Action", "tier": "S!", "mnemonic": "step"},
    11: {"name": "Potential", "tier": "S2", "mnemonic": "seed"},
    13: {"name": "Autonomy", "tier": "S2", "mnemonic": "silo"},
    17: {"name": "Relatedness", "tier": "S2", "mnemonic": "system"},
    19: {"name": "Mastery", "tier": "S2", "mnemonic": "standard"},
}
DEFAULT_PRIME_SYMBOLS = {prime: meta["name"] for prime, meta in DEFAULT_PRIME_SCHEMA.items()}
PRIME_ARRAY = tuple(DEFAULT_PRIME_SCHEMA.keys())
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
PRIME_SCHEMA = _fetch_prime_schema(DEFAULT_ENTITY)
if not PRIME_SCHEMA:
    PRIME_SCHEMA = DEFAULT_PRIME_SCHEMA.copy()
PRIME_SYMBOLS = {prime: data["name"] for prime, data in PRIME_SCHEMA.items()}
FALLBACK_PRIME = PRIME_ARRAY[0]

PRIME_SERVICE = PrimeService(API_SERVICE, FALLBACK_PRIME)
MEMORY_SERVICE = MemoryService(API_SERVICE, PRIME_WEIGHTS)
ENRICHMENT_HELPER = EnrichmentHelper(API_SERVICE, PRIME_SERVICE)
PROMPT_SERVICE = create_prompt_service(MEMORY_SERVICE)
LEDGER_TRAVERSAL_SERVICE = LedgerTraversalService(API_SERVICE, MEMORY_SERVICE)

LOGGER = logging.getLogger(__name__)

S1_PRIMES = {2, 3, 5, 7}
S2_PRIMES = {11, 13, 17, 19}


def _coerce_string(value) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _supports_traverse() -> bool:
    key = "__supports_traverse__"
    cached = st.session_state.get(key)
    if cached is None:
        try:
            cached = MEMORY_SERVICE.supports_traverse()
        except Exception:
            cached = False
        st.session_state[key] = bool(cached)
    return bool(cached)


def _supports_inference_state() -> bool:
    key = "__supports_inference_state__"
    cached = st.session_state.get(key)
    if cached is None:
        try:
            cached = MEMORY_SERVICE.supports_inference_state()
        except Exception:
            cached = False
        st.session_state[key] = bool(cached)
    return bool(cached)


def _format_traversal_path_ui(path: Mapping[str, Any]) -> str:
    nodes = path.get("nodes") if isinstance(path, Mapping) else None
    labels: list[str] = []
    if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes)):
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            label = node.get("label") if isinstance(node.get("label"), str) else None
            if not label and isinstance(node.get("prime"), int):
                label = f"Prime {node['prime']}"
            if not label and isinstance(node.get("note"), str):
                label = node["note"]
            label = (label or "node").strip()
            weight = node.get("weight") if isinstance(node.get("weight"), (int, float)) else None
            if weight is not None:
                label = f"{label} ({weight:.2f})"
            labels.append(label)
    if not labels:
        labels.append("(no nodes)")
    score = path.get("score") if isinstance(path.get("score"), (int, float)) else None
    joined = " → ".join(labels)
    if score is not None:
        return f"{joined} — score {score:.2f}"
    return joined


def _render_traversal_tab(entity: str | None) -> None:
    if not entity:
        st.info("Select an entity to inspect traversal paths.")
        return
    payload = MEMORY_SERVICE.traversal_paths(
        entity,
        ledger_id=st.session_state.get("ledger_id"),
        limit=8,
    )
    if not payload.get("supported", True):
        st.info("Traversal endpoint unavailable on this backend.")
        return
    paths = payload.get("paths")
    if not isinstance(paths, Sequence) or not paths:
        message = payload.get("message") if isinstance(payload.get("message"), str) else None
        if message:
            st.info(message)
        else:
            st.info("No traversal paths available yet. Anchor memories to populate the graph.")
        return
    for idx, path in enumerate(paths[:8], start=1):
        if not isinstance(path, Mapping):
            continue
        st.markdown(f"**Path {idx}:** {_format_traversal_path_ui(path)}")
        nodes = path.get("nodes") if isinstance(path.get("nodes"), Sequence) else []
        if nodes:
            for node in nodes:
                if not isinstance(node, Mapping):
                    continue
                prime = node.get("prime")
                label = node.get("label") if isinstance(node.get("label"), str) else None
                note = node.get("note") if isinstance(node.get("note"), str) else None
                weight = node.get("weight") if isinstance(node.get("weight"), (int, float)) else None
                bullet = label or (f"Prime {prime}" if isinstance(prime, int) else "Node")
                if weight is not None:
                    bullet = f"{bullet} ({weight:.2f})"
                if note:
                    bullet = f"{bullet} – {note}"
                st.caption(f"• {bullet}")
        st.divider()


def _format_inference_row(entry: Mapping[str, Any]) -> str:
    if not isinstance(entry, Mapping):
        return ""
    label = entry.get("label") if isinstance(entry.get("label"), str) else None
    if not label and isinstance(entry.get("prime"), int):
        label = f"Prime {entry['prime']}"
    status = entry.get("status") if isinstance(entry.get("status"), str) else None
    score = entry.get("score") if isinstance(entry.get("score"), (int, float)) else None
    note = entry.get("note") if isinstance(entry.get("note"), str) else None
    timestamp = entry.get("timestamp")
    ts_label = None
    if isinstance(timestamp, (int, float)):
        try:
            ts_label = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M")
        except (ValueError, OverflowError, OSError):
            ts_label = None
    parts = [part for part in (label, f"[{status}]" if status else None, ts_label) if part]
    summary = " ".join(parts) if parts else "(entry)"
    if score is not None:
        summary = f"{summary} – score {score:.2f}"
    if note:
        summary = f"{summary} – {note}"
    return summary


def _render_inference_tab(entity: str | None) -> None:
    if not entity:
        st.info("Select an entity to inspect inference status.")
        return
    payload = MEMORY_SERVICE.fetch_inference_state(
        entity,
        ledger_id=st.session_state.get("ledger_id"),
        include_history=True,
        limit=6,
    )
    if not payload.get("supported", True):
        st.info("Inference state endpoint unavailable on this backend.")
        return
    status = payload.get("status") if isinstance(payload.get("status"), str) else None
    if status:
        st.markdown(f"**State:** {status}")
    active = payload.get("active") if isinstance(payload.get("active"), Mapping) else None
    if active:
        st.markdown(f"**Active:** {_format_inference_row(active)}")
    queue = payload.get("queue") if isinstance(payload.get("queue"), Sequence) else []
    if queue:
        st.markdown("**Queue:**")
        for entry in queue[:6]:
            summary = _format_inference_row(entry)
            if summary:
                st.caption(f"• {summary}")
    history = payload.get("history") if isinstance(payload.get("history"), Sequence) else []
    if history:
        st.markdown("**Recent Completions:**")
        for entry in history[:6]:
            summary = _format_inference_row(entry)
            if summary:
                st.caption(f"• {summary}")
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    if metrics:
        metric_rows = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_rows.append(f"{key}: {value:.2f}")
        if metric_rows:
            st.markdown("**Metrics:**")
            for row in metric_rows[:8]:
                st.caption(row)
    message = payload.get("message") if isinstance(payload.get("message"), str) else None
    if message and not (queue or history or active):
        st.info(message)


def _as_string_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                result.append(cleaned)
    return result


def _body_chunks(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    chunks: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    chunks.append(cleaned)
            elif isinstance(item, dict):
                text = _coerce_string(item.get("text"))
                if text:
                    chunks.append(text)
    return chunks


def _normalize_slot(slot: dict) -> dict | None:
    if not isinstance(slot, dict):
        return None
    prime = slot.get("prime")
    if not isinstance(prime, int):
        return None
    title = _coerce_string(slot.get("title") or slot.get("name"))
    summary = _coerce_string(slot.get("summary") or slot.get("synopsis") or slot.get("description"))
    tags = _as_string_list(slot.get("tags"))
    score = slot.get("score")
    if isinstance(score, (int, float)):
        score_value = float(score)
    else:
        score_value = 0.0
    timestamp = slot.get("timestamp")
    if isinstance(timestamp, (int, float)):
        timestamp_value = int(timestamp)
    else:
        timestamp_value = None
    body_candidates = slot.get("body") or slot.get("chunks") or slot.get("body_chunks")
    bodies = _body_chunks(body_candidates)
    normalized = {
        "prime": prime,
        "title": title,
        "summary": summary,
        "tags": tags,
        "score": score_value,
        "timestamp": timestamp_value,
        "body": bodies,
        "raw": slot,
    }
    return normalized


def _extract_structured_views(payload: dict | None) -> dict:
    payload = payload or {}
    slots_raw = payload.get("slots") if isinstance(payload, dict) else None
    if not isinstance(slots_raw, list):
        slots_raw = []
    normalized_slots = [slot for slot in (_normalize_slot(item) for item in slots_raw) if slot]
    ranked_slots = sorted(
        normalized_slots,
        key=lambda item: (item.get("score", 0.0), item.get("timestamp") or 0),
        reverse=True,
    )

    s1_slots = []
    s2_slots = []
    body_entries = []
    for slot in ranked_slots:
        prime = slot["prime"]
        if prime in S1_PRIMES and (slot.get("title") or slot.get("tags")):
            s1_slots.append({
                "prime": prime,
                "title": slot.get("title"),
                "tags": slot.get("tags"),
                "score": slot.get("score", 0.0),
            })
        if prime in S2_PRIMES and (slot.get("summary") or slot.get("body")):
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

    return {
        "raw": payload or {},
        "slots": ranked_slots,
        "s1": s1_slots,
        "s2": s2_slots,
        "bodies": body_entries,
        "updated_at": time.time(),
    }


def _persist_structured_views(entity: str, structured: dict, *, ledger_id: str | None) -> dict:
    if not isinstance(structured, Mapping):
        structured = {}

    flat_map = _derive_flat_s2_map(structured)

    writer_payload: dict[str, Any] = {}
    if structured:
        writer_payload = dict(structured)
        writer_payload["s2"] = []

    score_payload: Mapping[str, Any] | None = None
    metrics_response: Mapping[str, Any] | None = None

    try:
        if writer_payload:
            write_structured_views(
                API_SERVICE,
                entity,
                writer_payload,
                ledger_id=ledger_id,
            )
    except requests.RequestException as exc:
        LOGGER.warning("Failed to persist structured S1 views: %s", exc)

    total_word_count = 0
    if flat_map:
        for entry in flat_map.values():
            summary_value = entry.get("summary") if isinstance(entry, Mapping) else None
            if isinstance(summary_value, str):
                total_word_count += len(summary_value.split())

    if flat_map and total_word_count < _S2_MIN_WORDS:
        LOGGER.info(
            "Skip S2 – text too short (words=%s, threshold=%s)",
            total_word_count,
            _S2_MIN_WORDS,
        )
        return {"s2": flat_map}

    if flat_map:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if SETTINGS.api_key:
            headers["x-api-key"] = SETTINGS.api_key
        if ledger_id:
            headers["X-Ledger-ID"] = ledger_id
        try:
            response = requests.post(
                f"{API.rstrip('/')}/score/s2",
                params={"entity": entity},
                json=flat_map,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            try:
                score_candidate = response.json()
            except ValueError:
                score_candidate = None
            if isinstance(score_candidate, Mapping):
                score_payload = dict(score_candidate)
                metrics_candidate = score_candidate.get("metrics")
                if not isinstance(metrics_candidate, Mapping):
                    metrics_candidate = score_candidate
                if isinstance(metrics_candidate, Mapping):
                    try:
                        metrics_response = API_SERVICE.patch_metrics(
                            entity,
                            metrics_candidate,
                            ledger_id=ledger_id,
                        )
                    except requests.RequestException as exc:
                        LOGGER.warning("Failed to persist S2 metrics: %s", exc)
        except requests.RequestException as exc:
            LOGGER.warning("Failed to score S2 ledger map: %s", exc)

    try:
        API_SERVICE.put_ledger_s2(
            entity,
            flat_map,
            ledger_id=ledger_id,
        )
    except requests.RequestException as exc:
        LOGGER.warning("Failed to persist S2 ledger map: %s", exc)

    result: dict[str, Any] = {"s2": flat_map}
    if isinstance(score_payload, Mapping):
        result["score"] = score_payload
    if isinstance(metrics_response, Mapping):
        result["metrics"] = dict(metrics_response)
    return result


def _extract_structured_persist_outputs(
    payload: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return sanitized S2 map and metrics from a persistence result."""

    s2_map: dict[str, Any] = {}
    metrics_payload: dict[str, Any] = {}

    if isinstance(payload, Mapping):
        s2_candidate = payload.get("s2") if hasattr(payload, "get") else None
        if isinstance(s2_candidate, Mapping):
            s2_map = {k: v for k, v in s2_candidate.items() if isinstance(k, str)}
        else:
            filtered: dict[str, Any] = {}
            for key, value in payload.items():
                if not isinstance(key, str) or key not in _S2_PRIME_KEYS:
                    continue
                if isinstance(value, Mapping):
                    filtered[key] = value
            if filtered:
                s2_map = filtered

        metrics_candidate = payload.get("metrics") if hasattr(payload, "get") else None
        if isinstance(metrics_candidate, Mapping):
            metrics_payload = dict(metrics_candidate)

    return s2_map, metrics_payload


def _persist_structured_views_from_ledger(entity: str) -> None:
    ledger_id = st.session_state.get("ledger_id")
    try:
        payload = API_SERVICE.fetch_ledger(entity, ledger_id=ledger_id)
    except requests.RequestException as exc:
        LOGGER.warning("Failed to refresh ledger after anchor: %s", exc)
        return

    structured_data = _extract_structured_views(payload)
    structured = structured_data if isinstance(structured_data, Mapping) else {}
    flat_map = _derive_flat_s2_map(structured)
    if not structured.get("slots"):
        st.session_state.latest_structured_ledger = flat_map
        st.session_state.latest_structured_metrics = {}
        return

    persisted = _persist_structured_views(entity, structured, ledger_id=ledger_id)
    s2_map, metrics_payload = _extract_structured_persist_outputs(persisted)
    st.session_state.latest_structured_ledger = s2_map
    st.session_state.latest_structured_metrics = metrics_payload


def _execute_enrichment(entity: str, *, limit: int = 50) -> dict[str, Any]:
    ledger_id = st.session_state.get("ledger_id")
    MEMORY_SERVICE.maybe_refresh_mobius_alignment(entity, ledger_id=ledger_id)
    try:
        memories = API_SERVICE.fetch_memories(
            entity,
            ledger_id=ledger_id,
            limit=max(1, min(int(limit or 0), 200)),
        )
    except requests.RequestException as exc:
        return {"error": f"Failed to load memories: {exc}"}

    if not isinstance(memories, list) or not memories:
        return {"message": "No memories available for enrichment.", "enriched": 0, "total": 0, "reports": []}

    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    ethics_service = _ethics_service()

    try:
        ledger_snapshot = API_SERVICE.fetch_ledger(entity, ledger_id=ledger_id)
    except requests.RequestException:
        ledger_snapshot = {}

    enrichment_supported = API_SERVICE.supports_enrich()
    if not enrichment_supported:
        st.warning(
            "Remote enrichment endpoint is unavailable; storing bodies without structured updates."
        )

    summary: dict[str, Any] = {
        "enriched": 0,
        "total": len(memories),
        "reports": [],
        "failures": [],
        "enrichment_supported": enrichment_supported,
    }

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
            ref_prime = PRIME_SERVICE.fallback_prime

        try:
            factor_deltas = PRIME_SERVICE.build_factors(
                text,
                schema,
                llm_extractor=_llm_factor_extractor,
            )
            result = ENRICHMENT_HELPER.submit(
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
            summary.setdefault("failures", []).append(f"{label}: {exc}")
            continue

        flow_errors = (
            result.get("flow_errors") if isinstance(result, dict) else None
        )
        if flow_errors:
            summary.setdefault("failures", []).append(
                f"ref {ref_prime}: {'; '.join(flow_errors)}"
            )
            continue

        if not result.get("enrichment_supported", True):
            summary["enrichment_supported"] = False
            message = (
                "Enrichment endpoint unavailable; bodies stored without remote enrichment."
            )
            failures = summary.setdefault("failures", [])
            if message not in failures:
                failures.append(message)
            result["text"] = text
            summary["reports"].append(result)
            continue

        summary["enriched"] += 1
        response_payload = result.get("response") if isinstance(result, dict) else {}
        structured = response_payload.get("structured") if isinstance(response_payload, dict) else None
        if structured:
            persisted = _persist_structured_views(entity, structured, ledger_id=ledger_id)
            s2_map, metrics_payload = _extract_structured_persist_outputs(persisted)
            st.session_state.latest_structured_ledger = s2_map
            st.session_state.latest_structured_metrics = metrics_payload
        try:
            ledger_snapshot = API_SERVICE.fetch_ledger(entity, ledger_id=ledger_id)
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

    MEMORY_SERVICE.realign_with_ledger(entity, ledger_id=ledger_id)
    if not summary.get("enrichment_supported", True) and not summary.get("failures"):
        summary.setdefault("failures", []).append(
            "Enrichment endpoint unavailable; ledger bodies persisted without enrichment."
        )
    return summary


def _render_enrichment_panel(report: dict | None) -> None:
    if not report:
        return

    enriched = report.get("enriched", 0)
    total = report.get("total", 0)
    st.markdown("#### Enrichment ethics")
    st.caption(f"Latest run processed {enriched}/{total} memories.")

    reports = report.get("reports") if isinstance(report, dict) else None
    if not isinstance(reports, list) or not reports:
        st.info("No enrichment events captured yet.")
        return

    for idx, entry in enumerate(reports[:3], start=1):
        if not isinstance(entry, dict):
            continue
        ref_prime = entry.get("ref_prime")
        header = f"Memory {idx}"
        if isinstance(ref_prime, int):
            header += f" · ref prime {ref_prime}"
        st.markdown(f"**{header}**")

        ethics = entry.get("ethics") if isinstance(entry.get("ethics"), dict) else {}
        cols = st.columns(4)
        cols[0].metric("Lawfulness", f"{float(ethics.get('lawfulness', 0.0)):.2f}")
        cols[1].metric("Evidence", f"{float(ethics.get('evidence', 0.0)):.2f}")
        cols[2].metric("Non-harm", f"{float(ethics.get('non_harm', 0.0)):.2f}")
        cols[3].metric("Coherence", f"{float(ethics.get('coherence', 0.0)):.2f}")

        notes = ethics.get("notes") if isinstance(ethics.get("notes"), list) else []
        for note in notes[:3]:
            st.caption(f"• {note}")
        bodies = entry.get("bodies") if isinstance(entry.get("bodies"), list) else []
        if bodies:
            excerpt = None
            for body_entry in bodies:
                text = body_entry.get("body") if isinstance(body_entry, dict) else None
                if isinstance(text, str) and text.strip():
                    excerpt = text.strip()
                    break
            if excerpt:
                st.caption(f"Excerpt: {excerpt[:200]}")

def _prime_semantics_block() -> str:
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    lines = ["Prime semantics:"]
    for prime in PRIME_ARRAY:
        meta = schema.get(prime, DEFAULT_PRIME_SCHEMA.get(prime, {}))
        name = meta.get("name", f"Prime {prime}")
        tier = meta.get("tier", "")
        mnemonic = meta.get("mnemonic", "")
        descriptor = meta.get("description", "")
        detail = ", ".join(filter(None, [tier, mnemonic, descriptor]))
        if detail:
            lines.append(f"{prime} ({name}) = {detail}")
        else:
            lines.append(f"{prime} ({name})")
    return "\n".join(lines)

def _load_base64_image(name: str) -> str | None:
    path = ASSET_DIR / name
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except FileNotFoundError:
        return None

def _process_memory_text(text: str, use_openai: bool, *, attachments: list[dict] | None = None):
    cleaned = (text or "").strip()
    clean_query = re.sub(r"^\[Attachment:[^\]]+\]\s*", "", cleaned).strip()
    if not cleaned:
        st.warning("Enter some text first.")
        return
    entity = _get_entity()
    ledger_id = st.session_state.get("ledger_id")
    # Let the LLM structure the memory for anchoring.
    factors = _let_llm_structure_memory(cleaned, prefer_openai=use_openai)
    anchored_primary = _anchor(cleaned, record_chat=False, notify=False, factors_override=factors)
    if anchored_primary:
        _maybe_refresh_search_index(entity, ledger_id)
    agent_payload = _maybe_extract_agent_payload(cleaned)
    if agent_payload:
        agent_text, factors_override = agent_payload
        if _anchor(agent_text, record_chat=True, notify=True, factors_override=factors_override):
            _maybe_refresh_search_index(entity, ledger_id)
            st.session_state.chat_history.append(("Agent", agent_text))
        return
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append(("You", cleaned))
    if _maybe_handle_recall_query(clean_query):
        return
    quote_mode = is_recall_query(clean_query)
    quote_count = estimate_quote_count(clean_query) if quote_mode else None
    if attachments:
        for attachment in attachments:
            preview = (attachment.get("text") or "").strip().replace("\n", " ")
            short_preview = preview[:160]
            if len(preview) > len(short_preview):
                short_preview = f"{short_preview}…"
            st.session_state.chat_history.append(
                (
                    "Attachment",
                    f"{attachment.get('name', 'attachment')} → {short_preview}" if short_preview else attachment.get("name", "attachment"),
                )
            )
    bot_reply = _chat_response(
        clean_query,
        use_openai=use_openai,
        quote_count=quote_count,
        attachments=attachments,
    )
    if bot_reply is None:
        bot_reply = ""
    _update_rolling_memory(cleaned, bot_reply, quote_mode=quote_mode)

def _normalize_audio(raw_bytes: bytes) -> io.BytesIO:
    # The OpenAI API expects a file with a name.
    buf = io.BytesIO(raw_bytes)
    buf.name = "input.wav"
    with wave.open(buf, "rb") as wf:
        params = wf.getparams()
        audio = wf.readframes(params.nframes)
        sampwidth = params.sampwidth
        channels = params.nchannels
        rate = params.framerate
    if sampwidth != 2:
        audio = audioop.lin2lin(audio, sampwidth, 2)
        sampwidth = 2
    if channels != 1:
        audio = audioop.tomono(audio, sampwidth, 0.5, 0.5)
        channels = 1
    target_rate = 16000
    if rate != target_rate:
        audio, _ = audioop.ratecv(audio, sampwidth, channels, rate, target_rate, None)
    peak = audioop.max(audio, sampwidth) or 1
    if peak < 8000:
        audio = audioop.mul(audio, sampwidth, min(4.0, 20000 / peak))
    buf = io.BytesIO()
    buf.name = "input.wav"
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_rate)
        wf.writeframes(audio)
    buf.seek(0)
    return buf



def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _semantic_score(prompt: str) -> float:
    client = _get_openai_client()
    if not client:
        return 0.0
    try:
        target = "provide exact quotes from prior user statements"
        emb_prompt = client.embeddings.create(model="text-embedding-3-small", input=prompt).data[0].embedding
        emb_target = client.embeddings.create(model="text-embedding-3-small", input=target).data[0].embedding
        return _cosine(emb_prompt, emb_target)
    except Exception:
        return 0.0


def _refresh_capabilities_block() -> str:
    history: list[tuple[str, str]] = st.session_state.get("chat_history") or []
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    entity = _get_entity()
    block = PROMPT_SERVICE.build_capabilities_block(
        entity=entity,
        schema=schema,
        chat_history=history,
        prime_semantics=_prime_semantics_block(),
        ledger_id=st.session_state.get("ledger_id"),
        last_anchor_error=st.session_state.get("last_anchor_error"),
    )
    st.session_state.capabilities_block = block
    return block


def _extract_json_object(raw: str) -> dict | None:
    if not raw:
        return None
    trimmed = raw.strip()
    if "{" not in trimmed:
        return None
    candidate = trimmed
    if "```" in trimmed:
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = trimmed[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _build_traversal_intent_prompt(question: str, entity: str | None, ledger_id: str | None) -> str:
    default_entity = entity or DEFAULT_ENTITY
    return build_traversal_intent_prompt(
        question,
        default_entity=default_entity,
        ledger_id=ledger_id,
    )


def _translate_traversal_intent(question: str) -> dict | None:
    if not (genai and GENAI_KEY):
        return None
    prompt = _build_traversal_intent_prompt(
        question,
        _get_entity() or DEFAULT_ENTITY,
        st.session_state.get("ledger_id"),
    )
    try:
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config={"response_mime_type": "application/json"},
            system_instruction=TRANSLATOR_SYSTEM_PROMPT,
        )
        response = model.generate_content(prompt)
    except Exception:
        return None
    raw = getattr(response, "text", None) or ""
    return _extract_json_object(raw)


def _record_latest_assembly(
    *,
    entity: str | None,
    ledger_id: str | None,
    k: int | None,
    quote_safe: bool,
    since: int | None,
    assembly: object,
) -> None:
    if not entity or assembly is None:
        return
    st.session_state.latest_assembly = {
        "entity": entity,
        "ledger_id": ledger_id,
        "k": k,
        "quote_safe": quote_safe,
        "since": since,
        "captured_at": time.time(),
        "payload": assembly,
    }


def _augment_prompt(
    user_question: str,
    *,
    attachments: list[dict] | None = None,
    assembly: dict | None = None,
    since: int | None = None,
    until: int | None = None,
    quote_safe: bool | None = None,
    entity: str | None = None,
) -> str:
    entity = entity or _get_entity()
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    ledger_id = st.session_state.get("ledger_id")
    derived_since, derived_until = derive_time_filters(user_question)
    if since is None:
        since = derived_since
    if until is None:
        until = derived_until
    if quote_safe is None:
        quote_safe = is_recall_query(user_question)
    history = st.session_state.get("chat_history") or []
    return PROMPT_SERVICE.build_augmented_prompt(
        entity=entity,
        question=user_question,
        schema=schema,
        chat_history=history,
        ledger_id=ledger_id,
        attachments=attachments or [],
        assembly=assembly,
        since=since,
        until=until,
        quote_safe=quote_safe,
    )


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trigger_rerun():
    if _RERUN_FN:
        try:
            _RERUN_FN()
        except RuntimeError:
            pass


def _extract_transcript_text(transcript) -> str | None:
    if not transcript:
        return None
    direct = getattr(transcript, "text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    data = None
    if isinstance(transcript, dict):
        data = transcript
    else:
        for attr in ("model_dump", "dict", "to_dict"):
            method = getattr(transcript, attr, None)
            if callable(method):
                try:
                    candidate = method()
                except TypeError:
                    try:
                        candidate = method({})
                    except TypeError:
                        continue
                if isinstance(candidate, dict):
                    data = candidate
                    break
    if data:
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        segments = data.get("segments")
        if isinstance(segments, list):
            combined = " ".join(
                seg.get("text", "").strip() for seg in segments if isinstance(seg, dict) and seg.get("text")
            ).strip()
            if combined:
                return combined
    return None
def _ingest_attachment(uploaded_file) -> dict | None:
    if uploaded_file is None:
        return None

    name = getattr(uploaded_file, "name", None) or "attachment"
    mime = getattr(uploaded_file, "type", None) or mimetypes.guess_type(name)[0] or "application/octet-stream"

    try:
        data = uploaded_file.getvalue()
    except AttributeError:
        data = uploaded_file.read()

    if not data:
        return None

    text: str | None = None
    if (mime == "application/pdf" or name.lower().endswith(".pdf")) and PdfReader:
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join(filter(None, (page.extract_text() for page in reader.pages)))
        except Exception:
            text = None
    if text is None and (mime.startswith("text/") or mime in {"application/json", "application/xml", "application/javascript"}):
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
    if text is None:
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = None

    if text is None:
        size_kb = len(data) / 1024
        text = f"(Binary attachment of type {mime} ~{size_kb:.1f} KB could not be decoded to text.)"

    text = _normalize_attachment_text(text)
    max_chars = 8_000
    if len(text) > max_chars:
        text = f"{text[:max_chars]}\n… (truncated)"

    return {"name": name, "mime": mime, "text": text}


def _normalize_attachment_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace single newlines (line-wrapped words) with spaces while keeping paragraph breaks.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_attachment_text(text: str, *, max_chars: int = 900) -> list[str]:
    text = text.strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not paragraph:
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(paragraph) <= max_chars:
                current = paragraph
            else:
                for i in range(0, len(paragraph), max_chars):
                    chunks.append(paragraph[i : i + max_chars])
                current = ""
    if current:
        chunks.append(current)
    return chunks


def _anchor_attachment(attachment: dict):
    name = attachment.get("name") or "attachment"
    text = (attachment.get("text") or "").strip()
    if not text:
        st.session_state.chat_history.append(("Attachment", f"{name} contained no text to anchor."))
        return
    chunks = _chunk_attachment_text(text)
    if not chunks:
        st.session_state.chat_history.append(("Attachment", f"{name} contained no text to anchor."))
        return
    anchored = 0
    total = len(chunks)
    for idx, chunk in enumerate(chunks, 1):
        payload = f"[Attachment: {name} | chunk {idx}/{total}]\n{chunk}"
        if _anchor(payload, record_chat=False, notify=False):
            anchored += 1
        else:
            st.warning(f"Failed to anchor chunk {idx} of {name}.")
    status = (
        f"Anchored {anchored}/{total} chunks from {name}."
        if anchored
        else f"Could not anchor {name} – see warnings above."
    )
    if anchored:
        _maybe_refresh_search_index(_get_entity(), st.session_state.get("ledger_id"))
        _store_attachment_preview(name, chunks)
    st.session_state.chat_history.append(("Attachment", status))


def _maybe_extract_agent_payload(raw_text: str) -> tuple[str, list[dict]] | None:
    cleaned = (raw_text or "").strip()
    if not cleaned.startswith("{") or "factors" not in cleaned:
        return None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    text = (payload.get("text") or "").strip()
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    factors = normalize_override_factors(payload.get("factors"), tuple(schema.keys()))
    if not text or not factors:
        return None
    return text, factors


def _latest_user_transcript(current_request: str, *, limit: int = 5) -> str | None:
    entity = _get_entity()
    if not entity:
        return None
    ledger_id = st.session_state.get("ledger_id")
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    start_ms, _ = derive_time_filters(current_request)
    return MEMORY_SERVICE.latest_user_transcript(
        entity,
        schema,
        ledger_id=ledger_id,
        limit=limit,
        since=start_ms,
    )


def _let_llm_structure_memory(text: str, *, prefer_openai: bool | None = None) -> list[dict] | None:
    """Prompt the LLM to extract key concepts as prime factors."""
    prefer_openai = use_openai_provider() if prefer_openai is None else prefer_openai
    provider = _select_llm_provider(prefer_openai)
    if provider is None:
        st.warning("No LLM provider configured for memory structuring.")
        return None

    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    schema_lines = [f'- {p}: {d.get("name")} ({d.get("mnemonic")})' for p, d in schema.items()]
    schema_str = "\n".join(schema_lines)

    prompt = f"""
    You are an AI assistant that structures memories.
    Analyze the following text and identify the key concepts.
    Represent these concepts as a list of prime factors based on the following schema:
    {schema_str}

    Respond with a JSON object containing a "factors" key, like this:
    {{"factors": [{{"prime": 2, "delta": 1}}, {{"prime": 5, "delta": 2}}]}}

    Text to analyze:
    {text}
    """

    payload: dict | None = None
    provider_label = "OpenAI" if provider == "openai" else "Gemini"
    try:
        if provider == "openai":
            client = _get_openai_client()
            if not client:
                st.warning("OpenAI unavailable for memory structuring.")
                return None
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            payload = _extract_json_object(response.choices[0].message.content)
        else:
            model = _build_gemini_model(json_mode=True)
            if not model:
                st.warning("Gemini API key missing for memory structuring.")
                return None
            response = model.generate_content(prompt)
            payload = _extract_json_object(getattr(response, "text", ""))
    except Exception as e:
        st.error(f"Failed to structure memory with {provider_label}: {e}")
        return None

    factors = payload.get("factors") if isinstance(payload, Mapping) else None
    if isinstance(factors, list):
        return factors
    st.warning(f"{provider_label} did not return structured factors.")
    return None


def _update_rolling_memory(user_text: str, bot_reply: str, quote_mode: bool = False):
    if user_text is None and bot_reply is None:
        return
    st.session_state.rolling_text.append(f"You: {user_text}\nBot: {bot_reply}")
    window_s = 120  # Anchor every 2 minutes
    max_tokens = 500  # Or when the conversation chunk gets long enough
    full_block = "\n".join(st.session_state.rolling_text)
    should_anchor = (
        time.time() - st.session_state.last_anchor_ts > window_s
        or len(full_block.split()) > max_tokens
        or quote_mode
    )
    if should_anchor:
        factors = _let_llm_structure_memory(full_block, prefer_openai=use_openai_provider())
        if _anchor(full_block, record_chat=False, factors_override=factors):
            st.session_state.rolling_text = []
            st.session_state.last_anchor_ts = time.time()
            _maybe_refresh_search_index(_get_entity(), st.session_state.get("ledger_id"))


def _maybe_handle_recall_query(text: str) -> bool:
    """Check for recall triggers and reply with ledger content if matched."""
    clean_query = re.sub(r"^\[Attachment:[^\]]+\]\s*", "", text or "").strip()
    if not is_recall_query(clean_query):
        return False

    entity = _get_entity()
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    ledger_id = st.session_state.get("ledger_id")
    _maybe_refresh_search_index(entity, ledger_id)
    if entity:
        placeholder_metrics = dict(PLACEHOLDER_METRICS_GUARD)
        try:
            API_SERVICE.patch_metrics(
                entity,
                placeholder_metrics,
                ledger_id=ledger_id,
            )
        except requests.RequestException as exc:
            LOGGER.warning("Failed to persist placeholder recall metrics: %s", exc)
    fallback_mode = st.session_state.get("recall_mode") or "all"
    recall_mode, _ = _resolve_recall_mode(clean_query, fallback=fallback_mode)
    LOGGER.info(f"Recall query: '{clean_query}', Mode: '{recall_mode}'")
    try:
        response = MEMORY_SERVICE.build_recall_response(
            entity,
            clean_query,
            schema,
            ledger_id=ledger_id,
            mode=recall_mode,
        )
    except requests.RequestException as exc:
        st.error(f"Recall failed: {exc}")
        st.session_state.recall_mode = "all"
        return True

    LOGGER.info(f"Recall response payload: {response}")
    st.session_state.recall_mode = "all"

    fallback_attachment = False
    response_text = response or ""
    fallback_attachment = (
        not response_text
        or response_text.strip().lower().startswith("- no ledger memories matched")
    )
    if fallback_attachment:
        attachment_response = _attachment_quote_fallback(clean_query)
        if attachment_response:
            st.session_state.chat_history.append(("Bot", attachment_response))
            return True
    if response_text:
        st.session_state.chat_history.append(("Bot", response_text))
    return True


def _anchor(text: str, *, record_chat: bool = True, notify: bool = True, factors_override: list[dict] | None = None):
    entity = _get_entity()
    if not entity:
        st.error("No active entity; cannot anchor.")
        return False

    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    ledger_id = st.session_state.get("ledger_id")
    MEMORY_SERVICE.maybe_refresh_mobius_alignment(entity, ledger_id=ledger_id)
    try:
        ingest_result = PRIME_SERVICE.ingest(
            entity,
            text,
            schema,
            ledger_id=ledger_id,
            factors_override=factors_override,
            llm_extractor=_llm_factor_extractor,
            metadata={"source": "chat_demo"},
        )
    except requests.RequestException as exc:
        st.session_state.last_anchor_error = str(exc)
        st.session_state.last_anchor_payload = None
        _refresh_capabilities_block()
        st.error(f"Anchor failed: {exc}")
        return False

    st.session_state.last_anchor_error = None
    _refresh_capabilities_block()
    flow_errors = (
        ingest_result.get("flow_errors")
        if isinstance(ingest_result, dict)
        else None
    )
    anchor_payload = (
        ingest_result.get("anchor") if isinstance(ingest_result, dict) else None
    )
    st.session_state.last_anchor_payload = anchor_payload
    if flow_errors:
        message = "; ".join(flow_errors)
        st.session_state.last_anchor_error = message
        st.error(f"Anchor blocked: {message}")
        return False
    structured = ingest_result.get("structured") if isinstance(ingest_result, dict) else {}
    if structured:
        persisted = _persist_structured_views(entity, structured, ledger_id=ledger_id)
        s2_map, metrics_payload = _extract_structured_persist_outputs(persisted)
        st.session_state.latest_structured_ledger = s2_map
        st.session_state.latest_structured_metrics = metrics_payload
    else:
        _persist_structured_views_from_ledger(entity)
    _mark_search_index_dirty(entity, ledger_id)
    if record_chat:
        st.session_state.chat_history.append(("You", text))
    if notify:
        st.success("Anchored into ledger.")
    return True


def _recall():
    entity = _get_entity()
    if not entity:
        return
    try:
        payload = API_SERVICE.retrieve(entity, ledger_id=st.session_state.get("ledger_id"))
        st.session_state.recall_payload = payload
    except requests.RequestException as exc:
        st.session_state.recall_payload = {"error": str(exc)}


def _load_ledger():
    entity = _get_entity()
    if not entity:
        return
    try:
        data = API_SERVICE.fetch_ledger(entity, ledger_id=st.session_state.get("ledger_id"))
    except requests.RequestException as exc:
        st.session_state.ledger_state = {"error": str(exc)}
        return

    factors = data.get("factors") if isinstance(data, dict) else None
    if isinstance(factors, list):
        schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
        for item in factors:
            prime = item.get("prime")
            if isinstance(prime, int):
                meta = schema.get(prime, DEFAULT_PRIME_SCHEMA.get(prime, {}))
                item["symbol"] = meta.get("name", f"Prime {prime}")
    st.session_state.ledger_state = data


def _render_ledger_state(data):
    if not data:
        return
    if isinstance(data, dict):
        factors = data.get("factors")
        if isinstance(factors, list) and factors:
            schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
            rows = []
            for item in factors:
                prime = item.get("prime")
                if prime not in PRIME_ARRAY:
                    continue
                meta = schema.get(prime, DEFAULT_PRIME_SCHEMA.get(prime, {}))
                label = meta.get("name", f"Prime {prime}")
                tier = meta.get("tier", "")
                rows.append(
                    {
                        "Prime": f"{prime} ({label}{' | ' + tier if tier else ''})",
                        "Value": item.get("value", 0),
                    }
                )
            if rows:
                st.table(rows)
                return
    st.json(data)


def _chat_response(
    prompt: str,
    use_openai=False,
    *,
    quote_count: int | None = None,
    attachments: list[dict] | None = None,
):
    """Generate a chat response, unshackling the LLM to use the full context."""
    # The new prompt augmentation function provides all the necessary context.
    entity = _get_entity()
    ledger_id = st.session_state.get("ledger_id")
    since, until = derive_time_filters(prompt)
    quote_mode = quote_count is not None and quote_count > 0
    assembly = None
    assembly_k = quote_count if quote_count else LEDGER_SNIPPET_LIMIT
    assembly_quote_safe = quote_mode
    target_entity = entity

    translator_intent = _translate_traversal_intent(prompt)
    if translator_intent:
        traversal = LEDGER_TRAVERSAL_SERVICE.execute_intent(
            translator_intent,
            default_entity=entity or DEFAULT_ENTITY,
            ledger_id=ledger_id,
            quote_safe_default=quote_mode,
            since=since,
        )
        target_entity = traversal.get("entity") or target_entity
        assembly = traversal.get("assembly")
        assembly_k = traversal.get("k") or assembly_k
        assembly_quote_safe = traversal.get("quote_safe", assembly_quote_safe)
        _record_latest_assembly(
            entity=target_entity,
            ledger_id=ledger_id,
            k=assembly_k,
            quote_safe=assembly_quote_safe,
            since=since,
            assembly=assembly,
        )

    if assembly is None and target_entity:
        assembly = MEMORY_SERVICE.assemble_context(
            target_entity,
            ledger_id=ledger_id,
            k=assembly_k,
            quote_safe=assembly_quote_safe,
            since=since,
        )
        _record_latest_assembly(
            entity=target_entity,
            ledger_id=ledger_id,
            k=assembly_k,
            quote_safe=assembly_quote_safe,
            since=since,
            assembly=assembly,
        )
    retrieved_context = _augment_prompt(
        prompt,
        attachments=attachments,
        assembly=assembly,
        since=since,
        until=until,
        quote_safe=assembly_quote_safe,
        entity=target_entity,
    )
    llm_prompt = build_synthesis_prompt(prompt, retrieved_context)

    provider = _select_llm_provider(use_openai)
    if provider == "openai":
        client = _get_openai_client()
        if not client:
            if _gemini_ready():
                st.warning("OpenAI unavailable – switching to Gemini.")
                provider = "gemini"
            else:
                st.warning("OpenAI API key missing.")
                return None
    if provider == "openai":
        messages = [{"role": "user", "content": llm_prompt}]
        try:
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        except Exception as exc:
            if OpenAIClientBadRequest and isinstance(exc, OpenAIClientBadRequest):
                detail = getattr(exc, "message", None) or getattr(getattr(exc, "response", None), "text", "") or str(exc)
                st.error(f"OpenAI request rejected: {detail}")
            else:
                st.error(f"OpenAI request failed: {exc}")
            return None
        full = response.choices[0].message.content
        st.session_state.chat_history.append(("Bot", full))
        return full

    if provider != "gemini":
        return "Gemini API key missing."
    model = _build_gemini_model()
    if not model:
        return "Gemini API key missing."
    # Provide a more complete chat history to the model.
    history = [
        {"role": "user" if h[0] == "You" else "model", "parts": [h[1]]}
        for h in st.session_state.chat_history
    ]
    chat = model.start_chat(history=history)
    chunks = []
    for chunk in chat.send_message(llm_prompt, stream=True):
        chunks.append(chunk.text)
    full = "".join(chunks).strip()
    st.session_state.chat_history.append(("Bot", full))
    return full or "(No response)"


def _get_query_params() -> dict:
    try:
        return st.query_params.to_dict()
    except Exception:
        return {}


def _reset_session():
    st.session_state.clear()
    _trigger_rerun()


def _reset_chat_state(clear_query: bool = True):
    st.session_state.chat_history = []
    st.session_state.ledger_state = None
    st.session_state.recall_payload = None
    st.session_state.rolling_text = []
    st.session_state.last_anchor_ts = time.time()
    st.session_state.input_mode = "text"
    st.session_state.top_input = ""
    st.session_state.prefill_top_input = None
    st.session_state.pending_attachments = []
    if clear_query:
        try:
            st.query_params.clear()
        except Exception:
            pass


def _maybe_handle_demo_mode():
    params = _get_query_params()
    if params.get("demo") not in ("true", "1"):
        return
    if _reset_discrete_state():
        st.toast("Demo mode: Ledger has been reset.", icon="✅")
    else:
        st.toast("Demo mode: Ledger reset failed.", icon="⚠️")
    _reset_chat_state(clear_query=True)


def _get_digest(key: str, fallback: str | None = None) -> str | None:
    return _secret(key) or os.getenv(key) or fallback


def _payload_contains_body_shards(value: object, *, depth: int = 0, max_depth: int = 4) -> bool:
    if depth > max_depth:
        return False
    if isinstance(value, Mapping):
        bodies = value.get("bodies")
        if isinstance(bodies, Sequence) and bodies:
            return True
        s1_section = value.get("s1")
        if isinstance(s1_section, Mapping):
            s1_bodies = s1_section.get("bodies")
            if isinstance(s1_bodies, Sequence) and s1_bodies:
                return True
        payload = value.get("payload")
        if _payload_contains_body_shards(payload, depth=depth + 1, max_depth=max_depth):
            return True
        for item in value.values():
            if _payload_contains_body_shards(item, depth=depth + 1, max_depth=max_depth):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if _payload_contains_body_shards(item, depth=depth + 1, max_depth=max_depth):
                return True
    return False


_RECALL_MODE_PATTERNS: dict[str, re.Pattern[str]] = {
    "body": re.compile(r"\b(body|attachments?|attachment|transcript|document|pdf|full[\s-]?text)\b", re.IGNORECASE),
    "slots": re.compile(r"\bslots?\b", re.IGNORECASE),
    "s1": re.compile(r"\bs1\b", re.IGNORECASE),
    "s2": re.compile(r"\bs2\b|\bprime\s*(11|13|17|19)\b", re.IGNORECASE),
}


def _resolve_recall_mode(query: str, fallback: str = "all") -> tuple[str, bool]:
    """Infer the best recall mode from the prompt text."""

    normalized = (query or "").strip()
    for mode, pattern in _RECALL_MODE_PATTERNS.items():
        if pattern.search(normalized):
            return mode, True
    return (fallback or "all", False)


def _default_recall_mode() -> str:
    return "all"


DEMO_USERS = {
    "Developer": {
        "entity": "Demo_dev",
        "digest": _get_digest("DEMO_DEV_DIGEST", "21e467a7efd35f56"),
    },
    "Demo user": {
        "entity": "Demo_new",
        "digest": _get_digest("DEMO_NEW_DIGEST", "2de003d819aafc55"),
    },
}


def _check_password(password: str, digest: str) -> bool:
    if not password or not digest:
        return False
    hasher = hashlib.sha1(password.encode())
    return hasher.hexdigest()[:16] == digest


def _handle_login():
    user_type = st.session_state.get("login_type")
    password = st.session_state.get("login_password")
    if not user_type or not password:
        st.sidebar.error("Please select a user type and enter a password.")
        return

    user_data = DEMO_USERS.get(user_type)
    if not user_data or not _check_password(password, user_data["digest"]):
        st.sidebar.error("Invalid credentials.")
        st.session_state.authenticated = False
        return

    st.session_state.authenticated = True
    st.session_state.entity = user_data["entity"]
    st.session_state.user_type = user_type
    MEMORY_SERVICE.clear_entity_cache()
    _reset_chat_state(clear_query=False)

    if user_type == "Demo user":
        if _reset_discrete_state():
            st.toast("New demo session started. Ledger has been reset.", icon="✅")
        else:
            st.toast("Ledger reset failed.", icon="⚠️")
        st.session_state.login_time = time.time()
    st.session_state.prime_schema = _fetch_prime_schema(user_data["entity"])
    st.session_state.prime_symbols = {
        prime: meta.get("name", f"Prime {prime}") for prime, meta in st.session_state.prime_schema.items()
    }
    _refresh_capabilities_block()


def _render_login_form():
    st.sidebar.selectbox(
        "Select user type",
        ["Developer", "Demo user"],
        index=None,
        placeholder="Select user type...",
        key="login_type",
    )
    st.sidebar.text_input("Enter password", type="password", key="login_password")
    st.sidebar.button("Submit", on_click=_handle_login, key="login_submit")


def _render_app():
    st.set_page_config(page_title="Ledger Chat", layout="wide")

    if "prime_schema" not in st.session_state:
        st.session_state.prime_schema = _fetch_prime_schema(_get_entity() or DEFAULT_ENTITY)
    if "prime_symbols" not in st.session_state:
        st.session_state.prime_symbols = {
            prime: meta.get("name", f"Prime {prime}")
            for prime, meta in (st.session_state.prime_schema or PRIME_SCHEMA).items()
        }
    if "last_anchor_error" not in st.session_state:
        st.session_state.last_anchor_error = None
    if "last_anchor_payload" not in st.session_state:
        st.session_state.last_anchor_payload = None
    if "capabilities_block" not in st.session_state:
        _refresh_capabilities_block()
    if "recall_mode" not in st.session_state:
        st.session_state.recall_mode = _default_recall_mode()

    if not st.session_state.get("authenticated"):
        _render_login_form()
        return

    _maybe_handle_demo_mode()
    _ensure_ledger_bootstrap()
    _auto_promote_entity_if_needed()

    send_icon = _load_base64_image("right-up.png")
    attach_icon = _load_base64_image("add.png")
    mic_icon = _load_base64_image("marketing.png")

    css_chunks = [
        ".main-title {font-size:2rem !important;font-weight:400 !important;text-align:center;margin-top:0.5rem;margin-bottom:0.5rem;}",
        ".stBottomBlockContainer {position:static !important;margin-top:0 !important;}",
        ".stVerticalBlock:has(> .st-key-top_attach) {position:relative;display:flex;justify-content:center;align-items:center;gap:0;}",
        ".stVerticalBlock:has(> .st-key-top_attach) > .st-key-top_input {flex:1 1 auto;}",
        ".stVerticalBlock:has(> .st-key-top_attach) > .st-key-top_attach,.stVerticalBlock:has(> .st-key-top_attach) > .st-key-top_mic {flex:0 0 auto;}",
        ".st-key-top_attach {position: absolute; left:5px; z-index: 100; opacity: 0.5; bottom: 2px}",
        ".st-key-top_mic {position: absolute; right:40px !important; bottom: 2px; opacity: 0.5;}",
        ".exaa2ht0 div[data-baseweb='textarea'] {max-height:120px !important; min-height:120px !important;}",
        "div.exaa2ht1 {max-height:120px !important; min-height:120px !important;}",
        ".st-key-top_mic > button {}",
        "div[data-testid='stChatInput'] {position:static !important;margin:0.25rem auto 0;}",
        "div[data-testid='stChatInput'] > div:first-child {position:relative;border:1px solid rgba(255,255,255,0.18);padding:1.5rem 4.5rem 1.5rem 3.25rem;transition:border-color 0.2s ease, box-shadow 0.2s ease;}",
        "div[data-testid='stChatInput']:focus-within > div:first-child {border-color:rgba(255,255,255,0.3);box-shadow:0 0 0 1px rgba(255,255,255,0.18);}",
        "textarea[data-testid='stChatInputTextArea'] {max-height:120px!important; overflow: none !important; padding-left:0 !important;padding-right:0 !important; padding-top: 25px}",
        "textarea[data-testid='stChatInputTextArea'] {min-height:120px !important;}",
        "div.stElementContainer .st-bw {}",
        ".st-key-top_attach button div,.st-key-top_mic button div {display:none;}",
        ".st-key-top_attach button,.st-key-top_mic button {width:38px;height:38px;background-color:rgba(255,255,255,0.08);background-repeat:no-repeat;background-position:center;background-size:24px 24px;border:1px solid rgba(255,255,255,0.14);transition:background-color 0.2s ease,border-color 0.2s ease;}",
        ".st-key-top_attach button:hover,.st-key-top_mic button:hover {border-color:rgba(255,255,255,0.35);background-color:rgba(255,255,255,0.12);}",
        "div[data-testid='stChatInput'] button[data-testid='stChatInputSubmitButton'] {width:44px;height:44px;border:none;background-color:rgba(255,255,255,0.08);background-repeat:no-repeat;background-position:center;background-size:24px 24px;}",
        "div[data-testid='stChatInput'] button[data-testid='stChatInputSubmitButton']:not(:disabled) {opacity:1;}",
        "div[data-testid='stChatInput'] button[data-testid='stChatInputSubmitButton'] svg {display:none;}",
        "div[data-testid='stChatInput'] button[data-testid='stChatInputSubmitButton']:disabled {opacity:0.5;}",
    ]

    if attach_icon:
        css_chunks.append(
            f".st-key-top_attach button {{background-image:url('data:image/png;base64,{attach_icon}');}}"
        )
    if mic_icon:
        css_chunks.append(
            f".st-key-top_mic button {{background-image:url('data:image/png;base64,{mic_icon}');}}"
        )
    if send_icon:
        css_chunks.append(
            f"div[data-testid='stChatInput'] button[data-testid='stChatInputSubmitButton'] {{background-image:url('data:image/png;base64,{send_icon}');}}"
        )

    style_block = "\n".join(css_chunks)
    st.markdown(
        f"""
        <style>
        {style_block}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<h1 class="main-title">What needs remembering next?</h1>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_audio_digest" not in st.session_state:
        st.session_state.last_audio_digest = None
    if "ledger_state" not in st.session_state:
        st.session_state.ledger_state = None
    if "recall_payload" not in st.session_state:
        st.session_state.recall_payload = None
    if "rolling_text" not in st.session_state:
        st.session_state.rolling_text = []
    if "last_anchor_ts" not in st.session_state:
        st.session_state.last_anchor_ts = time.time()
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "text"
    if "top_input" not in st.session_state:
        st.session_state["top_input"] = ""
    if "prefill_top_input" not in st.session_state:
        st.session_state.prefill_top_input = None
    if "pending_attachments" not in st.session_state:
        st.session_state.pending_attachments = []
    if "login_time" not in st.session_state:
        st.session_state.login_time = None
    if "prime_symbols" not in st.session_state:
        st.session_state.prime_symbols = DEFAULT_PRIME_SYMBOLS
    init_llm_provider(
        openai_ready=_openai_ready(),
        gemini_ready=_gemini_ready(),
    )

    if st.session_state.user_type == "Demo user" and st.session_state.login_time:
        remaining = 600 - (time.time() - st.session_state.login_time)
        if remaining <= 0:
            st.sidebar.error("Demo session has expired.")
            if st.sidebar.button("Log out"):
                _reset_session()
            return
        st.sidebar.info(f"Time remaining: {int(remaining // 60)}m {int(remaining % 60)}s")

    st.sidebar.button("Log out", on_click=_reset_session, key="logout_button")

    with st.sidebar.expander("Debugging"):
        st.subheader("Raw Ledger")
        entity = _get_entity()
        if entity:
            raw = MEMORY_SERVICE.memory_lookup(
                entity,
                ledger_id=st.session_state.get("ledger_id"),
                limit=20,
            )
        else:
            raw = []
        for m in raw:
            st.caption(f"**{m.get('timestamp', 'N/A')}**")
            st.code(m.get('text', '')[:200] + ("…" if len(m.get('text', '')) > 200 else ""))

    render_llm_selector(
        openai_ready=_openai_ready(),
        gemini_ready=_gemini_ready(),
    )

    if st.session_state.get("prefill_top_input"):
        st.session_state["top_input"] = st.session_state.prefill_top_input
        st.session_state.prefill_top_input = None

    with st.container():
        attach_clicked = st.button("Attach", key="top_attach", help="Attach a memory file", type="secondary")
        prompt_top = st.chat_input("type, speak or attach a new memory", key="top_input")
        mic_clicked = st.button("Mic", key="top_mic", help="Record voice memory", type="secondary")

    if attach_clicked:
        st.session_state.input_mode = "file"
    if mic_clicked:
        st.session_state.input_mode = "mic"

    if prompt_top:
        attachments = list(st.session_state.pending_attachments)
        _process_memory_text(prompt_top, use_openai=use_openai_provider(), attachments=attachments)
        st.session_state.pending_attachments = []
        # Streamlit clears chat inputs automatically after submission, so avoid
        # writing to the widget-managed key here to prevent SessionState errors.

    # ---------- investor KPI ----------
    entity = _get_entity()
    snapshot = fetch_metrics_snapshot(
        API_SERVICE,
        entity,
        ledger_id=st.session_state.get("ledger_id"),
        metric_floors=METRIC_FLOORS,
    )
    tokens_saved_value = _coerce_float(snapshot.get("tokens_saved"))
    ledger_integrity = _coerce_float(snapshot.get("ledger_integrity")) or METRIC_FLOORS["ledger_integrity"]
    durability_h = _coerce_float(snapshot.get("durability_hours")) or METRIC_FLOORS["durability_h"]
    tokens_saved = f"{int(tokens_saved_value or 0):,}"
    if snapshot.get("error"):
        st.warning(f"Metrics unavailable: {snapshot['error']}")

    if st.session_state.input_mode == "mic":
        st.info("Voice mode active – hold to record.")
        audio = st.audio_input("Hold to talk", key="voice_input")
        if audio:
            audio_bytes = audio.getvalue()
            digest = hashlib.sha1(audio_bytes).hexdigest()
            if digest != st.session_state.last_audio_digest:
                st.session_state.last_audio_digest = digest
                norm = _normalize_audio(audio_bytes)
                client = _get_openai_client()
                if not client:
                    st.warning("OpenAI API key missing.")
                else:
                    try:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=norm,
                        )
                        text = _extract_transcript_text(transcript)
                        if text:
                            st.caption(f"Transcript: {text}")
                            st.session_state.prefill_top_input = text
                            _trigger_rerun()
                        else:
                            st.warning("No transcript returned from Whisper.")
                    except Exception as exc:
                        st.error(f"Transcription failed: {exc}")
            st.session_state.input_mode = "text"
    elif st.session_state.input_mode == "file":
        uploaded = st.file_uploader("Attach a new memory", label_visibility="collapsed")
        if uploaded:
            attachment = _ingest_attachment(uploaded)
            if attachment:
                st.session_state.pending_attachments.append(attachment)
                _anchor_attachment(attachment)
                snippet_preview = (attachment.get("text") or "").strip()
                preview = snippet_preview[:140].replace("\n", " ")
                if len(snippet_preview) > 140:
                    preview += "…"
                st.caption(f"Attached {attachment['name']} ({attachment['mime']}). Preview: {preview}")
            else:
                st.warning("Could not read the uploaded attachment.")
            st.session_state.input_mode = "text"

    traversal_supported = _supports_traverse()
    inference_supported = _supports_inference_state()
    tabs = st.tabs(
        [
            "Chat",
            "Memory & Inference",
            "Connectivity & Search",
            "Ledger & Metrics",
            "About DualSubstrate",
            "Coherence",
            "Ethics",
        ]
    )

    with tabs[0]:
        chat_tab.render_tab(
            st.session_state.chat_history,
            st.session_state.pending_attachments,
        )

    with tabs[1]:
        memory_inference.render_tab(
            entity=_get_entity(),
            traversal_supported=traversal_supported,
            inference_supported=inference_supported,
            render_traversal_callback=_render_traversal_tab,
            render_inference_callback=_render_inference_tab,
            inference_snapshot=snapshot,
        )

    with tabs[2]:
        connectivity_search.render_tab(
            api_base=API,
            settings=SETTINGS,
            api_service=API_SERVICE,
            default_entity=DEFAULT_ENTITY,
            get_entity=_get_entity,
            clean_attachment_header=_clean_attachment_header,
            apply_backdoor_promotion=_apply_backdoor_promotion,
            promotion_result_ok=_promotion_result_ok,
            reset_recall_mode=_reset_recall_mode,
            update_auto_promotion_tracker=_update_auto_promotion_tracker,
            get_auto_promotion_record=_get_auto_promotion_record,
            recommended_s2_metrics=RECOMMENDED_S2_METRICS,
            safe_promotion_metrics=SAFE_PROMOTION_METRICS,
            derive_flat_s2_map=_derive_flat_s2_map,
            s2_prime_keys=_S2_PRIME_KEYS,
        )

    with tabs[3]:
        ledger_metrics.render_tab(
            tokens_saved=tokens_saved,
            ledger_integrity=ledger_integrity,
            durability_hours=durability_h,
            add_ledger_option=ADD_LEDGER_OPTION,
            refresh_ledgers=_refresh_ledgers,
            create_or_switch_ledger=_create_or_switch_ledger,
            validate_ledger_name=_validate_ledger_name,
            load_ledger=_load_ledger,
            render_ledger_state=_render_ledger_state,
            get_entity=_get_entity,
            memory_service=MEMORY_SERVICE,
            perform_lattice_rotation_fn=perform_lattice_rotation,
            trigger_rerun=_trigger_rerun,
            api_service=API_SERVICE,
            execute_enrichment=_execute_enrichment,
            refresh_capabilities_block=_refresh_capabilities_block,
            render_enrichment_panel=_render_enrichment_panel,
        )

    with tabs[4]:
        about.render_tab()

    with tabs[5]:
        ui_coherence.render_coherence_tab(API_CLIENT)

    with tabs[6]:
        ui_ethics.render_ethics_tab(API_CLIENT)


def main() -> None:
    """Streamlit entry point for the public chat demo."""

    _render_app()
    if not st.session_state.get("capabilities_block"):
        _refresh_capabilities_block()


if __name__ == "__main__":
    main()
