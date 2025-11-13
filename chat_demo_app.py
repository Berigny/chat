import audioop
import base64
import hashlib
import html
import io
import json
import logging
import mimetypes
import os
import re
import time
import wave

from pathlib import Path

from typing import Dict, List
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

from app_settings import DEFAULT_METRIC_FLOORS, load_settings
from agent_selector import (
    init_llm_provider,
    render_llm_selector,
    use_openai_provider,
)
from services.api import ApiService, requests
from services.memory_service import (
    MemoryService,
    derive_time_filters,
    estimate_quote_count,
    is_recall_query,
    strip_ledger_noise,
)
from services.prompt_service import create_prompt_service
from services.prime_service import create_prime_service
from services.ledger_service import persist_structured_views
from services.ledger_tasks import (
    fetch_metrics_snapshot,
    perform_lattice_rotation,
    reset_discrete_ledger,
    run_enrichment_job,
)
from prime_pipeline import (
    call_factor_extraction_llm,
    normalize_override_factors,
)

SETTINGS = load_settings()
API = SETTINGS.api_base
DEFAULT_ENTITY = SETTINGS.default_entity
DEFAULT_LEDGER_ID = SETTINGS.default_ledger_id
ADD_LEDGER_OPTION = "➕ Add new ledger…"

GENAI_KEY = SETTINGS.genai_api_key
if genai and GENAI_KEY:
    genai.configure(api_key=GENAI_KEY)

OPENAI_API_KEY = SETTINGS.openai_api_key
ASSET_DIR = Path(__file__).parent

_RERUN_FN = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)


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


METRIC_FLOORS = {**DEFAULT_METRIC_FLOORS, **SETTINGS.metric_floors}
API_SERVICE = ApiService(API, SETTINGS.api_key)


def _get_entity() -> str | None:
    return st.session_state.get("entity")


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
            st.sidebar.error(f"Failed to load ledger list: {exc}")
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
            st.sidebar.error("Ledger ID cannot be blank.")
        return False
    try:
        API_SERVICE.create_ledger(ledger_id)
    except requests.RequestException as exc:
        if notify:
            st.sidebar.error(f"Could not create/switch ledger: {exc}")
        return False

    st.session_state.ledger_id = ledger_id
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

    if not st.session_state.get("ledgers"):
        _refresh_ledgers(silent=True)
    active = st.session_state.get("ledger_id") or DEFAULT_LEDGER_ID
    if active:
        _create_or_switch_ledger(active, notify=False)


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

PRIME_SERVICE = create_prime_service(API_SERVICE, FALLBACK_PRIME)
MEMORY_SERVICE = MemoryService(API_SERVICE, PRIME_WEIGHTS)
PROMPT_SERVICE = create_prompt_service(MEMORY_SERVICE)

LOGGER = logging.getLogger(__name__)

S1_PRIMES = {2, 3, 5, 7}
S2_PRIMES = {11, 13, 17, 19}


def _coerce_string(value) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


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
    if not structured:
        return {"slots": [], "s1": [], "s2": [], "bodies": []}
    try:
        persisted = persist_structured_views(
            API_SERVICE,
            entity,
            structured,
            ledger_id=ledger_id,
        )
    except requests.RequestException as exc:
        LOGGER.warning("Failed to persist structured views: %s", exc)
        return structured or {"slots": [], "s1": [], "s2": [], "bodies": []}
    return persisted


def _persist_structured_views_from_ledger(entity: str) -> None:
    ledger_id = st.session_state.get("ledger_id")
    try:
        payload = API_SERVICE.fetch_ledger(entity, ledger_id=ledger_id)
    except requests.RequestException as exc:
        LOGGER.warning("Failed to refresh ledger after anchor: %s", exc)
        return

    structured = _extract_structured_views(payload)
    if not structured.get("slots"):
        MEMORY_SERVICE.update_structured_ledger(entity, structured, ledger_id=ledger_id)
        st.session_state.latest_structured_ledger = structured
        return

    persisted = _persist_structured_views(entity, structured, ledger_id=ledger_id)
    MEMORY_SERVICE.update_structured_ledger(entity, persisted, ledger_id=ledger_id)
    st.session_state.latest_structured_ledger = persisted


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
    if not cleaned:
        st.warning("Enter some text first.")
        return
    # Let the LLM structure the memory for anchoring.
    factors = _let_llm_structure_memory(cleaned)
    _anchor(cleaned, record_chat=False, notify=False, factors_override=factors)
    agent_payload = _maybe_extract_agent_payload(cleaned)
    if agent_payload:
        agent_text, factors_override = agent_payload
        if _anchor(agent_text, record_chat=True, notify=True, factors_override=factors_override):
            st.session_state.chat_history.append(("Agent", agent_text))
        return
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append(("You", cleaned))
    if _maybe_handle_recall_query(cleaned):
        return
    quote_mode = is_recall_query(cleaned)
    quote_count = estimate_quote_count(cleaned) if quote_mode else None
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
        cleaned,
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
    if not (OpenAI and OPENAI_API_KEY):
        return 0.0
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
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


def _augment_prompt(user_question: str, *, attachments: list[dict] | None = None) -> str:
    entity = _get_entity()
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    ledger_id = st.session_state.get("ledger_id")
    since, until = derive_time_filters(user_question)
    history = st.session_state.get("chat_history") or []
    return PROMPT_SERVICE.build_augmented_prompt(
        entity=entity,
        question=user_question,
        schema=schema,
        chat_history=history,
        ledger_id=ledger_id,
        attachments=attachments or [],
        since=since,
        until=until,
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


def _let_llm_structure_memory(text: str) -> list[dict] | None:
    """Prompt the LLM to extract key concepts as prime factors."""
    if not (OpenAI and OPENAI_API_KEY):
        st.warning("OpenAI API key missing for memory structuring.")
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
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content)
        if "factors" in payload and isinstance(payload["factors"], list):
            return payload["factors"]
    except Exception as e:
        st.error(f"Failed to structure memory with LLM: {e}")
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
        factors = _let_llm_structure_memory(full_block)
        if _anchor(full_block, record_chat=False, factors_override=factors):
            st.session_state.rolling_text = []
            st.session_state.last_anchor_ts = time.time()


def _maybe_handle_recall_query(text: str) -> bool:
    """Check for recall triggers and reply with ledger content if matched."""
    if not is_recall_query(text):
        return False

    entity = _get_entity()
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    ledger_id = st.session_state.get("ledger_id")
    response = MEMORY_SERVICE.build_recall_response(
        entity,
        text,
        schema,
        ledger_id=ledger_id,
    )
    if response:
        st.session_state.chat_history.append(("Bot", response))
    else:
        st.session_state.chat_history.append(("Bot", "I couldn't find any matching memories in the ledger."))
    return True


def _anchor(text: str, *, record_chat: bool = True, notify: bool = True, factors_override: list[dict] | None = None):
    entity = _get_entity()
    if not entity:
        st.error("No active entity; cannot anchor.")
        return False

    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    try:
        ingest_result = PRIME_SERVICE.ingest(
            entity,
            text,
            schema,
            ledger_id=st.session_state.get("ledger_id"),
            factors_override=factors_override,
            llm_extractor=_llm_factor_extractor,
            metadata={"source": "chat_demo"},
        )
    except requests.RequestException as exc:
        st.session_state.last_anchor_error = str(exc)
        _refresh_capabilities_block()
        st.error(f"Anchor failed: {exc}")
        return False

    st.session_state.last_anchor_error = None
    _refresh_capabilities_block()
    structured = ingest_result.get("structured") if isinstance(ingest_result, dict) else {}
    ledger_id = st.session_state.get("ledger_id")
    if structured:
        persisted = _persist_structured_views(entity, structured, ledger_id=ledger_id)
        MEMORY_SERVICE.update_structured_ledger(entity, persisted, ledger_id=ledger_id)
        st.session_state.latest_structured_ledger = persisted
    else:
        _persist_structured_views_from_ledger(entity)
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
    llm_prompt = _augment_prompt(prompt, attachments=attachments)

    if use_openai:
        if not (OpenAI and OPENAI_API_KEY):
            st.warning("OpenAI API key missing.")
            return
        client = OpenAI(api_key=OPENAI_API_KEY)
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

    if not (genai and GENAI_KEY):
        return "Gemini API key missing."
    model = genai.GenerativeModel("gemini-2.0-flash")
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
    if "capabilities_block" not in st.session_state:
        _refresh_capabilities_block()

    if not st.session_state.get("authenticated"):
        _render_login_form()
        return

    _maybe_handle_demo_mode()
    _ensure_ledger_bootstrap()

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
        openai_ready=bool(OpenAI and OPENAI_API_KEY),
        gemini_ready=bool(genai and GENAI_KEY),
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

    st.sidebar.subheader("Ledger routing")
    if st.sidebar.button("Refresh ledgers", key="refresh_ledgers_btn"):
        _refresh_ledgers()
    raw_ledgers = st.session_state.get("ledgers", [])
    ledger_options: list[str] = []
    for entry in raw_ledgers:
        lid = entry.get("ledger_id")
        if lid and lid not in ledger_options:
            ledger_options.append(lid)
    active_ledger = st.session_state.get("ledger_id") or DEFAULT_LEDGER_ID
    if active_ledger and active_ledger not in ledger_options:
        ledger_options.insert(0, active_ledger)
    available_options = list(ledger_options)
    available_options.append(ADD_LEDGER_OPTION)
    initial_index = available_options.index(active_ledger) if active_ledger in available_options else 0
    selection = st.sidebar.selectbox(
        "Active ledger",
        available_options,
        index=initial_index,
        help="All API calls send X-Ledger-ID so memories stay scoped per tenant.",
    )
    if selection == ADD_LEDGER_OPTION:
        st.sidebar.caption("Rules: 3-32 chars, lowercase letters/digits, hyphens allowed in the middle.")
        new_ledger = st.sidebar.text_input("New ledger ID", placeholder="team-alpha", key="new_ledger_id")
        if st.sidebar.button("Create ledger", key="create_ledger_btn"):
            valid, error = _validate_ledger_name(new_ledger)
            if not valid:
                st.sidebar.error(error)
            elif _create_or_switch_ledger(new_ledger):
                _refresh_ledgers(silent=True)
                _trigger_rerun()
    elif selection != active_ledger:
        if _create_or_switch_ledger(selection):
            _refresh_ledgers(silent=True)

    if st.session_state.get("ledgers"):
        st.sidebar.caption("Ledger directories:")
        for entry in st.session_state["ledgers"]:
            ledger_id = entry.get("ledger_id")
            path = entry.get("path") or "—"
            st.sidebar.caption(f"• {ledger_id}: {path}")
    else:
        st.sidebar.info("No ledgers detected yet — choose “Add new ledger…” to create one.")

    render_llm_selector(
        openai_ready=bool(OpenAI and OPENAI_API_KEY),
        gemini_ready=bool(genai and GENAI_KEY),
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
                if not (OpenAI and OPENAI_API_KEY):
                    st.warning("OpenAI API key missing.")
                else:
                    client = OpenAI(api_key=OPENAI_API_KEY)
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

    tab_chat, tab_about = st.tabs(["Chat", "About DualSubstrate"])

    with tab_chat:
        recent_history = list(reversed(st.session_state.chat_history[-20:]))
        if st.session_state.pending_attachments:
            for attachment in st.session_state.pending_attachments:
                preview = (attachment.get("text") or "").strip()
                summary = preview[:200].replace("\n", " ")
                if len(preview) > 200:
                    summary += "…"
                st.info(f"Attachment ready: {attachment['name']} – {summary}")
        if recent_history:
            entries = [
                f"<div class='chat-entry'><strong>{html.escape(role)}:</strong> {html.escape(content)}</div>"
                for role, content in recent_history
            ]
            stream_html = "<hr>".join(entries)
        else:
            stream_html = "<div class='chat-entry'>No chat history yet.</div>"
        st.markdown(f"<div class='chat-stream'>{stream_html}</div>", unsafe_allow_html=True)
        st.markdown("<hr class='full-divider'>", unsafe_allow_html=True)

    with tab_about:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(
                """

                <div class="about-col about-col-left">
                    <h2 class="about-heading" style="font-size: 01.2rem; font-weight: 400">DualSubstrate ledger demo</h2>
                    <p class="about-text">To test this DualSubstrate ledger demo speak or type. Everything anchors to the prime-based ledger. Tip: type /q or “what did I say at 7 pm” and I’ll quote you word-for-word from the prime-ledger. Anything else = normal chat.</p>
                    <hr>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="prime-ledger-block">
                    <h2 class="prime-heading" style="font-size: 1.2rem; font-weight: 400">Prime-Ledger Snapshot</h2>
                    <p class="prime-text">A live, word-perfect copy of everything you’ve anchored - sealed in primes, mathematically identical forever.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load ledger", key="load_ledger_about"):
                _load_ledger()
            if st.session_state.ledger_state:
                _render_ledger_state(st.session_state.ledger_state)
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
                st.metric("Durability h", f"{durability_h:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("### Möbius lattice rotation")
            if st.button("♾️ Möbius Transform", help="Reproject the exponent lattice"):
                entity = _get_entity()
                if not entity:
                    st.warning("No active entity.")
                    return
                try:
                    data = perform_lattice_rotation(
                        API_SERVICE,
                        entity,
                        ledger_id=st.session_state.get("ledger_id"),
                        axis=(0.0, 0.0, 1.0),
                        angle=1.0472,
                    )
                    st.success(
                        f"Rotated lattice. Δenergy = {data.get('energy_cycles')}, "
                        f"checksum {data.get('original_checksum')} → {data.get('rotated_checksum')}."
                    )
                    _load_ledger()
                    if st.session_state.ledger_state:
                        st.caption("Updated ledger snapshot after Möbius transform:")
                        _render_ledger_state(st.session_state.ledger_state)
                    _trigger_rerun()
                except requests.RequestException as exc:
                    st.error(f"Möbius rotation failed: {exc}")
            if st.button("Initiate Enrichment", help="Replay stored transcripts with richer prime coverage"):
                with st.spinner("Enriching memories…"):
                    entity = _get_entity()
                    if not entity:
                        st.warning("No active entity.")
                    else:
                        schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
                        result = run_enrichment_job(
                            API_SERVICE,
                            PRIME_SERVICE,
                            entity,
                            ledger_id=st.session_state.get("ledger_id"),
                            schema=schema,
                            llm_extractor=_llm_factor_extractor,
                        )
                        if result.get("error"):
                            st.error(result["error"])
                        elif result.get("message"):
                            st.info(result["message"])
                        else:
                            st.success(f"Enriched {result.get('enriched', 0)}/{result.get('total', 0)} memories.")
                            if result.get("failures"):
                                st.warning("Some entries failed: " + "; ".join(result["failures"]))
                        _refresh_capabilities_block()


def main() -> None:
    """Streamlit entry point for the public chat demo."""

    _render_app()
    if not st.session_state.get("capabilities_block"):
        _refresh_capabilities_block()


if __name__ == "__main__":
    main()
