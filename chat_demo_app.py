import audioop
import base64
import hashlib
import html
import io
import json
import mimetypes
import os
import re
import time
import wave
from datetime import datetime, timedelta

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
    import parsedatetime as pdt
except ModuleNotFoundError:
    pdt = None
try:
    from pypdf import PdfReader
except ModuleNotFoundError:
    PdfReader = None
try:
    import dateparser
except ModuleNotFoundError:
    dateparser = None

from app_settings import DEFAULT_METRIC_FLOORS, load_settings
from services.api import ApiService, requests
from services.memory_resolver import MemoryResolver
from validators import validate_prime_sequence, get_tier_value
from prime_pipeline import (
    build_anchor_batches,
    call_factor_extraction_llm,
    map_to_primes,
    normalize_override_factors,
)
from prime_tagger import tag_primes

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
    quote_mode = _is_quote_request(cleaned)
    quote_count = _estimate_quote_count(cleaned) if quote_mode else None
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





def _encode_prime_signature(text: str) -> dict[int, float]:
    signature: dict[int, float] = {}
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    primes = tag_primes(text, schema)
    factors = [{"prime": p, "delta": 1} for p in primes]
    for factor in factors:
        prime = factor.get("prime")
        delta = factor.get("delta", 0)
        if not isinstance(prime, int):
            continue
        weight = PRIME_WEIGHTS.get(prime, 1.0)
        signature[prime] = signature.get(prime, 0.0) + float(delta) * weight
    return signature


def _prime_topological_distance(query_sig: dict[int, float], memory_sig: dict[int, float]) -> float:
    if not query_sig and not memory_sig:
        return 0.0
    distance = 0.0
    all_primes = set(query_sig.keys()) | set(memory_sig.keys())
    for prime in all_primes:
        query_val = query_sig.get(prime, 0.0)
        memory_val = memory_sig.get(prime, 0.0)
        weight = PRIME_WEIGHTS.get(prime, 1.0)
        if query_val > 0 and memory_val == 0:
            distance += abs(query_val) * weight * 2.0
        else:
            distance += abs(query_val - memory_val) * weight
    return distance


def _is_user_content(text: str) -> bool:
    """Check if a memory entry appears to be user-authored content."""
    normalized = (text or "").strip().lower()
    if len(normalized) < 20:
        return False

    # Stricter check for bot-like prefixes.
    bot_prefixes = (
        "bot:",
        "assistant:",
        "system:",
        "model:",
        "ai:",
        "llm:",
        "response:",
        "here’s what the ledger currently recalls:",
        "[",
    )
    if normalized.startswith(bot_prefixes):
        return False

    # More specific check for ledger factor lines.
    if "• ledger" in normalized and "ledger factor" in normalized:
        return False

    # Avoid filtering just because the word "ledger" is present.
    # The original check was too broad.
    return True


def _select_lawful_context(
    query: str,
    *,
    limit: int = 5,
    time_window_hours: int = 72,
    since: int | None = None,
    until: int | None = None,
) -> list[dict]:
    """Select relevant memories based on keyword matching and recency."""
    entity = _get_entity()
    if not entity:
        return []

    keywords = _keywords_from_prompt(query)
    fetch_limit = min(100, max(limit * 5, 25))
    window_start = since
    if window_start is None:
        window_start = int((time.time() - time_window_hours * 3600) * 1000)

    raw_memories = _memory_lookup(limit=fetch_limit, since=window_start)
    if until is not None:
        raw_memories = [m for m in raw_memories if m.get("timestamp", 0) <= until]

    scored_memories = []
    now = time.time()
    for entry in raw_memories:
        text = _strip_ledger_noise(entry.get("text", ""))
        if not text or not _is_user_content(text):
            continue

        score = 0
        for keyword in keywords:
            score += text.lower().count(keyword.lower())

        timestamp = entry.get("timestamp", 0)
        age_hours = (now - timestamp / 1000) / 3600
        score += max(0, 1 - age_hours / (time_window_hours * 2))  # Recency boost

        entry["_sanitized_text"] = text
        scored_memories.append({"entry": entry, "score": score})

    scored_memories.sort(key=lambda x: x["score"], reverse=True)
    return [item["entry"] for item in scored_memories[:limit]]


TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?\s?(?:am|pm)?)\b", re.IGNORECASE)
RELATIVE_NUMBER_PATTERN = re.compile(r"\b(\d+)\s+(minute|hour|day|week)s?\s+ago\b", re.IGNORECASE)
RELATIVE_ARTICLE_PATTERN = re.compile(r"\b(an|a)\s+(minute|hour|day|week)\s+ago\b", re.IGNORECASE)
LAST_RANGE_PATTERN = re.compile(r"\blast\s+(\d+)\s+(minute|hour|day|week)s?\b", re.IGNORECASE)
PAST_RANGE_PATTERN = re.compile(r"\bpast\s+(\d+)\s+(minute|hour|day|week)s?\b", re.IGNORECASE)
TIME_RANGE_PATTERN = re.compile(
    r"(?:yesterday|today)\s+between\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*-\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    re.IGNORECASE,
)
CAL = pdt.Calendar() if pdt else None
QUOTE_KEYWORD_PATTERN = re.compile(r"\b(quote|verbatim|exact)\b", re.I)
RECALL_KEYWORD_PATTERN = re.compile(r"\b(recall|retrieve|topics?|covered|definitions?)\b", re.I)
RECALL_PHRASES = (
    "what did i say",
    "what did we talk about",
    "did we talk about",
    "what did we discuss",
    "did we discuss",
    "what did we cover",
    "did we cover",
    "what topics did we cover",
    "what topics",
    "last few days",
    "last 24 hours",
    "last 48 hours",
    "past day",
    "past few days",
    "definitions of god",
    "definitions of gods",
    "god definitions",
)
RELATIVE_WORD_OFFSETS = {
    "yesterday": 24 * 3600,
    "last night": 12 * 3600,
    "earlier today": 6 * 3600,
    "this morning": 6 * 3600,
    "this afternoon": 6 * 3600,
    "this evening": 6 * 3600,
    "an hour ago": 3600,
    "an hour earlier": 3600,
    "last week": 7 * 24 * 3600,
}
UNIT_TO_SECONDS = {"minute": 60, "hour": 3600, "day": 86400, "week": 7 * 86400}
PREFIXES = ("/q", "@ledger", "::memory")
_DIGIT_PATTERN = re.compile(r"\b\d+\b")
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_QUANTITY_HINTS = {
    "a couple": 2,
    "couple": 2,
    "a few": 3,
    "few": 3,
    "handful": 4,
    "some": 4,
    "several": 6,
    "many": 8,
    "plenty": 8,
    "all": 15,
    "entire": 15,
}
_STOPWORDS = {
    "about",
    "ago",
    "been",
    "could",
    "does",
    "days",
    "discuss",
    "discussed",
    "discussion",
    "document",
    "explain",
    "explanation",
    "excerpts",
    "few",
    "from",
    "give",
    "have",
    "hours",
    "information",
    "kindly",
    "last",
    "ledger",
    "long",
    "longer",
    "memory",
    "memories",
    "more",
    "over",
    "paper",
    "please",
    "provide",
    "quote",
    "quotes",
    "said",
    "say",
    "says",
    "some",
    "talk",
    "talked",
    "talking",
    "tell",
    "than",
    "that",
    "this",
    "today",
    "topic",
    "topics",
    "verbatim",
    "verbatims",
    "what",
    "which",
    "with",
    "would",
    "yesterday",
}

_RECALL_SKIP_PREFIXES = (
    "what information have we been discussing",
    "what information do we have",
    "what did we talk about",
    "can you provide",
    "what topics did we cover",
    "what information have we been",
)


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


def _memory_lookup(limit: int = 3, since: int | None = None):
    entity = _get_entity()
    if not entity:
        return []
    ledger_id = st.session_state.get("ledger_id")
    try:
        data = API_SERVICE.fetch_memories(
            entity,
            ledger_id=ledger_id,
            limit=limit,
            since=since,
        )
        if any(isinstance(item, dict) and item.get("text") for item in data):
            print("[DEBUG] /memories returned:", data[:3])
            return data
        if data:
            print("[DEBUG] /memories unexpected payload:", data)
    except requests.RequestException as exc:
        print("[DEBUG] /memories error:", exc)

    try:
        ledger_payload = API_SERVICE.fetch_ledger(entity, ledger_id=ledger_id)
    except requests.RequestException as exc:
        print("[DEBUG] /ledger fallback failed:", exc)
        return []

    factors = []
    if isinstance(ledger_payload, dict):
        factors = ledger_payload.get("factors") or []

    now_ms = int(time.time() * 1000)
    synthetic: list[dict] = []
    for entry in factors:
        if len(synthetic) >= max(1, limit):
            break
        if not isinstance(entry, dict):
            continue
        prime = entry.get("prime")
        value = entry.get("value", 0)
        if prime in PRIME_ARRAY and value:
            synthetic.append(
                {
                    "timestamp": now_ms,
                    "text": f"(Ledger factor) Prime {prime} = {value}",
                    "meta": {"source": "ledger"},
                }
            )

    if synthetic:
        print("[DEBUG] Fallback memories constructed:", synthetic[:3])
    return synthetic


def _render_memories(entries):
    if not entries:
        st.session_state.chat_history.append(("Memory", "Ledger recall: no matching memories."))
        return
    for entry in entries:
        stamp = entry.get("timestamp")
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stamp / 1000)) if stamp else "unknown time"
        text = entry.get("text", "(no text)")
        msg = f"{ts} — {text}"
        st.session_state.chat_history.append(("Memory", msg))


def _is_quote_request(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.startswith(PREFIXES) or QUOTE_KEYWORD_PATTERN.search(normalized) is not None


def _extract_requested_count(text: str) -> int | None:
    if not text:
        return None

    digits = [int(match) for match in _DIGIT_PATTERN.findall(text)]
    if digits:
        return digits[-1]

    lowered = text.lower()
    for phrase, value in _QUANTITY_HINTS.items():
        if phrase in lowered:
            return value

    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return value

    return None


def _estimate_quote_count(text: str) -> int:
    explicit = _extract_requested_count(text)
    if explicit:
        return max(1, min(explicit, 25))

    lowered = (text or "").lower()
    for phrase, value in _QUANTITY_HINTS.items():
        if phrase in lowered:
            return max(1, min(value, 25))

    return 5


QUOTE_LIST_PATTERN = re.compile(r"^\s*\d{1,2}[).:-]\s+")
BULLET_QUOTE_PATTERN = re.compile(r"^\s*[-\u2022]\s+")


def _strip_ledger_noise(text: str, *, user_only: bool = False) -> str:
    """Strip bot-generated prefixes and other machine noise from memory text."""
    if not text:
        return text

    # Keep only lines that appear to be from a human user.
    # This is more targeted than the original implementation.
    clean_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            continue
        if lowered.startswith("bot:") or "• ledger" in lowered:
            continue
        if len(stripped) > 20:
            clean_lines.append(stripped)

    return "\n".join(clean_lines)


def _build_lawful_augmentation_prompt(
    user_question: str,
    *,
    attachments: list[dict] | None = None,
    since: int | None = None,
    until: int | None = None,
) -> str:
    """Build a rich prompt that unshackles the LLM to use its full context window."""
    context_memories = _select_lawful_context(
        user_question,
        limit=10,  # Provide more memories for richer context.
        time_window_hours=168,  # Extend to a full week.
        since=since,
        until=until,
    )

    # Rephrase the prompt to be less restrictive and more empowering.
    prompt_lines = [
        "You are a helpful assistant with access to a perfect, exact memory ledger.",
        "Your goal is to provide the most relevant, insightful, and natural response.",
        "Use the provided ledger memories and conversation history to understand the full context.",
        "You are free to synthesize information, draw conclusions, and ask clarifying questions.",
        "Your response should be helpful and conversational, not a rigid report.",
    ]

    if context_memories:
        prompt_lines.append("\n--- Ledger Memories (most relevant first) ---")
        for entry in context_memories:
            sanitized = entry.get("_sanitized_text") or _strip_ledger_noise((entry.get("text") or "").strip())
            if not sanitized:
                continue
            stamp = entry.get("timestamp")
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp / 1000)) if stamp else "No timestamp"
            prompt_lines.append(f"[{ts}] {sanitized}")
    else:
        prompt_lines.append("\n--- Ledger Memories ---")
        prompt_lines.append("(No specific memories matched the query, but the full ledger is available.)")

    # Provide a much larger view of the recent conversation.
    chat_block = _recent_chat_block(max_entries=15)
    if chat_block:
        prompt_lines.append("\n--- Recent Conversation ---")
        prompt_lines.append(chat_block)

    if attachments:
        prompt_lines.append("\n--- Attachments ---")
        for attachment in attachments:
            name = attachment.get("name", "attachment")
            snippet = (attachment.get("text") or "").strip()[:1000]
            if snippet:
                prompt_lines.append(f"[{name}] {snippet}...")

    prompt_lines.append(f"\n--- Your Turn ---")
    prompt_lines.append(f"User's request: {user_question}")
    prompt_lines.append("Your response:")
    return "\n".join(prompt_lines)


def _augment_prompt(user_question: str, *, attachments: list[dict] | None = None) -> str:
    start_ms, end_ms = _parse_time_range(user_question)
    return _build_lawful_augmentation_prompt(
        user_question,
        attachments=attachments,
        since=start_ms,
        until=end_ms,
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


def _keywords_from_prompt(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{3,}", (text or "").lower())
    keywords: list[str] = []
    seen = set()
    for token in tokens:
        if token in _STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


MEMORY_RESOLVER = MemoryResolver(
    _keywords_from_prompt,
    lambda text: _strip_ledger_noise(text),
)


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


def _summarize_accessible_memories(limit: int, since: int | None = None, *, keywords: list[str] | None = None) -> str | None:
    fetch_limit = limit
    if keywords:
        fetch_limit = min(100, max(limit * 4, 20))
    entries = _memory_lookup(limit=fetch_limit, since=since)
    if not entries:
        return "Ledger currently has no stored memories yet."
    lines: list[str] = []
    normalized_keywords = [k.lower() for k in (keywords or []) if len(k) >= 3]
    for entry in entries:
        stamp = entry.get("timestamp")
        human_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stamp / 1000)) if stamp else "unknown time"
        text = _strip_ledger_noise((entry.get("text") or "").strip())
        if not text:
            continue
        lowered = text.lower()
        if lowered.startswith("(ledger reset"):
            continue
        if normalized_keywords and not any(k in lowered for k in normalized_keywords):
            continue
        snippet = text[:320].replace("\n", " ")
        if len(text) > len(snippet):
            snippet = f"{snippet}…"
        lines.append(f"{human_ts}: {snippet or '(no text)'}")
    if not lines:
        if keywords:
            focus = ", ".join(keywords[:3])
            return f"No stored memories matched the requested topic ({focus})."
        return "Ledger currently has no user-authored memories yet."
    scope = (
        f"since {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(since / 1000))}"
        if since
        else "from the latest anchors"
    )
    return f"I could not extract exact verbatim quotes for that query, but here is what I can access {scope}:\n" + "\n".join(
        lines
    )


def _memory_context_block(limit: int = 3, since: int | None = None, *, keywords: list[str] | None = None) -> str:
    fetch_limit = limit
    if keywords:
        fetch_limit = min(100, max(limit * 4, 20))
    entries = _memory_lookup(limit=fetch_limit, since=since)
    snippets: list[str] = []
    normalized_keywords = [k.lower() for k in (keywords or []) if len(k) >= 3]
    for entry in entries:
        text = _strip_ledger_noise((entry.get("text") or "").strip())
        if not text:
            continue
        lowered = text.lower()
        if lowered.startswith("(ledger reset before enrichment"):
            continue
        if normalized_keywords and not any(k in lowered for k in normalized_keywords):
            continue
        source = entry.get("meta", {}).get("source") or entry.get("name") or entry.get("attachment") or ""
        snippet = text[:240].replace("\n", " ")
        if len(text) > len(snippet):
            snippet += "…"
        label = f" ({source})" if source else ""
        snippets.append(f"- {snippet}{label}")
    if not snippets and keywords:
        focus = ", ".join(keywords[:3])
        return f"- No ledger memories matched the topic ({focus})."
    return "\n".join(snippets)


def _recent_chat_block(max_entries: int = 15) -> str | None:
    """Format the recent chat history into a string for the LLM context."""
    history = st.session_state.get("chat_history") or []
    if not history:
        return ""
    lines: list[str] = []
    for role, content in history[-max_entries:]:
        # Include all roles for a more complete picture of the conversation.
        snippet = (content or "").strip().replace("\n", " ")
        if not snippet:
            continue
        if len(snippet) > 300:
            snippet = f"{snippet[:300]}…"
        lines.append(f"{role}: {snippet}")
    return "\n".join(lines) if lines else None


def _build_capabilities_block() -> str:
    last_error = st.session_state.get("last_anchor_error")
    anchor_note = (
        "Latest anchor failed; new text may not be stored."
        if last_error
        else "Latest anchor succeeded."
    )
    lines = [
        "Capabilities & Instructions:",
        "- Cite only the memories listed below; do not invent new quotes.",
        "- If the ledger has no entry for the requested topic, say so explicitly.",
        "- Anchors succeed only when the ledger accepts the factors; report failures if they occur.",
        "",
        anchor_note,
        "",
        _prime_semantics_block(),
    ]
    ledger = _memory_context_block(limit=5)
    if ledger:
        lines.extend(["", "Recent ledger memories:", ledger])
    recent_chat = _recent_chat_block()
    if recent_chat:
        lines.extend(["", "Recent chat summary:", recent_chat])
    return "\n".join(lines)


def _parse_time_range(text: str) -> tuple[int | None, int | None]:
    if not text:
        return None, None
    match = TIME_RANGE_PATTERN.search(text)
    if not match:
        return None, None

    lowered = text.lower()
    now = datetime.now()
    base_date = now - timedelta(days=1) if "yesterday" in lowered else now

    try:
        start_hour = int(match.group(1))
        start_minute = int(match.group(2) or 0)
        start_ampm = (match.group(3) or "").lower()
        end_hour = int(match.group(4))
        end_minute = int(match.group(5) or 0)
        end_ampm = (match.group(6) or "").lower()

        if not start_ampm and end_ampm:
            start_ampm = end_ampm
        if start_ampm == "pm" and start_hour != 12:
            start_hour += 12
        if start_ampm == "am" and start_hour == 12:
            start_hour = 0
        if not end_ampm and start_ampm:
            end_ampm = start_ampm
        if end_ampm == "pm" and end_hour != 12:
            end_hour += 12
        if end_ampm == "am" and end_hour == 12:
            end_hour = 0

        start_dt = base_date.replace(hour=start_hour % 24, minute=start_minute, second=0, microsecond=0)
        end_dt = base_date.replace(hour=end_hour % 24, minute=end_minute, second=0, microsecond=0)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)
    except Exception:
        return None, None


def _infer_relative_timestamp(text: str) -> int | None:
    if not text:
        return None
    lowered = text.lower()
    now = time.time()
    for phrase, seconds in RELATIVE_WORD_OFFSETS.items():
        if phrase in lowered:
            return int((now - seconds) * 1000)
    match = RELATIVE_NUMBER_PATTERN.search(lowered)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower().rstrip("s")
        seconds = UNIT_TO_SECONDS.get(unit)
        if seconds:
            return int((now - amount * seconds) * 1000)
    match = RELATIVE_ARTICLE_PATTERN.search(lowered)
    if match:
        unit = match.group(2).lower()
        seconds = UNIT_TO_SECONDS.get(unit)
        if seconds:
            return int((now - seconds) * 1000)
    match = LAST_RANGE_PATTERN.search(lowered) or PAST_RANGE_PATTERN.search(lowered)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower().rstrip("s")
        seconds = UNIT_TO_SECONDS.get(unit)
        if seconds:
            return int((now - amount * seconds) * 1000)
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
        schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
        factors_override = map_to_primes(
            chunk,
            schema,
            fallback_prime=FALLBACK_PRIME,
            llm_extractor=_llm_factor_extractor,
        )
        if _anchor(payload, record_chat=False, notify=False, factors_override=factors_override):
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
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _reset_entity_factors() -> bool:
    entity = _get_entity()
    if not entity:
        return False
    try:
        data = API_SERVICE.fetch_ledger(entity, ledger_id=st.session_state.get("ledger_id"))
    except requests.RequestException as exc:
        st.warning(f"Could not fetch ledger for reset: {exc}")
        return False

    factors = data.get("factors") if isinstance(data, dict) else None
    if not isinstance(factors, list):
        return False
    reset_deltas: list[dict] = []
    for entry in factors:
        prime = entry.get("prime")
        value = entry.get("value", 0)
        if prime in PRIME_ARRAY and value:
            try:
                reset_deltas.append({"prime": int(prime), "delta": -int(value)})
            except (TypeError, ValueError):
                continue
    if not reset_deltas:
        return True
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    for seq in build_anchor_batches(
        "",
        schema,
        fallback_prime=FALLBACK_PRIME,
        factors_override=reset_deltas,
        llm_extractor=None,
    ):
        try:
            API_SERVICE.anchor(entity, seq, ledger_id=st.session_state.get("ledger_id"))
        except requests.RequestException as exc:
            st.warning(f"Ledger reset failed: {exc}")
            return False
    return True


def _run_enrichment(limit: int = 200, reset_first: bool = True):
    entity = _get_entity()
    if not entity:
        return
    fetch_limit = max(1, min(limit, 100))
    try:
        memories = API_SERVICE.fetch_memories(
            entity,
            ledger_id=st.session_state.get("ledger_id"),
            limit=fetch_limit,
        )
    except requests.RequestException as exc:
        st.error(f"Failed to load memories: {exc}")
        return
    if not memories:
        st.info("No memories found to enrich.")
        return

    if reset_first:
        ok = _reset_entity_factors()
        if not ok:
            st.error("Reset failed – aborting enrichment.")
            return

    enriched = 0
    total = len(memories)
    for entry in memories:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
        batches = build_anchor_batches(
            text,
            schema,
            fallback_prime=FALLBACK_PRIME,
            llm_extractor=_llm_factor_extractor,
        )
        success = True
        for idx_batch, seq in enumerate(batches):
            try:
                API_SERVICE.anchor(
                    entity,
                    seq,
                    ledger_id=st.session_state.get("ledger_id"),
                    text=text if idx_batch == 0 else None,
                )
            except requests.RequestException as exc:
                st.warning(f"Skip enrichment for {entry.get('timestamp')}: {exc}")
                success = False
                break
        if success:
            enriched += 1
    st.success(f"Enriched {enriched}/{total} memories.")
    st.session_state.capabilities_block = _build_capabilities_block()


def _latest_user_transcript(current_request: str, *, limit: int = 5) -> str | None:
    start_ms, end_ms = _parse_time_range(current_request)
    context_candidates = _select_lawful_context(
        current_request,
        limit=max(1, limit),
        time_window_hours=48,
        since=start_ms,
        until=end_ms,
    )
    for entry in context_candidates:
        sanitized = _strip_ledger_noise(
            entry.get("_sanitized_text") or entry.get("text", ""),
            user_only=True,
        )
        if sanitized:
            return sanitized

    entries = _memory_lookup(limit=max(1, limit * 2))
    if not entries:
        return None

    user_entries: list[tuple[int, str]] = []
    for entry in entries:
        text = entry.get("text", "")
        if not text:
            continue
        normalized = text.lower()
        if any(
            marker in normalized
            for marker in (
                "ten exact quotes",
                "bot:",
                "assistant:",
                "ledger recall:",
                "no stored memories matched",
                "exact quotes:",
                "transcript:",
            )
        ):
            continue
        sanitized = _strip_ledger_noise(text, user_only=True)
        if not sanitized:
            continue
        lowered = sanitized.lower()
        tokens = lowered.split()
        if not tokens:
            continue
        if any(token in tokens[:10] for token in ["i", "we", "our", "what", "how", "why"]):
            timestamp = entry.get("timestamp") or 0
            user_entries.append((int(timestamp), sanitized))

    if not user_entries:
        return None

    user_entries.sort(key=lambda item: item[0], reverse=True)
    return user_entries[0][1]


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
    normalized = text.strip().lower()
    # Simplified trigger: check for keywords and phrases.
    # This is more direct than the original scoring system.
    triggers = [
        normalized.startswith(PREFIXES),
        any(k in normalized for k in ["quote", "verbatim", "exact", "recall", "retrieve"]),
        any(p in normalized for p in RECALL_PHRASES),
    ]
    if not any(triggers):
        return False

    limit = _estimate_quote_count(normalized)
    since_ms, until_ms = _parse_time_range(normalized)
    if since_ms is None:
        since_ms = _infer_relative_timestamp(normalized)

    # Use the more reliable lawful context selection.
    memories = _select_lawful_context(text, limit=limit, since=since_ms, until=until_ms)
    if not memories:
        st.session_state.chat_history.append(("Bot", "I couldn't find any matching memories in the ledger."))
        return True

    # Format the response.
    response_lines = ["Here’s what the ledger currently recalls:"]
    for entry in memories:
        timestamp = entry.get("timestamp")
        date_str = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M")
        text_content = entry.get("_sanitized_text", entry.get("text", "")).strip()
        response_lines.append(f"[{date_str} • ledger] {text_content}")

    st.session_state.chat_history.append(("Bot", "\n".join(response_lines)))
    return True


def _anchor(text: str, *, record_chat: bool = True, notify: bool = True, factors_override: list[dict] | None = None):
    entity = _get_entity()
    if not entity:
        st.error("No active entity; cannot anchor.")
        return False

    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    batches = build_anchor_batches(
        text,
        schema,
        fallback_prime=FALLBACK_PRIME,
        factors_override=factors_override,
        llm_extractor=_llm_factor_extractor,
    )
    if not batches:
        st.error("Anchor failed: no lawful factor batches could be generated.")
        st.session_state.last_anchor_error = "No lawful batches"
        return False

    for i, factors in enumerate(batches):
        if not validate_prime_sequence(factors, schema):
            st.error(f"Anchor failed: Invalid prime sequence for tier batch {i}.")
            st.session_state.last_anchor_error = "Invalid prime sequence"
            continue

        try:
            API_SERVICE.anchor(
                entity,
                factors,
                ledger_id=st.session_state.get("ledger_id"),
                text=text if i == 0 else None,
            )
        except requests.RequestException as exc:
            st.session_state.last_anchor_error = str(exc)
            st.session_state.capabilities_block = _build_capabilities_block()
            st.error(f"Anchor failed: {exc}")
            return False

    st.session_state.last_anchor_error = None
    st.session_state.capabilities_block = _build_capabilities_block()
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
    if _reset_entity_factors():
        st.toast("Demo mode: Ledger has been reset.", icon="✅")
    else:
        st.toast("Demo mode: Ledger reset failed.", icon="⚠️")
    _reset_chat_state(clear_query=True)


def _get_digest(key: str, fallback: str | None = None) -> str | None:
    return _secret(key) or os.getenv(key) or fallback


DEMO_USERS = {
    "Developer": {
        "entity": "Demo_dev",
        "digest": _get_digest("DEMO_DEV_DIGEST", "92bfc5adfab78a72"),
    },
    "Demo user": {
        "entity": "Demo_new",
        "digest": _get_digest("DEMO_NEW_DIGEST", "6f6850b5f086fb49"),
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
        if _reset_entity_factors():
            st.toast("New demo session started. Ledger has been reset.", icon="✅")
        else:
            st.toast("Ledger reset failed.", icon="⚠️")
        st.session_state.login_time = time.time()
    st.session_state.prime_schema = _fetch_prime_schema(user_data["entity"])
    st.session_state.prime_symbols = {
        prime: meta.get("name", f"Prime {prime}") for prime, meta in st.session_state.prime_schema.items()
    }
    st.session_state.capabilities_block = _build_capabilities_block()


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
        st.session_state.capabilities_block = _build_capabilities_block()

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
        raw = _memory_lookup(limit=20)
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
        _process_memory_text(prompt_top, use_openai=True, attachments=attachments)
        st.session_state.pending_attachments = []
        # Streamlit clears chat inputs automatically after submission, so avoid
        # writing to the widget-managed key here to prevent SessionState errors.

    # ---------- investor KPI ----------
    entity = _get_entity()
    if entity:
        try:
            metrics = API_SERVICE.fetch_metrics(ledger_id=st.session_state.get("ledger_id"))
        except requests.RequestException:
            metrics = {"tokens_deduped": "N/A", "ledger_integrity": 0.0}

        try:
            memories = API_SERVICE.fetch_memories(
                entity,
                ledger_id=st.session_state.get("ledger_id"),
                limit=1,
            )
        except requests.RequestException:
            memories = []
    else:
        metrics = {"tokens_deduped": "N/A", "ledger_integrity": 0.0}
        memories = []

    oldest = (
        memories[-1].get("timestamp", time.time() * 1000) / 1000
        if isinstance(memories, list) and memories
        else time.time()
    )
    durability_h = (time.time() - oldest) / 3600
    durability_h = max(durability_h, METRIC_FLOORS["durability_h"])

    tokens_saved_value = _coerce_float(metrics.get("tokens_deduped"))
    if tokens_saved_value is None:
        tokens_saved_value = METRIC_FLOORS["tokens_deduped"]
    else:
        tokens_saved_value = max(tokens_saved_value, METRIC_FLOORS["tokens_deduped"])
    tokens_saved = f"{int(tokens_saved_value):,}"

    ledger_integrity_value = _coerce_float(metrics.get("ledger_integrity"))
    if ledger_integrity_value is None:
        ledger_integrity_value = METRIC_FLOORS["ledger_integrity"]
    elif ledger_integrity_value > 1.5:
        ledger_integrity_value = ledger_integrity_value / 100.0
    ledger_integrity = max(ledger_integrity_value, METRIC_FLOORS["ledger_integrity"])

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
                    data = API_SERVICE.rotate(
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
                    _run_enrichment()


if __name__ == "__main__":
    _render_app()
    capabilities_block = st.session_state.get("capabilities_block")
    capabilities_block = st.session_state.get("capabilities_block") or _build_capabilities_block()
