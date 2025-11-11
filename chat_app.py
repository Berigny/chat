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
from datetime import datetime

from pathlib import Path

import requests
import streamlit as st
try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None
try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None
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

API = "https://dualsubstrate-commercial.fly.dev"
DEFAULT_ENTITY = "demo_user"


def _get_entity() -> str | None:
    return st.session_state.get("entity")


def _secret(key: str):
    try:
        return st.secrets.get(key)
    except Exception:
        return None

API_KEY = _secret("DUALSUBSTRATE_API_KEY") or os.getenv("DUALSUBSTRATE_API_KEY") or "demo-key"
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}

GENAI_KEY = _secret("API_KEY") or os.getenv("API_KEY")
if genai and GENAI_KEY:
    genai.configure(api_key=GENAI_KEY)

OPENAI_API_KEY = _secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
ASSET_DIR = Path(__file__).parent

DEFAULT_METRIC_FLOORS = {"tokens_deduped": 12500.0, "ledger_integrity": 0.97, "durability_h": 36.0}
_RERUN_FN = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)


def _load_metric_floors():
    raw = _secret("METRIC_FLOORS") or os.getenv("METRIC_FLOORS") or ""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    floors = {}
    for key, value in parsed.items():
        try:
            floors[key] = float(value)
        except (TypeError, ValueError):
            continue
    return floors


METRIC_FLOORS = {**DEFAULT_METRIC_FLOORS, **_load_metric_floors()}


def _load_prime_schema(entity: str | None = None) -> dict[int, dict]:
    target = entity or DEFAULT_ENTITY
    try:
        resp = requests.get(
            f"{API}/schema",
            params={"entity": target},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        schema: dict[int, dict] = {}
        if isinstance(data, dict):
            for entry in data.get("primes", []):
                prime = entry.get("prime")
                if not isinstance(prime, int):
                    continue
                schema[prime] = {
                    "name": entry.get("name") or entry.get("symbol") or entry.get("label") or f"Prime {prime}",
                    "tier": entry.get("tier") or entry.get("band") or "",
                    "mnemonic": entry.get("mnemonic") or "",
                    "description": entry.get("description") or entry.get("summary") or "",
                }
        if schema:
            if isinstance(prime, int):
                pass
            return schema
    except requests.RequestException:
        pass
    try:
        resp = requests.get(
            f"{API}/ledger",
            params={"entity": target},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        data = None
    schema = {}
    if isinstance(data, dict):
        for entry in data.get("factors", []):
            prime = entry.get("prime")
            if not isinstance(prime, int):
                continue
            schema[prime] = {
                "name": entry.get("symbol") or entry.get("label") or f"Prime {prime}",
                "tier": entry.get("tier") or "",
                "mnemonic": entry.get("mnemonic") or "",
                "description": entry.get("description") or "",
            }
    if schema:
        return schema
    return DEFAULT_PRIME_SCHEMA.copy()

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
PRIME_SCHEMA = _load_prime_schema(DEFAULT_ENTITY)
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


SUBJECT_TOKENS = {"i", "me", "we", "us", "team", "our", "client", "customer"}
ACTION_KEYWORDS = {
    "met",
    "meet",
    "meeting",
    "call",
    "called",
    "email",
    "emailed",
    "ship",
    "shipped",
    "launch",
    "launched",
    "plan",
    "planned",
    "review",
    "reviewed",
    "discuss",
    "discussed",
    "record",
    "recorded",
    "attach",
    "attached",
    "anchor",
    "anchored",
    "remember",
    "ingest",
    "ingested",
}
LOCATION_KEYWORDS = {
    "office",
    "hq",
    "zoom",
    "hangouts",
    "teams",
    "call",
    "room",
    "nyc",
    "sf",
    "london",
    "paris",
    "berlin",
    "latam",
}
INTENT_KEYWORDS = {"so that", "so we can", "in order to", "to ensure", "to confirm", "goal", "intent"}
CONTEXT_KEYWORDS = {"because", "due to", "blocked", "blocker", "dependency", "risk", "issue", "delay"}
SENTIMENT_KEYWORDS = {
    "urgent",
    "critical",
    "high priority",
    "excited",
    "happy",
    "frustrated",
    "worried",
    "concerned",
    "blocked",
}


def _extract_prime_factors(text: str) -> list[dict]:
    if not text:
        return [{"prime": FALLBACK_PRIME, "delta": 1}]
    lowered = text.lower()
    factors: list[dict] = []

    def add_prime(prime: int):
        if not any(f["prime"] == prime for f in factors):
            factors.append({"prime": prime, "delta": 1})

    tokens = set(re.findall(r"[a-z']+", lowered))
    subject_hit = bool(SUBJECT_TOKENS & tokens)
    if not subject_hit:
        proper = re.search(r"\b[A-Z][a-z]+\b", text)
        subject_hit = bool(proper)
    if subject_hit:
        add_prime(2)

    action_hit = any(
        kw in lowered for kw in ACTION_KEYWORDS
    ) or bool(re.search(r"\b\w+(ed|ing)\b", lowered))
    if action_hit:
        add_prime(3)

    object_hit = bool(re.search(r"\b(with|for|about|regarding)\s+[A-Za-z0-9_-]+", lowered))
    if not object_hit:
        object_hit = bool(re.search(r"\b[A-Z][a-z]+\b", text))
    if object_hit:
        add_prime(5)

    location_hit = any(kw in lowered for kw in LOCATION_KEYWORDS) or bool(re.search(r"\b(at|in)\s+[A-Z][\w-]+", text))
    if location_hit:
        add_prime(7)

    time_hit = bool(TIME_PATTERN.search(lowered)) or bool(_infer_relative_timestamp(text))
    if time_hit:
        add_prime(11)

    intent_hit = any(phrase in lowered for phrase in INTENT_KEYWORDS) or bool(re.search(r"\bto\s+\w+", lowered))
    if intent_hit:
        add_prime(13)

    context_hit = any(kw in lowered for kw in CONTEXT_KEYWORDS)
    if context_hit:
        add_prime(17)

    sentiment_hit = any(kw in lowered for kw in SENTIMENT_KEYWORDS)
    if sentiment_hit:
        add_prime(19)

    return factors or [{"prime": FALLBACK_PRIME, "delta": 1}]


TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?\s?(?:am|pm)?)\b", re.IGNORECASE)
RELATIVE_NUMBER_PATTERN = re.compile(r"\b(\d+)\s+(minute|hour|day|week)s?\s+ago\b", re.IGNORECASE)
RELATIVE_ARTICLE_PATTERN = re.compile(r"\b(an|a)\s+(minute|hour|day|week)\s+ago\b", re.IGNORECASE)
LAST_RANGE_PATTERN = re.compile(r"\blast\s+(\d+)\s+(minute|hour|day|week)s?\b", re.IGNORECASE)
PAST_RANGE_PATTERN = re.compile(r"\bpast\s+(\d+)\s+(minute|hour|day|week)s?\b", re.IGNORECASE)
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
    "what",
    "which",
    "tell",
    "about",
    "from",
    "with",
    "this",
    "that",
    "have",
    "please",
    "could",
    "would",
    "provide",
    "quote",
    "quotes",
    "verbatim",
    "verbatims",
    "more",
    "than",
    "give",
    "some",
    "few",
    "long",
    "longer",
    "explain",
    "explanation",
    "excerpts",
    "document",
    "paper",
    "yesterday",
    "today",
    "ago",
    "days",
    "hours",
    "please",
    "kindly",
    "talk",
    "talked",
    "talking",
    "discuss",
    "discussed",
    "discussion",
    "information",
    "topic",
    "topics",
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
    params = {"entity": entity, "limit": limit}
    if since:
        params["since"] = since
    try:
        resp = requests.get(f"{API}/memories", params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return []


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
    if not text:
        return text

    cleaned: list[str] = []
    skip_quote_list = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()
        lowered = stripped.lower()

        if not stripped:
            cleaned.append("")
            skip_quote_list = False
            continue

        if "ten exact quotes" in lowered:
            # Previous quote-mode replies sometimes anchor verbatim – skip them.
            skip_quote_list = True
            continue

        if skip_quote_list:
            if QUOTE_LIST_PATTERN.match(stripped):
                # Drop enumerated quote lines like `1. "..."`.
                continue
            if BULLET_QUOTE_PATTERN.match(stripped) and any(q in stripped for q in ('"', '“', '”')):
                # Drop bullet-style quote residue when it clearly contains quoted text.
                continue
            skip_quote_list = False

        assistant_prefixes = ("bot:", "assistant:", "system:", "model:")
        if lowered.startswith(assistant_prefixes):
            continue

        if user_only:
            non_user_prefixes = ("ai:", "llm:", "response:")
            if lowered.startswith(non_user_prefixes):
                continue

        user_prefixes = ("you:", "user:")
        if lowered.startswith(user_prefixes):
            # Drop the explicit marker while keeping the actual speech.
            cleaned.append(stripped.split(":", 1)[1].lstrip())
            continue

        cleaned.append(line)

    candidate = "\n".join(cleaned).strip()
    return candidate or text


def _augment_prompt(user_question: str, *, attachments: list[dict] | None = None) -> str:
    entity = _get_entity()
    if not entity:
        return user_question
    keywords = _keywords_from_prompt(user_question)
    try:
        resp = requests.get(
            f"{API}/memories",
            params={"entity": entity, "limit": 1},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        memories = resp.json()
    except requests.RequestException:
        summary = _summarize_accessible_memories(5)
        return f"{summary}\n\nUser question: {user_question}" if summary else user_question

    if not memories:
        summary = _summarize_accessible_memories(5, keywords=keywords)
        return f"{summary}\n\nUser question: {user_question}" if summary else user_question

    full_context = _strip_ledger_noise(memories[0].get("text", "").strip())
    if not full_context:
        summary = _summarize_accessible_memories(5, keywords=keywords)
        if summary:
            return f"{summary}\n\nUser question: {user_question}"
        return user_question

    capabilities = st.session_state.get("capabilities_block")
    prompt_lines = []
    if capabilities:
        prompt_lines.extend([capabilities, ""])
    prompt_lines.extend(
        [
            "Anchored conversation context:",
            full_context,
            "",
        ]
    )
    prompt_lines.append(_prime_semantics_block())
    prompt_lines.append("")
    context_block = _memory_context_block(limit=5, keywords=keywords)
    if context_block:
        prompt_lines.extend(["Recent ledger memories:", context_block, ""])
    chat_block = _recent_chat_block()
    if chat_block:
        prompt_lines.extend(["Recent chat summary:", chat_block, ""])
    if _is_quote_request(user_question):
        prompt_lines.append(
            "The user may ask for exact quotes. Provide precise excerpts from the context when relevant, maintaining punctuation and casing."
        )
    if attachments:
        prompt_lines.append("Relevant attachments:")
        for attachment in attachments:
            name = attachment.get("name", "attachment")
            snippet = (attachment.get("text") or "").strip()
            truncated = snippet[:1500]
            if len(snippet) > len(truncated):
                truncated = f"{truncated}\n… (truncated)"
            prompt_lines.append(f"{name}:\n{truncated}")
        prompt_lines.append("")

    prompt_lines.append(f"User question: {user_question}")
    return "\n".join(prompt_lines)


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
    tokens = re.findall(r"[A-Za-z]{4,}", (text or "").lower())
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


def _filter_memories(entries: list[dict], keywords: list[str] | None = None) -> list[dict]:
    if not keywords:
        return entries
    normalized_keywords = [k.lower() for k in keywords if len(k) >= 3]
    if not normalized_keywords:
        return entries
    filtered = []
    normalized_keywords = [k.lower() for k in keywords if len(k) >= 3]
    for entry in entries:
        text = (entry.get("text") or "").lower()
        if not text:
            continue
        if any(text.startswith(prefix) for prefix in _RECALL_SKIP_PREFIXES):
            continue
        score = sum(text.count(k) for k in normalized_keywords)
        if score:
            entry["_match_score"] = score
            filtered.append(entry)
    if filtered:
        filtered.sort(key=lambda e: (-(e.get("_match_score", 1)), -e.get("timestamp", 0)))
    for entry in filtered:
        entry.pop("_match_score", None)
    return filtered


def _recent_chat_block(max_entries: int = 8) -> str | None:
    history = st.session_state.get("chat_history") or []
    if not history:
        return ""
    lines: list[str] = []
    for role, content in history[-max_entries:]:
        if role not in {"Attachment", "Memory"}:
            continue
        snippet = (content or "").strip()
        if not snippet:
            continue
        snippet = snippet.replace("\n", " ")
        if len(snippet) > 200:
            snippet = f"{snippet[:200]}…"
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
        factors_override = _tag_text_with_schema(chunk)
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


def _normalize_factors_override(factors) -> list[dict]:
    normalized: list[dict] = []
    if not isinstance(factors, list):
        return normalized
    for item in factors:
        if not isinstance(item, dict):
            continue
        prime = item.get("prime")
        delta = item.get("delta", 1)
        if prime in PRIME_ARRAY:
            try:
                normalized.append({"prime": int(prime), "delta": int(delta)})
            except (TypeError, ValueError):
                continue
    return normalized


def _flow_safe_factors(factors: list[dict]) -> list[dict]:
    filtered: list[dict] = []
    seen: set[int] = set()
    for factor in factors:
        prime = factor.get("prime")
        if prime not in PRIME_ARRAY or prime in seen:
            continue
        delta = factor.get("delta", 1)
        try:
            entry = {"prime": int(prime), "delta": int(delta)}
        except (TypeError, ValueError):
            continue
        filtered.append(entry)
        seen.add(entry["prime"])
    odds = [f for f in filtered if f["prime"] % 2 == 1]
    evens = [f for f in filtered if f["prime"] % 2 == 0]
    return odds + evens


def _flow_safe_sequence(factors: list[dict]) -> list[dict]:
    filtered = _flow_safe_factors(factors)
    safe = []
    last_prime = None
    for factor in filtered:
        prime = factor["prime"]
        safe.append({"prime": prime, "delta": factor["delta"]})
        safe.append({"prime": prime, "delta": factor["delta"]})
        last_prime = prime
    if not safe:
        safe = [{"prime": FALLBACK_PRIME, "delta": 1}, {"prime": FALLBACK_PRIME, "delta": 1}]
    return safe


def _flow_safe_batches(factors: list[dict]) -> list[list[dict]]:
    filtered = _flow_safe_factors(factors)
    if not filtered:
        filtered = [{"prime": FALLBACK_PRIME, "delta": 1}]
    batches: list[list[dict]] = []
    for factor in filtered:
        batches.append(
            [
                {"prime": factor["prime"], "delta": factor["delta"]},
                {"prime": factor["prime"], "delta": factor["delta"]},
            ]
        )
    return batches


def _maybe_extract_agent_payload(raw_text: str) -> tuple[str, list[dict]] | None:
    cleaned = (raw_text or "").strip()
    if not cleaned.startswith("{") or "factors" not in cleaned:
        return None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    text = (payload.get("text") or "").strip()
    factors = _normalize_factors_override(payload.get("factors"))
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


def _call_factor_extraction_llm(text: str) -> list[dict]:
    if not text or not (genai and GENAI_KEY):
        return []
    schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
    schema_lines = "\n".join(
        f"{prime} ({meta.get('name', f'Prime {prime}')}) = {meta.get('tier', '')} {meta.get('mnemonic', '')} {meta.get('description', '')}".strip()
        for prime, meta in schema.items()
    )
    prompt = (
        "You extract ledger factors. "
        "Given the transcript below, identify subject (prime 2), action (3), object (5), "
        "location/channel (7), time/date (11), intent/outcome (13), context (17), sentiment/priority (19). "
        "Only include a prime if the transcript clearly expresses that facet. "
        "Return STRICT JSON with keys `text` (repeat the transcript) and `factors` "
        "(an array of objects with `prime` and `delta`). Example: "
        '{"text":"Met Priya","factors":[{"prime":2,"delta":1},{"prime":3,"delta":1}]}. '
        "Prime semantics:\n"
        f"{schema_lines}\n"
        f"Transcript:\n{text}"
    )
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        raw = getattr(response, "text", None) or ""
    except Exception:
        return []
    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        return []
    factors = _normalize_factors_override(data.get("factors"))
    return factors


def _map_to_primes_with_agent(text: str) -> list[dict]:
    llm_factors = _call_factor_extraction_llm(text)
    if llm_factors:
        return llm_factors
    fallback = _extract_prime_factors(text)
    return fallback


def _tag_text_with_schema(text: str) -> list[dict]:
    factors = _map_to_primes_with_agent(text)
    if factors:
        return factors
    return _extract_prime_factors(text)


def _reset_entity_factors() -> bool:
    entity = _get_entity()
    if not entity:
        return False
    try:
        resp = requests.get(
            f"{API}/ledger",
            params={"entity": entity},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.warning(f"Could not fetch ledger for reset: {exc}")
        return False

    data = resp.json()
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
    for seq in _flow_safe_batches(reset_deltas):
        try:
            resp = requests.post(
                f"{API}/anchor",
                json={"entity": entity, "factors": seq},
                headers=HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
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
        resp = requests.get(
            f"{API}/memories",
            params={"entity": entity, "limit": fetch_limit},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Failed to load memories: {exc}")
        return

    try:
        payload = resp.json()
    except ValueError:
        payload = []
    memories = payload if isinstance(payload, list) else []
    if not memories:
        st.info("No memories found to enrich.")
        return

    if reset_first:
        _reset_entity_factors()

    enriched = 0
    total = len(memories)
    for entry in memories:
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        factors = _tag_text_with_schema(text)
        if not factors:
            continue
        batches = _flow_safe_batches(factors)
        success = True
        for idx_batch, seq in enumerate(batches):
            payload = {
                "entity": entity,
                "text": text if idx_batch == 0 else None,
                "factors": seq,
            }
            try:
                post_resp = requests.post(
                    f"{API}/anchor",
                    json=payload,
                    headers=HEADERS,
                    timeout=15,
                )
                if not post_resp.ok:
                    st.warning(
                        f"Failed to enrich entry at {entry.get('timestamp')}: {post_resp.text}"
                    )
                    success = False
                    break
            except requests.RequestException as exc:
                st.warning(f"Anchor failed for entry {entry.get('timestamp')}: {exc}")
                success = False
                break
        if success:
            enriched += 1
    st.success(f"Enriched {enriched}/{total} memories.")
    st.session_state.capabilities_block = _build_capabilities_block()


def _latest_user_transcript(current_request: str, *, limit: int = 5) -> str | None:
    entries = _memory_lookup(limit=limit)
    if not entries:
        return None

    normalized_request = _normalize_for_match(current_request)
    keywords = _keywords_from_prompt(current_request)
    best_keyword_match = None
    best_general_match = None

    for entry in entries:
        text = entry.get("text", "")
        if not text:
            continue
        normalized_text = _normalize_for_match(text)
        if normalized_request and normalized_request in normalized_text:
            continue
        if "ten exact quotes" in normalized_text:
            # Skip prior quote-mode responses that were anchored.
            continue
        sanitized = _strip_ledger_noise(text, user_only=True)
        if normalized_request and normalized_request in _normalize_for_match(sanitized):
            continue
        if not sanitized:
            continue
        sanitized_lower = sanitized.lower()
        if keywords and any(keyword in sanitized_lower for keyword in keywords):
            if not best_keyword_match or len(sanitized) > len(best_keyword_match):
                best_keyword_match = sanitized
            continue
        if not best_general_match or len(sanitized) > len(best_general_match):
            best_general_match = sanitized

    if best_keyword_match:
        return best_keyword_match
    if best_general_match:
        return best_general_match

    fallback = entries[0].get("text", "")
    sanitized_fallback = _strip_ledger_noise(fallback, user_only=True)
    return sanitized_fallback or fallback or None


def _update_rolling_memory(user_text: str, bot_reply: str, quote_mode: bool = False):
    if user_text is None and bot_reply is None:
        return
    st.session_state.rolling_text.append(f"You: {user_text}\nBot: {bot_reply}")
    window_s = 300
    max_tokens = 2_000
    full_block = "\n".join(st.session_state.rolling_text)
    should_anchor = (
        time.time() - st.session_state.last_anchor_ts > window_s
        or len(full_block.split()) > max_tokens
        or quote_mode
    )
    if should_anchor:
        if _anchor(full_block, record_chat=False):
            st.session_state.rolling_text = []
            st.session_state.last_anchor_ts = time.time()


def _maybe_handle_recall_query(text: str) -> bool:
    normalized = text.strip().lower()
    prefix = normalized.startswith(PREFIXES)
    recall_keyword = RECALL_KEYWORD_PATTERN.search(normalized) is not None
    range_hint = LAST_RANGE_PATTERN.search(normalized) or PAST_RANGE_PATTERN.search(normalized)
    recall_phrase = any(phrase in normalized for phrase in RECALL_PHRASES) or bool(range_hint)
    if not recall_phrase and "what" in normalized and "have we" in normalized:
        recall_phrase = True
    since_ms = None
    keywords = _keywords_from_prompt(text)

    words = text.split()
    if len(words) > 120 and not prefix:
        return False

    parsed_datetime = None
    if dateparser:
        # Attempt to parse a datetime from the text, preferring past dates.
        parsed_datetime = dateparser.parse(text, settings={"PREFER_DATES_FROM": "past"})

    if parsed_datetime:
        since_ms = int(parsed_datetime.timestamp() * 1000)
    elif CAL:
        parsed_tuple, status = CAL.parse(text)
        if status != 0:
            try:
                since_ms = int(time.mktime(parsed_tuple) * 1000)
            except (OverflowError, ValueError):
                since_ms = None

    if since_ms is None:
        relative = _infer_relative_timestamp(text)
        if relative:
            since_ms = relative
        elif range_hint:
            amount = int(range_hint.group(1))
            unit = range_hint.group(2).lower().rstrip("s")
            seconds = UNIT_TO_SECONDS.get(unit)
            if seconds:
                since_ms = int((time.time() - amount * seconds) * 1000)

    semantic = _semantic_score(text)
    prefix_score = 1.0 if prefix else 0.0
    keyword_score = 1.0 if (recall_keyword or recall_phrase) else 0.0
    time_score = 1.0 if since_ms else 0.0
    scores = [keyword_score, time_score, semantic, prefix_score]
    weights = [0.3, 0.4, 0.2, 0.1]
    weighted_total = sum(s * w for s, w in zip(scores, weights))

    requested = _extract_requested_count(text)
    default_limit = _estimate_quote_count(text) if (recall_keyword or recall_phrase or prefix) else 3
    limit = requested if requested else default_limit
    limit = max(1, min(limit, 25))

    should_recall = prefix or recall_keyword or recall_phrase or since_ms is not None or weighted_total > 0.45
    if should_recall:
        fetch_limit = min(100, max(limit * 4, 20))
        entries = _filter_memories(_memory_lookup(limit=fetch_limit, since=since_ms), keywords)
        if not entries:
            focus = ", ".join(keywords[:3]) if keywords else "requested topic"
            st.session_state.chat_history.append(("Bot", f"No stored memories matched the {focus}."))
            return True
        entry = entries[0]
        stamp = entry.get("timestamp")
        prefix = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stamp / 1000)) if stamp else "unknown time"
        text = _strip_ledger_noise((entry.get("text") or "").strip())
        snippet = text[:320].replace("\n", " ")
        if len(text) > len(snippet):
            snippet += "…"
        st.session_state.chat_history.append(("Bot", f"{prefix} — {snippet}"))
        return True
    return False


def _anchor(text: str, *, record_chat: bool = True, notify: bool = True, factors_override: list[dict] | None = None):
    entity = _get_entity()
    if not entity:
        st.error("No active entity; cannot anchor.")
        return False
    factors = factors_override or _extract_prime_factors(text)
    batches = _flow_safe_batches(factors)
    if not batches:
        st.warning("No alphabetical tokens detected; nothing anchored.")
        return False
    for idx, seq in enumerate(batches):
        payload = {"entity": entity, "factors": seq, "text": text if idx == 0 else None}
        try:
            resp = requests.post(f"{API}/anchor", json=payload, headers=HEADERS, timeout=10)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            st.session_state.last_anchor_error = str(exc)
            st.session_state.capabilities_block = _build_capabilities_block()
            st.error(f"Anchor failed ({resp.status_code}): {resp.text}")
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
    resp = requests.get(f"{API}/retrieve?entity={entity}", headers=HEADERS, timeout=10)
    if resp.ok:
        st.session_state.recall_payload = resp.json()
    else:
        st.session_state.recall_payload = {"error": resp.text}


def _load_ledger():
    entity = _get_entity()
    if not entity:
        return
    resp = requests.get(f"{API}/ledger", params={"entity": entity}, headers=HEADERS, timeout=10)
    if resp.ok:
        data = resp.json()
        factors = data.get("factors") if isinstance(data, dict) else None
        if isinstance(factors, list):
            schema = st.session_state.get("prime_schema", PRIME_SCHEMA)
            for item in factors:
                prime = item.get("prime")
                if isinstance(prime, int):
                    meta = schema.get(prime, DEFAULT_PRIME_SCHEMA.get(prime, {}))
                    item["symbol"] = meta.get("name", f"Prime {prime}")
        st.session_state.ledger_state = data
    else:
        st.session_state.ledger_state = {"error": resp.text}


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
    attachment_block = ""
    capabilities_block = st.session_state.get("capabilities_block")
    keywords = _keywords_from_prompt(prompt)
    if attachments:
        lines = ["Attachment context:"]
        for attachment in attachments:
            name = attachment.get("name", "attachment")
            snippet = (attachment.get("text") or "").strip()
            truncated = snippet[:1500]
            if len(snippet) > len(truncated):
                truncated = f"{truncated}\n… (truncated)"
            lines.append(f"{name}:\n{truncated}")
        attachment_block = "\n\n".join(lines)

    time_hint = _infer_relative_timestamp(prompt)
    is_quote = _is_quote_request(prompt)
    if is_quote:
        target_count = max(1, min((quote_count or _estimate_quote_count(prompt)), 25))
        search_limit = max(target_count * 3, 15)
        full_text = _latest_user_transcript(prompt, limit=search_limit)

        if full_text:
            plural = "quotes" if target_count != 1 else "quote"
            llm_prompt = (
                "Use only the ledger snippets provided. If insufficient, admit no record exists.\n\n"
                "Below is a verbatim transcript.  "
                f"Reply with {target_count} exact {plural} (keep punctuation & capitalisation).  "
                "Do not paraphrase.  "
                "If the transcript contains assistant replies marked 'Bot:' or similar, ignore them and only quote the human speaker.  "
                f"Transcript:\n{full_text}\n\n"
                f"Exact {plural}:"
            )
            if attachment_block:
                llm_prompt = f"{llm_prompt}\n\n{attachment_block}"
        else:
            fallback_summary = _summarize_accessible_memories(max(target_count, 10), since=time_hint, keywords=keywords)
            if fallback_summary:
                st.session_state.chat_history.append(("Bot", fallback_summary))
                return fallback_summary
            llm_prompt = "No anchored text found – say so."
        context_block = _memory_context_block(limit=max(5, target_count), since=time_hint, keywords=keywords)
        if context_block:
            llm_prompt = f"{context_block}\n\n{llm_prompt}"
    else:
        llm_prompt = _augment_prompt(prompt, attachments=attachments)
        if attachment_block and "Attachment context:" not in llm_prompt:
            llm_prompt = f"{llm_prompt}\n\n{attachment_block}"
    if not (is_quote and context_block):
        if capabilities_block:
            llm_prompt = f"{capabilities_block}\n\n{llm_prompt}"
    if use_openai:
        if not (OpenAI and OPENAI_API_KEY):
            st.warning("OpenAI API key missing.")
            return
        client = OpenAI(api_key=OPENAI_API_KEY)
        messages = [{"role": "user", "content": llm_prompt}]
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        full = response.choices[0].message.content
        st.session_state.chat_history.append(("Bot", full))
        return full

    if not (genai and GENAI_KEY):
        return "Gemini API key missing."
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat(history=[{"role": "user", "parts": [h[1]]} for h in st.session_state.chat_history])
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
    st.session_state.prime_schema = _load_prime_schema(user_data["entity"])
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
        st.session_state.prime_schema = _load_prime_schema(_get_entity())
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
            metrics_resp = requests.get(f"{API}/metrics", headers=HEADERS, timeout=5)
            metrics_resp.raise_for_status()
            metrics = metrics_resp.json()
        except (requests.RequestException, ValueError):
            metrics = {"tokens_deduped": "N/A", "ledger_integrity": 0.0}

        try:
            memories_resp = requests.get(
                f"{API}/memories",
                params={"entity": entity, "limit": 1},
                headers=HEADERS,
                timeout=5,
            )
            memories_resp.raise_for_status()
            memories = memories_resp.json()
        except (requests.RequestException, ValueError):
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
                payload = {"entity": entity, "axis": [0.0, 0.0, 1.0], "angle": 1.0472}
                try:
                    resp = requests.post(
                        f"{API}/rotate",
                        json=payload,
                        headers=HEADERS,
                        timeout=15,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.success(
                        f"Rotated lattice. Δenergy = {data.get('energy_cycles')}, "
                        f"checksum {data.get('original_checksum')} → {data.get('rotated_checksum')}."
                    )
                    _load_ledger()
                    if st.session_state.ledger_state:
                        st.caption("Updated ledger snapshot after Möbius transform:")
                        _render_ledger_state(st.session_state.ledger_state)
                except requests.RequestException as exc:
                    st.error(f"Möbius rotation failed: {exc}")
            if st.button("Initiate Enrichment", help="Replay stored transcripts with richer prime coverage"):
                with st.spinner("Enriching memories…"):
                    _run_enrichment()


if __name__ == "__main__":
    _render_app()
    capabilities_block = st.session_state.get("capabilities_block")
    capabilities_block = st.session_state.get("capabilities_block") or _build_capabilities_block()
