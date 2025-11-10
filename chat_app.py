import audioop
import base64
import hashlib
import html
import io
import mimetypes
import os
import re
import time
import wave

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
    import dateparser
except ModuleNotFoundError:
    dateparser = None

API = "https://dualsubstrate-commercial.fly.dev"
ENTITY = "demo_user"
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


PRIME_ARRAY = (2, 3, 5, 7, 11, 13, 17, 19)
PRIME_WORDS = ("identity", "memory", "voice", "ledger", "trust", "security", "insight", "story")
WORD_TO_PRIME = {w: p for w, p in zip(PRIME_WORDS, PRIME_ARRAY)}
FALLBACK_PRIME = PRIME_ARRAY[0]


def _hash_text(text: str):
    tokens = re.findall(r"[A-Za-z]+", text)
    return [{"prime": WORD_TO_PRIME.get(tok.lower(), FALLBACK_PRIME), "delta": 1} for tok in tokens][:30]


TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?\s?(?:am|pm)?)\b", re.IGNORECASE)
CAL = pdt.Calendar() if pdt else None
KEYWORD_PATTERN = re.compile(r"\b(quote|verbatim|exact|recall|retrieve|what did i say)\b", re.I)
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
    params = {"entity": ENTITY, "limit": limit}
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
        st.session_state.chat_history.append(("Memory", "No matching memories."))
        return
    for entry in entries:
        stamp = entry.get("timestamp")
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stamp / 1000)) if stamp else "unknown time"
        text = entry.get("text", "(no text)")
        msg = f"{ts} — {text}"
        st.session_state.chat_history.append(("Memory", msg))


def _is_quote_request(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.startswith(PREFIXES) or KEYWORD_PATTERN.search(normalized) is not None


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
    try:
        resp = requests.get(
            f"{API}/memories",
            params={"entity": ENTITY, "limit": 1},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        memories = resp.json()
    except requests.RequestException:
        return user_question

    if not memories:
        return user_question

    full_context = _strip_ledger_noise(memories[0].get("text", "").strip())
    if not full_context:
        return user_question

    prompt_lines = [
        "Anchored conversation context:",
        full_context,
        "",
    ]
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
    if mime.startswith("text/") or mime in {"application/json", "application/xml", "application/javascript"}:
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

    max_chars = 8_000
    if len(text) > max_chars:
        text = f"{text[:max_chars]}\n… (truncated)"

    return {"name": name, "mime": mime, "text": text}


def _latest_user_transcript(current_request: str, *, limit: int = 5) -> str | None:
    entries = _memory_lookup(limit=limit)
    if not entries:
        return None

    normalized_request = _normalize_for_match(current_request)

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
        if sanitized:
            return sanitized

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
    keyword = KEYWORD_PATTERN.search(normalized) is not None
    since_ms = None

    parsed_datetime = None
    parsed_epoch = None
    if dateparser:
        parsed_datetime = dateparser.parse(text, settings={"PREFER_DATES_FROM": "past"})
    if CAL and parsed_datetime is None:
        match = TIME_PATTERN.search(text)
        if match:
            parsed_tuple, status = CAL.parse(match.group(1))
            if status != 0:
                parsed_epoch = int(time.mktime(parsed_tuple) * 1000)
                if dateparser:
                    parsed_str = time.strftime("%Y-%m-%d %H:%M:%S", parsed_tuple)
                    parsed_datetime = dateparser.parse(parsed_str)
    if parsed_datetime:
        since_ms = int(parsed_datetime.timestamp() * 1000)
    elif parsed_epoch:
        since_ms = parsed_epoch

    semantic = _semantic_score(text)
    prefix_score = 1.0 if prefix else 0.0
    keyword_score = 1.0 if keyword else 0.0
    time_score = 1.0 if since_ms else 0.0
    scores = [keyword_score, time_score, semantic, prefix_score]
    weights = [0.3, 0.4, 0.2, 0.1]
    weighted_total = sum(s * w for s, w in zip(scores, weights))

    requested = _extract_requested_count(text)
    default_limit = _estimate_quote_count(text) if (keyword or prefix) else 3
    limit = requested if requested else default_limit
    limit = max(1, min(limit, 25))

    if weighted_total > 0.45 or prefix:
        entries = _memory_lookup(limit=limit, since=since_ms)
        _render_memories(entries)
        return True
    return False


def _anchor(text: str, *, record_chat: bool = True):
    factors = _hash_text(text)
    if not factors:
        st.warning("No alphabetical tokens detected; nothing anchored.")
        return False
    payload = {"entity": ENTITY, "factors": factors, "text": text}
    resp = requests.post(f"{API}/anchor", json=payload, headers=HEADERS, timeout=10)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        st.error(f"Anchor failed ({resp.status_code}): {resp.text}")
        return False
    if record_chat:
        st.session_state.chat_history.append(("You", text))
    st.success("Anchored into ledger.")
    return True


def _recall():
    resp = requests.get(f"{API}/retrieve?entity={ENTITY}", headers=HEADERS, timeout=10)
    if resp.ok:
        st.session_state.recall_payload = resp.json()
    else:
        st.session_state.recall_payload = {"error": resp.text}


def _load_ledger():
    resp = requests.get(f"{API}/ledger", params={"entity": ENTITY}, headers=HEADERS, timeout=10)
    st.session_state.ledger_state = resp.json() if resp.ok else {"error": resp.text}


def _chat_response(
    prompt: str,
    use_openai=False,
    *,
    quote_count: int | None = None,
    attachments: list[dict] | None = None,
):
    attachment_block = ""
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

    if _is_quote_request(prompt):
        target_count = max(1, min((quote_count or _estimate_quote_count(prompt)), 25))
        full_text = _latest_user_transcript(prompt)

        if full_text:
            plural = "quotes" if target_count != 1 else "quote"
            llm_prompt = (
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
            llm_prompt = "No anchored text found – say so."
    else:
        llm_prompt = _augment_prompt(prompt, attachments=attachments)
        if attachment_block and "Attachment context:" not in llm_prompt:
            llm_prompt = f"{llm_prompt}\n\n{attachment_block}"
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


def _render_app():
    st.set_page_config(page_title="Ledger Chat", layout="wide")

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
        ".st-key-top_mic {position: absolute; right:30px !important; bottom: 2px; opacity: 0.5;}",
        ".st-key-top_attach > button {}",
        ".st-key-top_mic > button {}",
        "div[data-testid='stChatInput'] {position:static !important;margin:0.25rem auto 0;}",
        "div[data-testid='stChatInput'] > div:first-child {position:relative;border:1px solid rgba(255,255,255,0.18);padding:1.5rem 4.5rem 1.5rem 3.25rem;transition:border-color 0.2s ease, box-shadow 0.2s ease;}",
        "div[data-testid='stChatInput']:active-within > div:first-child {border-color:rgba(255,255,255,0.3);box-shadow:0 0 0 1px rgba(255,255,255,0.18);}",
        "textarea[data-testid='stChatInputTextArea'] {max-height:250px!important; overflow-y:auto;padding-left:0 !important;padding-right:0 !important;}",
        "textarea[data-testid='stChatInputTextArea']:focus {min-height:250px !important;}",
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
    if "pending_attachments" not in st.session_state:
        st.session_state.pending_attachments = []

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
    try:
        metrics_resp = requests.get(f"{API}/metrics", headers=HEADERS, timeout=5)
        metrics_resp.raise_for_status()
        metrics = metrics_resp.json()
    except (requests.RequestException, ValueError):
        metrics = {"tokens_deduped": "N/A", "ledger_integrity": 0.0}

    try:
        memories_resp = requests.get(
            f"{API}/memories",
            params={"entity": ENTITY, "limit": 1},
            headers=HEADERS,
            timeout=5,
        )
        memories_resp.raise_for_status()
        memories = memories_resp.json()
    except (requests.RequestException, ValueError):
        memories = []

    oldest = (
        memories[-1].get("timestamp", time.time() * 1000) / 1000
        if isinstance(memories, list) and memories
        else time.time()
    )
    durability_h = (time.time() - oldest) / 3600
    tokens_saved = metrics.get("tokens_deduped", "N/A")
    ledger_integrity = metrics.get("ledger_integrity", 0.0)

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
                        text = getattr(transcript, "text", None) or transcript.get("text") if isinstance(transcript, dict) else None
                        if text:
                            st.caption(f"Transcript: {text}")
                            st.session_state.top_input = text
                        else:
                            st.warning("No transcript returned from Whisper.")
                    except Exception as exc:
                        st.error(f"Transcription failed: {exc}")
        if st.session_state.input_mode == "mic":
            st.session_state.input_mode = "text"
    elif st.session_state.input_mode == "file":
        uploaded = st.file_uploader("Attach a new memory", label_visibility="collapsed")
        if uploaded:
            attachment = _ingest_attachment(uploaded)
            if attachment:
                st.session_state.pending_attachments.append(attachment)
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
                st.json(st.session_state.ledger_state)
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


if __name__ == "__main__":
    _render_app()
