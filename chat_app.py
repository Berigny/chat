import audioop
import hashlib
import html
import io
import os
import re
import time
import wave

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

st.set_page_config(page_title="Ledger Chat", layout="wide")
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem !important;
        font-weight: 400 !important;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }
    </style>
    <h1 class="main-title">What needs remembering next?</h1>
    """,
    unsafe_allow_html=True,
)

TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?\s?(?:am|pm)?)\b", re.IGNORECASE)
CAL = pdt.Calendar() if pdt else None
KEYWORD_PATTERN = re.compile(r"\b(quote|verbatim|exact|recall|retrieve|what did i say)\b", re.I)
PREFIXES = ("/q", "@ledger", "::memory")

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


def _augment_prompt(user_question: str) -> str:
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
    prompt_lines.append(f"User question: {user_question}")
    return "\n".join(prompt_lines)


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


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

    if weighted_total > 0.45 or prefix:
        entries = _memory_lookup(limit=5 if prefix else 3, since=since_ms)
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


def _process_memory_text(text: str, use_openai: bool):
    cleaned = (text or "").strip()
    if not cleaned:
        st.warning("Enter some text first.")
        return
    if _maybe_handle_recall_query(cleaned):
        return
    quote_mode = _is_quote_request(cleaned)
    bot_reply = _chat_response(cleaned, use_openai=use_openai)
    if bot_reply is None:
        bot_reply = ""
    _update_rolling_memory(cleaned, bot_reply, quote_mode=quote_mode)


def _recall():
    resp = requests.get(f"{API}/retrieve?entity={ENTITY}", headers=HEADERS, timeout=10)
    if resp.ok:
        st.session_state.recall_payload = resp.json()
    else:
        st.session_state.recall_payload = {"error": resp.text}


def _load_ledger():
    resp = requests.get(f"{API}/ledger", params={"entity": ENTITY}, headers=HEADERS, timeout=10)
    st.session_state.ledger_state = resp.json() if resp.ok else {"error": resp.text}


def _chat_response(prompt: str, use_openai=False):
    if _is_quote_request(prompt):
        full_text = _latest_user_transcript(prompt)

        if full_text:
            llm_prompt = (
                "Below is a verbatim transcript.  "
                "Reply with TEN exact quotes (keep punctuation & capitalisation).  "
                "Do not paraphrase.  "
                "If the transcript contains assistant replies marked 'Bot:' or similar, ignore them and only quote the human speaker.  "
                f"Transcript:\n{full_text}\n\n"
                "Ten exact quotes:"
            )
        else:
            llm_prompt = "No anchored text found – say so."
    else:
        llm_prompt = _augment_prompt(prompt)
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

icon_cols = st.columns([0.1, 0.1, 0.8])
with icon_cols[0]:
    if st.button("🎙️", help="Record voice memory"):
        st.session_state.input_mode = "mic"
with icon_cols[1]:
    if st.button("📎", help="Attach a memory file"):
        st.session_state.input_mode = "file"
with icon_cols[2]:
    st.caption("type, speak or attach a new memory")

prompt = st.chat_input("type, speak or attach a new memory")
if prompt:
    _process_memory_text(prompt, use_openai=True)

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
                    text = transcript.text
                    st.caption(f"Transcript: {text}")
                    if text:
                        _process_memory_text(text, use_openai=True)
                        st.session_state.input_mode = "text"
                except Exception as exc:
                    st.error(f"Transcription failed: {exc}")
    if st.session_state.input_mode == "mic":
        st.session_state.input_mode = "text"
elif st.session_state.input_mode == "file":
    uploaded = st.file_uploader("Attach a new memory", label_visibility="collapsed")
    if uploaded:
        st.caption(f"Attached file: {uploaded.name}")
        st.session_state.input_mode = "text"

tab_chat, tab_about = st.tabs(["Chat", "About DualSubstrate"])

with tab_chat:
    recent_history = list(reversed(st.session_state.chat_history[-20:]))
    if recent_history:
        entries = [
            f"<div class='chat-entry'><strong>{html.escape(role)}:</strong> {html.escape(content)}</div>"
            for role, content in recent_history
        ]
        stream_html = "<hr>".join(entries)
    else:
        stream_html = "<div class='chat-entry'>No chat history yet.</div>"
    st.markdown(f"<div class='chat-stream'>{stream_html}</div>", unsafe_allow_html=True)

with tab_about:
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(
            """
            
            <div class="about-col about-col-left">
                <h2 class="about-heading" style="font-size: 0.95rem; font-weight: 400">DualSubstrate ledger demo</h2>
                <p class="about-text">To test this DualSubstrate ledger demo speak or type. Everything anchors to the prime-based ledger. Tip: type /q or “what did I say at 7 pm” and I’ll quote you word-for-word from the prime-ledger. Anything else = normal chat.</p>
                <hr>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_right:
        st.markdown(
            """
            <div class="about-col about-col-right">
                <div class="metrics-heading" style="font-size: 0.95rem; font-weight: 400">Metrics</div>
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

    st.markdown("<hr class='full-divider'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="prime-heading" style="font-size: 0.95rem; font-weight: 400">Prime-Ledger Snapshot</div>
        <p class="prime-text">A live, word-perfect copy of everything you’ve anchored - sealed in primes, mathematically identical forever.</p>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Load ledger", key="load_ledger"):
        _load_ledger()
    if st.session_state.ledger_state:
        st.json(st.session_state.ledger_state)
