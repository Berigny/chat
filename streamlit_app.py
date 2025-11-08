import audioop
import base64
import hashlib
import io
import math
import os
import re
import struct
import time
import wave

import requests
import streamlit as st
try:
    import speech_recognition as sr
except ModuleNotFoundError:  # pragma: no cover - Streamlit Cloud bootstrap
    sr = None
API = "https://dualsubstrate-commercial.fly.dev"
ENTITY = "demo_user"
API_KEY = st.secrets.get("DUALSUBSTRATE_API_KEY") or os.getenv("DUALSUBSTRATE_API_KEY") or "demo-key"
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}

st.set_page_config(page_title="DualSubstrate Live Demo", layout="centered")
st.title("Live Prime-Ledger Demo (Browser STT)")
st.markdown(
    """
    <style>
    .metric-card {transition: all 0.5s ease; border-radius: 0.75rem; padding: 0.5rem;}
    .metric-card:hover {transform: scale(1.05); box-shadow: 0 0.5rem 1.25rem rgba(0,0,0,0.15);}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- tiny helpers ----------
PRIME_WORDS = "the and is to of a in that it with on for are as this was at be by an".split()
WORD_TO_PRIME = {w: p for p, w in enumerate(PRIME_WORDS, start=11)}
PRIME_TO_WORD = {p: w for w, p in WORD_TO_PRIME.items()}
FALLBACK_LABEL = "novel token"


def _hash(s: str):
    """Map words → pseudo primes with unit deltas for the demo ledger."""
    tokens = re.findall(r"[A-Za-z]+", s)
    return [
        {"prime": WORD_TO_PRIME.get(word.lower(), 2), "delta": 1}
        for word in tokens
    ][:30]


def _tts(text: str) -> str:
    """Generate a short base64 WAV tone derived from the text."""
    if not text:
        return ""
    sample_rate = 16000
    duration = min(max(len(text) * 0.05, 0.4), 3.0)
    freq = 260 + (sum(ord(ch) for ch in text) % 480)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(int(sample_rate * duration)):
            sample = int(32767 * 0.25 * math.sin(2 * math.pi * freq * i / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wf.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode()


def _format_timestamp(ms: int) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(ms / 1000))
    except Exception:
        return str(ms)


def _top_primes(factors, limit: int = 3):
    if not factors:
        return []
    ranked = sorted(
        (f for f in factors if f.get("value")),
        key=lambda f: abs(f.get("value", 0)),
        reverse=True,
    )
    return ranked[:limit]


# ---------- browser recording ----------
recognizer = sr.Recognizer() if sr else None
if recognizer:
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 100
if "last_text" not in st.session_state:
    st.session_state.last_text = None
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "last_audio_digest" not in st.session_state:
    st.session_state.last_audio_digest = None
if "recall_status" not in st.session_state:
    st.session_state.recall_status = None
    st.session_state.recall_error = None
    st.session_state.recall_payload = None
if "ledger_payload" not in st.session_state:
    st.session_state.ledger_payload = None
    st.session_state.ledger_error = None
if "last_hypothesis" not in st.session_state:
    st.session_state.last_hypothesis = None
if "history" not in st.session_state:
    st.session_state.history = []
if "typed_input" not in st.session_state:
    st.session_state.typed_input = ""
if "clear_typed" not in st.session_state:
    st.session_state.clear_typed = False
if st.session_state.clear_typed:
    st.session_state.typed_input = ""
    st.session_state.clear_typed = False


def _normalize_audio(raw_bytes: bytes) -> io.BytesIO:
    """Convert arbitrary WAV input to mono 16kHz 16-bit PCM for SR."""
    try:
        with wave.open(io.BytesIO(raw_bytes)) as wf:
            params = wf.getparams()
            audio = wf.readframes(params.nframes)
            sampwidth = params.sampwidth
            channels = params.nchannels
            rate = params.framerate
    except wave.Error as exc:
        raise RuntimeError("Unsupported audio format. Please record again.") from exc

    # convert sample width to 16-bit
    if sampwidth != 2:
        audio = audioop.lin2lin(audio, sampwidth, 2)
        sampwidth = 2

    # collapse to mono
    if channels != 1:
        audio = audioop.tomono(audio, sampwidth, 0.5, 0.5)
        channels = 1

    # resample to 16kHz
    target_rate = 16000
    if rate != target_rate:
        audio, _ = audioop.ratecv(audio, sampwidth, channels, rate, target_rate, None)
        rate = target_rate
    # boost quiet clips
    peak = audioop.max(audio, sampwidth) or 1
    if peak < 8000:
        factor = min(4.0, 20000 / peak)
        audio = audioop.mul(audio, sampwidth, factor)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_rate)
        wf.writeframes(audio)
    buf.seek(0)
    return buf


def _transcribe_audio(raw_bytes: bytes) -> str:
    """Turn recorded audio into text using Google's free recognizer."""
    if recognizer is None:
        raise RuntimeError("Speech recognizer unavailable.")
    audio_buffer = _normalize_audio(raw_bytes)
    audio_buffer.name = "input.wav"
    with sr.AudioFile(audio_buffer) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.15)
        audio_data = recognizer.record(source)
    try:
        all_hyp = recognizer.recognize_google(audio_data, show_all=True)
    except sr.UnknownValueError as exc:
        st.session_state.last_hypothesis = {"error": "unknown_value"}
        raise RuntimeError("Could not understand the audio sample.") from exc
    except sr.RequestError as exc:
        st.session_state.last_hypothesis = {"error": "request", "detail": str(exc)}
        raise RuntimeError(f"Speech service unavailable: {exc}") from exc

    if isinstance(all_hyp, dict) and all_hyp.get("alternative"):
        best = all_hyp["alternative"][0]
        transcript = best.get("transcript", "").strip()
        if transcript:
            st.session_state.last_hypothesis = best
            return transcript
        raise RuntimeError("Google STT returned empty transcript.")
    st.session_state.last_hypothesis = all_hyp or {"error": "empty"}
    raise RuntimeError("No speech detected—try recording again a bit louder.")


def _anchor_text(text: str) -> None:
    """Send the captured sentence to the ledger API."""
    factors = _hash(text)
    if not factors:
        st.warning("Text did not contain any alphabetical tokens to anchor.")
        return
    payload = {"entity": ENTITY, "factors": factors, "text": text}
    try:
        resp = requests.post(
            f"{API}/anchor",
            json=payload,
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response else "?"
        body = exc.response.text if exc.response else ""
        st.error(f"Anchor failed ({status}): {body or exc}")
        return
    except requests.RequestException as exc:
        st.error(f"Anchor failed: {exc}")
        return

    st.success("Anchored last sentence.")
    resp_payload = None
    try:
        resp_payload = resp.json()
        st.write(resp_payload)
    except ValueError:
        st.info("Anchor succeeded but returned non-JSON payload.")
    st.session_state.last_text = text
    _fetch_ledger(ENTITY)
    history_entry = {
        "text": text,
        "timestamp": (resp_payload or {}).get("timestamp", int(time.time() * 1000)),
        "ledger": st.session_state.ledger_payload,
    }
    st.session_state.history.append(history_entry)
    st.session_state.history = st.session_state.history[-12:]


def _fetch_ledger(entity: str) -> None:
    """Load the persisted factor vector for this entity from RocksDB."""
    try:
        resp = requests.get(
            f"{API}/ledger",
            params={"entity": entity},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.session_state.ledger_payload = None
        st.session_state.ledger_error = str(exc)
        st.warning(f"Ledger fetch failed: {exc}")
        return
    st.session_state.ledger_error = None
    st.session_state.ledger_payload = resp.json()


col_voice, col_text = st.columns(2)
with col_voice:
    if recognizer is None:
        st.warning("SpeechRecognition package missing. Please reinstall dependencies.")
    else:
        audio_file = st.audio_input(
            "Press to talk",
            key="stream_audio",
            help="Hold to record, release to process.",
        )
        if audio_file:
            audio_bytes = audio_file.getvalue()
            st.session_state.last_audio = audio_bytes
            st.caption(
                f"Clip bytes: {len(audio_bytes)} | MIME: {audio_file.type or 'unknown'}"
            )
            st.audio(audio_bytes, format="audio/wav")
            digest = hashlib.sha1(audio_bytes).hexdigest()
            if digest == st.session_state.last_audio_digest:
                st.info("Audio already processed. Record again to capture new text.")
            else:
                st.session_state.last_audio_digest = digest
                try:
                    transcript = _transcribe_audio(audio_bytes)
                except RuntimeError as exc:
                    st.warning(str(exc))
                except Exception as exc:
                    detail = str(exc).strip() or exc.__class__.__name__
                    st.error(f"Transcription failed: {detail}")
                else:
                    if transcript.strip():
                        st.success(f"Captured: {transcript}")
                        _anchor_text(transcript)
                    else:
                        st.warning("No speech detected in the recording.")

with col_text:
    st.markdown("**Typed memory**")
    with st.form("typed_memory"):
        typed_text = st.text_area(
            "Describe the same identity in text", height=120, key="typed_input"
        )
        submitted = st.form_submit_button("Anchor typed text")
        if submitted:
            content = typed_text.strip()
            if not content:
                st.warning("Please enter some text before anchoring.")
            else:
                _anchor_text(content)
                st.session_state.clear_typed = True


if st.button("🔍 Recall last sentence"):
    try:
        resp = requests.get(f"{API}/retrieve?entity={ENTITY}", headers=HEADERS, timeout=10)
        st.session_state.recall_status = resp.status_code
        st.session_state.recall_payload = resp.json() if resp.ok else None
        st.session_state.recall_error = None if resp.ok else resp.text
    except requests.RequestException as exc:
        st.session_state.recall_status = None
        st.session_state.recall_payload = None
        st.session_state.recall_error = str(exc)
    finally:
        _fetch_ledger(ENTITY)

# render outside the button block so it survives reruns
if st.session_state.recall_error:
    st.warning(f"Recall failed: {st.session_state.recall_error}")
elif st.session_state.recall_payload:
    st.write(st.session_state.recall_payload)
    recalled_text = st.session_state.recall_payload.get("text")
    if not recalled_text and st.session_state.last_text:
        recalled_text = st.session_state.last_text
    audio_payload = _tts(recalled_text) if recalled_text else ""
    if audio_payload:
        st.components.v1.html(
            f'<audio autoplay><source src="data:audio/wav;base64,{audio_payload}" type="audio/wav"></audio>',
            height=0,
        )

if st.session_state.get("last_hypothesis"):
    with st.expander("Debug: last speech hypothesis"):
        st.json(st.session_state.last_hypothesis)

if st.session_state.ledger_payload:
    st.subheader("Ledger snapshot (RocksDB)")
    factors = st.session_state.ledger_payload.get("factors")
    top_primes = _top_primes(factors)
    if top_primes:
        trait_cols = st.columns(len(top_primes))
        for col, trait in zip(trait_cols, top_primes):
            label = PRIME_TO_WORD.get(trait["prime"], FALLBACK_LABEL).title()
            col.metric(
                label,
                trait["value"],
                help=f"Prime {trait['prime']}",
            )
    st.json(st.session_state.ledger_payload)
elif st.session_state.ledger_error:
    st.info(f"Ledger snapshot unavailable: {st.session_state.ledger_error}")

if st.session_state.history:
    st.subheader("Identity timeline")
    for idx, entry in enumerate(reversed(st.session_state.history), start=1):
        title = f"{_format_timestamp(entry['timestamp'])} • {entry['text'][:60]}"
        with st.expander(title):
            st.write(entry["text"])
            factors = (entry.get("ledger") or {}).get("factors")
            if factors:
                st.caption("Top primes")
                for trait in _top_primes(factors, limit=5):
                    label = PRIME_TO_WORD.get(trait["prime"], FALLBACK_LABEL).title()
                    st.write(f"- {label} (prime {trait['prime']}): {trait['value']}")
            else:
                st.caption("No ledger snapshot captured for this entry.")

# ---------- metrics ----------
tokens_saved = 0
integrity = 0.0
try:
    metrics_resp = requests.get(f"{API}/metrics", headers=HEADERS, timeout=10)
    metrics_resp.raise_for_status()
    m = metrics_resp.json()
    tokens_saved = m.get("tokens_deduped", m.get("tokens_saved", 0))
    integrity = m.get("ledger_integrity", m.get("integrity", 0.0))
except requests.RequestException as exc:
    st.warning(f"Metrics unavailable: {exc}")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("💰 Tokens saved", tokens_saved, label_visibility="visible")
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("🔒 Ledger integrity %", f"{integrity*100:.1f} %", label_visibility="visible")
    st.markdown("</div>", unsafe_allow_html=True)
