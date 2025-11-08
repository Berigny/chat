import base64
import hashlib
import io
import math
import os
import struct
import wave

import requests
import streamlit as st
try:
    import speech_recognition as sr
except ModuleNotFoundError:  # pragma: no cover - Streamlit Cloud bootstrap
    sr = None
API = "https://dualsubstrate-commercial.fly.dev"
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
def _hash(s: str):
    """Map words → pseudo primes with unit deltas for the demo ledger."""
    words = "the and is to of a in that it with on for are as this was at be by an".split()
    mapping = {w: p for p, w in enumerate(words, start=11)}
    return [{"prime": mapping.get(word.lower(), 2), "delta": 1} for word in s.split() if word.isalpha()][:30]


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


# ---------- browser recording ----------
recognizer = sr.Recognizer() if sr else None
if recognizer:
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 150
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


def _transcribe_audio(raw_bytes: bytes) -> str:
    """Turn recorded audio into text using Google's free recognizer."""
    if recognizer is None:
        raise RuntimeError("Speech recognizer unavailable.")
    audio_buffer = io.BytesIO(raw_bytes)
    audio_buffer.name = "input.wav"
    with sr.AudioFile(audio_buffer) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.05)
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:
        raise RuntimeError(f"Speech service unavailable: {exc}") from exc


def _anchor_text(text: str) -> None:
    """Send the captured sentence to the ledger API."""
    payload = {"entity": "demo_user", "factors": _hash(text), "text": text}
    try:
        resp = requests.post(
            f"{API}/anchor",
            json=payload,
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        st.success("Anchored last sentence.")
        st.write(resp.json())
        st.session_state.last_text = text
    except requests.RequestException as exc:
        st.error(f"Anchor failed: {exc}")


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
        st.audio(audio_bytes, format="audio/wav")
        digest = hashlib.sha1(audio_bytes).hexdigest()
        if digest == st.session_state.last_audio_digest:
            st.info("Audio already processed. Record again to capture new text.")
        else:
            st.session_state.last_audio_digest = digest
            try:
                transcript = _transcribe_audio(audio_bytes)
                if transcript:
                    st.success(f"Captured: {transcript}")
                    _anchor_text(transcript)
                else:
                    st.warning("No speech detected in the recording.")
            except sr.UnknownValueError:
                st.warning("Did not catch that—please try again with a clearer sample.")
            except Exception as exc:
                detail = str(exc).strip() or exc.__class__.__name__
                st.error(f"Transcription failed: {detail}")

if st.button("🔍 Recall last sentence"):
    try:
        resp = requests.get(f"{API}/retrieve?entity=demo_user", headers=HEADERS, timeout=10)
        st.session_state.recall_status = resp.status_code
        st.session_state.recall_payload = resp.json() if resp.ok else None
        st.session_state.recall_error = None if resp.ok else resp.text
    except requests.RequestException as exc:
        st.session_state.recall_status = None
        st.session_state.recall_payload = None
        st.session_state.recall_error = str(exc)

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
