import base64
import io
import math
import os
import struct
import wave

import requests
import streamlit as st

API = "https://dualsubstrate-commercial.fly.dev"
API_KEY = st.secrets.get("DUALSUBSTRATE_API_KEY") or os.getenv("DUALSUBSTRATE_API_KEY") or "demo-key"
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}

st.set_page_config(page_title="DualSubstrate Live Demo", layout="centered")
st.title("🎤 Live Prime-Ledger Demo")
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
    return [{"prime": mapping.get(word.lower(), 2), "k": 1} for word in s.split() if word.isalpha()][:30]


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


# ---------- session state ----------
if "last_text" not in st.session_state:
    st.session_state.last_text = None
if "recall_status" not in st.session_state:
    st.session_state.recall_status = None
    st.session_state.recall_error = None
    st.session_state.recall_payload = None


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

# ---- JS that actually talks to Streamlit ----
html = """
<button id="talk-button">Start talking</button>
<p id="status-message"></p>
<script>
    const button = document.getElementById('talk-button');
    const statusMessage = document.getElementById('status-message');
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition;

    if (!SpeechRecognition) {
        button.disabled = true;
        statusMessage.textContent = 'Speech recognition not supported in this browser.';
    } else {
        recognition = new SpeechRecognition();
        recognition.continuous = false; // Only capture a single utterance
        recognition.interimResults = false; // We only want the final result

        recognition.onstart = () => {
            button.textContent = 'Listening...';
            button.disabled = true;
        };

        recognition.onend = () => {
            button.textContent = 'Start talking';
            button.disabled = false;
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: { 'text': transcript }
            }, '*');
        };

        button.onclick = () => {
            button.textContent = 'Listening...';
            button.disabled = true;
            recognition.start();
        };
    }
</script>
"""

# ---- register the component ----
text = st.components.v1.html(html, height=80)

# ---- if JS sent something, process it ----
if isinstance(text, dict) and "text" in text:
    st.write(f"Captured: **{text['text']}**")
    _anchor_text(text['text'])

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
