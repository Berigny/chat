import base64
import io
import json
import math
import struct
import wave
from urllib.parse import unquote

import requests
import streamlit as st
API = "https://dualsubstrate-commercial.fly.dev"

st.set_page_config(page_title="DualSubstrate Live Demo", layout="centered")
st.title("🎤 Live Prime-Ledger Demo (Browser STT)")
st.markdown(
    """
    <style>
    .metric-card {transition: all 0.5s ease; border-radius: 0.75rem; padding: 0.5rem;}
    .metric-card:hover {transform: scale(1.05); box-shadow: 0 0.5rem 1.25rem rgba(0,0,0,0.15);}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- browser STT (no Whisper) ----------
html = """
<script>
function startSTT(){
  const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!Speech){
    alert("SpeechRecognition not supported in this browser.");
    return;
  }
  const r = new Speech();
  r.interimResults=false; r.lang='en-US';
  r.onresult=e=>{
    const t=e.results[0][0].transcript;
    const url = new URL(window.parent.location.href);
    url.searchParams.set('data', JSON.stringify({text:t}));
    window.parent.location.replace(url.toString());
  };
  r.onerror=err=>console.error(err);
  r.start();
}
</script>
<button onclick="startSTT()">Start talking</button>
"""
st.components.v1.html(html, height=100)

# ---------- callback route via Streamlit ----------
if "last" not in st.session_state:
    st.session_state.last = None

if st.query_params.get("data"):
    raw_payload = st.query_params["data"][0]
    try:
        st.session_state.last = json.loads(unquote(raw_payload))
    except json.JSONDecodeError:
        st.warning("Received malformed STT payload.")
    finally:
        st.query_params.clear()

# ---------- tiny helper ----------
def _hash(s: str):
    MAP = {w: p for p, w in enumerate("the and is to of a in that it with on for are as this was at be by an".split(), start=11)}
    return [{"prime": MAP.get(w.lower(), 2), "k": 1} for w in s.split() if w.isalpha()][:30]


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

# ---------- call engine ----------
if st.session_state.last:
    text = st.session_state.last.get("text", "")
    if text:
        try:
            resp = requests.post(
                f"{API}/anchor",
                json={"entity": "demo_user", "factors": _hash(text)},
                timeout=10,
            )
            resp.raise_for_status()
            st.success("Anchored last sentence.")
            st.write(resp.json())
        except requests.RequestException as exc:
            st.error(f"Anchor failed: {exc}")
        finally:
            st.session_state.last = None

if st.button("🔍 Recall last sentence"):
    try:
        resp = requests.get(f"{API}/retrieve?entity=demo_user", timeout=10)
        if not resp.ok:
            st.warning(f"Recall failed: {resp.status_code} {resp.text}")
        else:
            payload = resp.json()
            st.write(payload)
            recalled_text = payload.get("text")
            if not recalled_text and st.session_state.last:
                recalled_text = st.session_state.last.get("text")
            audio_payload = _tts(recalled_text) if recalled_text else ""
            if audio_payload:
                st.components.v1.html(
                    f'<audio autoplay><source src="data:audio/wav;base64,{audio_payload}" type="audio/wav"></audio>',
                    height=0,
                )
    except requests.RequestException as exc:
        st.error(f"Recall failed: {exc}")

# ---------- metrics ----------
tokens_saved = 0
integrity = 0.0
try:
    metrics_resp = requests.get(f"{API}/metrics", timeout=10)
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
