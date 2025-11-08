import audioop
import audioop
import base64
import hashlib
import io
import os
import re
import time
import wave

import requests
import streamlit as st
import speech_recognition as sr
try:
    import google.generativeai as genai
except ModuleNotFoundError:
    genai = None

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

st.set_page_config(page_title="Ledger Chat", layout="centered")
st.title("Ledger Chat with Persistent Memory")
st.caption("Speak or type. Everything anchors to the DualSubstrate ledger.")

recognizer = sr.Recognizer() if sr else None
if recognizer:
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 120

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_audio_digest" not in st.session_state:
    st.session_state.last_audio_digest = None
if "typed_input" not in st.session_state:
    st.session_state.typed_input = ""
if "clear_typed" not in st.session_state:
    st.session_state.clear_typed = False
if st.session_state.clear_typed:
    st.session_state.typed_input = ""
    st.session_state.clear_typed = False
if "ledger_state" not in st.session_state:
    st.session_state.ledger_state = None
if "recall_payload" not in st.session_state:
    st.session_state.recall_payload = None


def _normalize_audio(raw_bytes: bytes) -> io.BytesIO:
    with wave.open(io.BytesIO(raw_bytes)) as wf:
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


def _anchor(text: str):
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


def _chat_response(prompt: str):
    if not (genai and GENAI_KEY):
        return "Gemini API key missing."
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat(history=[{"role": "user", "parts": [h[1]]} for h in st.session_state.chat_history])
    chunks = []
    for chunk in chat.send_message(prompt, stream=True):
        chunks.append(chunk.text)
    full = "".join(chunks).strip()
    st.session_state.chat_history.append(("Bot", full))
    return full or "(No response)"


st.sidebar.header("Live Memory")
if st.sidebar.button("Recall"):
    _recall()
if st.sidebar.button("Load ledger"):
    _load_ledger()

if st.session_state.recall_payload:
    st.sidebar.write("Last recall:", st.session_state.recall_payload)
if st.session_state.ledger_state:
    st.sidebar.write("Ledger:", st.session_state.ledger_state)

col_voice, col_text = st.columns(2)
with col_voice:
    st.markdown("### Speak")
    if recognizer is None:
        st.warning("SpeechRecognition missing on this environment.")
    else:
        audio = st.audio_input("Hold to talk", key="voice_input")
        if audio:
            audio_bytes = audio.getvalue()
            digest = hashlib.sha1(audio_bytes).hexdigest()
            if digest == st.session_state.last_audio_digest:
                st.info("Audio already processed.")
            else:
                st.session_state.last_audio_digest = digest
                norm = _normalize_audio(audio_bytes)
                with sr.AudioFile(norm) as source:
                    audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    st.write(f"Transcript: {text}")
                    if text and _anchor(text):
                        _chat_response(text)
                except sr.UnknownValueError:
                    st.warning("Could not understand speech.")
                except Exception as exc:
                    st.error(f"Transcription failed: {exc}")

with col_text:
    st.markdown("### Type")
    with st.form("typed_memory"):
        typed_text = st.text_area("Memory prompt", key="typed_input")
        submitted = st.form_submit_button("Send")
        if submitted:
            text = typed_text.strip()
            if not text:
                st.warning("Enter some text first.")
            elif _anchor(text):
                if st.checkbox("Ask Gemini about this entry", key=f"ask_{time.time()}"):
                    reply = _chat_response(text)
                    st.info(reply)
                st.session_state.clear_typed = True

st.divider()
st.subheader("Chat History")
for role, content in st.session_state.chat_history[-20:]:
    st.markdown(f"**{role}:** {content}")

st.divider()
if st.button("Refresh ledger snapshot"):
    _load_ledger()
if st.session_state.ledger_state:
    st.json(st.session_state.ledger_state)
