import streamlit as st, requests, json
API = "https://dualsubstrate-commercial.fly.dev"

st.set_page_config(page_title="DualSubstrate Live Demo", layout="centered")
st.title("🎤 Live Prime-Ledger Demo (Browser STT)")

# ---------- browser STT (no Whisper) ----------
html = """
<script>
function startSTT(){
  const r=new (window.SpeechRecognition||webkitSpeechRecognition)();
  r.interimResults=false; r.lang='en-US';
  r.onresult=e=>{
    const t=e.results[0][0].transcript;
    fetch(`${window.parent.location.origin}/_stt_cb`, {
      method:'POST',
      body:JSON.stringify({text:t})
    });
  }; r.start();
}
</script>
<button onclick="startSTT()">Start talking</button>
"""
st.components.v1.html(html, height=100)

# ---------- callback route via Streamlit ----------
if "last" not in st.session_state:
    st.session_state.last = None

def stt_cb():
    st.session_state.last = json.loads(st.query_params["data"][0])
st.query_params.clear()
if st.query_params.get("data"):
    stt_cb()

# ---------- tiny helper ----------
def _hash(s: str):
    MAP = {w: p for p, w in enumerate("the and is to of a in that it with on for are as this was at be by an".split(), start=11)}
    return [{"prime": MAP.get(w.lower(), 2), "k": 1} for w in s.split() if w.isalpha()][:30]

# ---------- call engine ----------
if st.session_state.last:
    text = st.session_state.last["text"]
    r = requests.post(f"{API}/anchor", json={"entity": "demo_user", "factors": _hash(text)})
    st.write(r.json())

if st.button("Recall last"):
    resp = requests.get(f"{API}/retrieve?entity=demo_user").json()
    st.write(resp)

# ---------- metrics ----------
m = requests.get(f"{API}/metrics").json()
st.metric("Tokens saved", m["tokens_deduped"])
st.metric("Ledger integrity %", f"{m['ledger_integrity']*100:.1f} %")
