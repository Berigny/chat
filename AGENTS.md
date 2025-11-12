# Dual-Substrate Commercial: Purpose & Engine

## Purpose
This repository demonstrates a **working prototype of a new AI engine** based on the **dual-substrate architecture** introduced in *AI’s Brute Force Trap (2025)*:contentReference[oaicite:0]{index=0}.  
Rather than relying on brute-force scaling—“adding more pistons and petrol”—this design re-engineers the foundation of intelligence itself, coupling:

- a **continuous ℝ-domain** for adaptive learning and pattern recognition, and  
- a **discrete ℚₚ-domain** (prime ledger) for exact symbolic memory.

The goal is to demonstrate that *intelligence can evolve beyond brute force* toward an engine that is **faster, greener, and truer to intent**:contentReference[oaicite:1]{index=1}.

This prototype uses:

- **Fly.io** — lightweight distributed deployment layer.  
- **RocksDB** — embedded **prime-ledger store** providing exact recall, persistence, and arithmetic identity anchoring.  
- **Python + Streamlit** — rapid UI scaffolding and observability.

Together these components form a practical implementation of the paper’s thesis: *replace continuous approximation with meaning-preserving computation.*

## Streamlit Apps (Work in Progress)

1. **`chat_demo_app.py` — Investor / Prototype Demo**  
   Presents a modern LLM-style interface powered by the prime-ledger engine.  
   - Demonstrates **exact memory** and **identity persistence** across sessions.  
   - Designed for investors and interested partners to explore live coherence behaviour.

2. **`admin_app.py` — Ledger Administration Console**  
   Operator dashboard for creating and managing ledgers, inspecting prime signatures, and verifying persistence metrics.  
   - Internal tool for analysing ledger flows and performance.  
   - Future expansion: multi-tenant orchestration and audit view.

## Target User Experience
The intended UX should **feel like interacting with a contemporary LLM**, yet every exchange is backed by **exact memory and identity persistence**.  
Each message round-trips through the ledger, this gives users:

- Stable recall of previous context  
- Deterministic identity across sessions  
- Transparent reasoning trails visible via the admin console  

This fulfils the vision outlined in *AI’s Brute Force Trap*: a transition from energy-intensive scaling to **meaning-preserving, arithmetic-anchored intelligence**:contentReference[oaicite:2]{index=2}.

---

# Repository Guidelines

## Project Structure & Module Organization
The Streamlit surfaces (`admin_app.py`, `chat_demo_app.py`) live at repo root; they call into shared helpers such as `composer.py`, `flow_safe.py`, `intent_map.py`, `ledger_store.py`, and `validators.py` for prime-ledger logic. Tests reside in `tests/` (mirroring the module names), while marketing assets stay in the root as `.png` files. Keep new utilities colocated with their consumers; for example, UI-only helpers belong beside the Streamlit file, whereas ledger or schema helpers should sit with the other backend modules.

## Build, Test, and Development Commands
Run `make install` once to install `requirements.txt`. Use `make run` (or `make demo`) for the chat demo surface and `make chat` for the admin console.  
`pytest tests/test_anchor.py -k anchor` (or simply `pytest tests`) keeps ledger flows green; launch the Streamlit apps only when you need interactive verification.

## Coding Style & Naming Conventions
Follow idiomatic Python 3 with 4-space indentation, `snake_case` functions, and UpperCamelCase classes. Keep module-level constants uppercase (`API`, `ENTITY`) and stash configuration near the top of each file as in `chat_demo_app.py`. Type hints already exist in composer/validator modules—extend them when touching call sites. Prefer short, single-purpose functions and include docstrings when the intent is not obvious, especially around prime/ledger math.

## Testing Guidelines
Pytest discovers files named `test_*.py`; mimic the existing fixture pattern in `tests/conftest.py` and add scenario-focused tests next to the module you change (`tests/test_flow_safe.py` is a good template). When you add retrieval heuristics or new Streamlit callbacks, cover the pure logic in unit tests and document any manual UI verifications in the PR. Aim to keep new branches at or above the current coverage by testing both happy-path anchors and error handling.

## Commit & Pull Request Guidelines
Commits should use short, present-tense subjects (e.g., `Refine admin anchoring flow`) and group related changes so reviewers can bisect easily. Every PR should include: a problem summary, bullet list of changes, explicit test commands (`pytest`, `make run` smoke), and screenshots/GIFs when UI changes touch Streamlit. Link issues or TODOs inline if they exist, and call out any follow-up work so the ledger roadmap stays traceable.

## Security & Configuration Tips
Never hardcode API keys; rely on `st.secrets['DUALSUBSTRATE_API_KEY']` or the `DUALSUBSTRATE_API_KEY` environment variable, mirroring the existing Streamlit apps. When sharing demo URLs, reset the key or use the bundled `demo-key`. Treat ledger endpoints as production-facing—avoid logging factor payloads beyond DEBUG level and confirm HTTPS URLs before committing.

### Low security authentication for streamlit: 
- Demo_dev / Developer: kU7JLagw
- Demo_new / Demo User: anQDX18C
