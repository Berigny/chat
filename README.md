[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dualsubstrate-demo-chassis.streamlit.app)

“Engine stays hardened on Fly.
Demo chassis lives on Streamlit Cloud (free).
No shared code, no shared infra – zero blast-radius.
5-minute spin-up, 30-second deploy loop.”

## Real metrics patch (engine)

Before the `/metrics` route in your Fly-hosted `main.py`, initialize:

```python
tokens_saved = 0
total_calls = 0
duplicates = 0
```

Inside `/anchor`:

```python
global tokens_saved, total_calls, duplicates
total_calls += 1
if _already_seen(factors):
    duplicates += 1
    tokens_saved += len(factors)
```

And expose real numbers from `/metrics`:

```python
return {
    "tokens_deduped": tokens_saved,
    "ledger_integrity": 1 - duplicates / total_calls if total_calls else 1.0,
}
```

## Prompt patterns that trigger ledger retrieval

The chat client will reach into the DualSubstrate RocksDB-backed memory when a typed
prompt meets any of the following heuristics:

- **Command prefixes:** `/q …`, `@ledger …`, or `::memory …`.
- **Keywords:** Any casing of the words `quote`, `verbatim`, `exact`, `recall`,
  `retrieve`, or the phrase `what did I say`.
- **Time references:** Natural-language date/time expressions (e.g. “what did I say
  last Tuesday at 3pm”) that can be parsed by `dateparser` or `parsedatetime`.
- **Semantic similarity:** Queries that score highly against the target intent “provide
  exact quotes from prior user statements” via the embedding-based similarity check.

If the combined signal exceeds the retrieval threshold, the app fetches the most recent
anchored memories from the Fly-hosted ledger and surfaces them directly in the UI.

Any retrieved transcript is sanitised before being forwarded to the LLM: assistant turns,
system notes, bullet and numbered quote residues, and other quote-mode artefacts are
stripped so the model only sees the human-authored passages. This keeps quote requests
grounded in the original user speech instead of echoing past bot answers.
