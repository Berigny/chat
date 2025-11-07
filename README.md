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
