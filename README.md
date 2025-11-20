[![Live Demo](not ready)

“Engine stays hardened on Fly.
Demo chassis lives on Streamlit Cloud (free).
No shared code, no shared infra – zero blast-radius.
5-minute spin-up, 30-second deploy loop.”

## Dual-substrate architecture & deployment

- **Signal split:** A continuous ℝ substrate handles gradient-driven pattern search and embedding alignment, while the discrete ℚₚ prime ledger preserves symbolic identity, exact factors, and arithmetic anchoring. The two streams stay synchronized through shared entity IDs, mirrored factor slots, and cross-projections that keep approximate vectors tethered to stable prime signatures.
- **Fly.io layout:** The Fly-hosted engine exposes `/anchor`, `/memories`, `/rotate`, `/traverse`, and `/inference/state`, backed by RocksDB for the prime ledger and a lightweight Python service for continuous updates. Stateless Fly machines fan out the API, while ledger volumes stay colocated for low-latency retrieval and Möbius refresh cycles.
- **Front-ends:** Streamlit surfaces (`chat_demo_app.py` for the demo/chat UX and `admin_app.py` for operators) call the Fly engine directly. They live in this [Chat repo](./) alongside shared helpers and can be redeployed independently of the Fly layer.

### Q&A

- **Is the arithmetic true p-adic or ultrametric-inspired?** The ledger operates on an ultrametric-inspired prime-factor lattice rather than full field-complete p-adic arithmetic; primes anchor identity and distance while avoiding heavy p-adic carries.
- **How are ℝ and ℚₚ composed?** The system treats them as a direct product: continuous embeddings stay in ℝ, discrete factors in ℚₚ, and prompts stitch the two via tensor-style projections that inject prime weights into the continuous context window without collapsing either space.
- **What is the normalization mechanism?** Continuous updates use gradient-style normalization (token and energy scaling), while the discrete ledger leans on variational balancing of prime deltas to keep slot weights bounded without erasing exactness.
- **Which emergent behaviors are observed?** Stable recall across sessions, clustering of recurring entities along shared prime slots, lower prompt churn due to deduped anchors, and faster retrieval routes when Möbius rotations keep the lattice fresh.
- **What happens if you relax K_Unity = 1?** Allowing K_Unity to drift above or below 1 introduces amplitude skew between continuous and discrete streams—ℝ vectors over-amplify or underweight prime guidance—so coherence falls off and retrieval paths become noisy until renormalized.

**Stage:** integration

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

The chat client will reach into the DualSubstrate remote ledger memory when a typed
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

## Prime-aware anchoring for external agents

To ensure every external agent spreads factor weights across the full eight-prime topology,
hand it the following framing before it begins logging memories:

```
You are using the DualSubstrate ledger. For each utterance, extract up to one entry for each slot:

- Prime 2: subject or speaker (“Alice”, “I”)
- Prime 3: primary action (“met”, “emailed”, “call”)
- Prime 5: object or recipient (“Bob”, “the investor”)
- Prime 7: location or channel (“NYC office”, “Zoom”)
- Prime 11: time anchor (“today 7:01pm”, “tomorrow morning”)
- Prime 13: intent or outcome (“hire them”, “ship release”)
- Prime 17: supporting context (risks, blockers, dependencies)
- Prime 19: sentiment/priority (“urgent”, “blocked”, “excited”)

When you call POST /anchor, send a factors array containing every slot you filled:
[
  {"prime":2,"delta":1},
  {"prime":3,"delta":1},
  {"prime":5,"delta":1},
  ...
]
If a slot has no info, omit it (or use delta 0). Include the raw transcript in text.

After anchoring, call GET /memories?entity=demo_user&limit=20 to retrieve the exact strings you just logged.
```

### Ready-to-test script

```bash
curl -X POST https://dualsubstrate-commercial.fly.dev/anchor \
  -H "x-api-key: demo-key" \
  -H "Content-Type: application/json" \
  -d '{
        "entity":"demo_user",
        "text":"Met Priya at the NYC office to finalize Tuesday’s launch plan.",
        "factors":[
          {"prime":2,"delta":1},
          {"prime":3,"delta":1},
          {"prime":5,"delta":1},
          {"prime":7,"delta":1},
          {"prime":11,"delta":1},
          {"prime":13,"delta":1},
          {"prime":19,"delta":1}
        ]
      }'
```

### Implementation notes

- `/memories` now reflects the exact anchored strings and timestamps, so `/q` quote
  requests can filter the JSON directly.
- If you need automatic slot extraction, place a lightweight mapper (regex or spaCy)
  ahead of the agent to pre-fill the prime assignments; the agent merely confirms or edits.
- This framing keeps the ledger balanced across all primes and makes topic-specific
  retrieval trivial, because every anchor contains structured who/what/where/when signals.

## Möbius transform CTA

The Fly engine exposes `/rotate`, which performs the quaternion pack/rotate/unpack cycle
to regenerate the exponent lattice. You can call it directly:

```bash
curl -X POST https://dualsubstrate-commercial.fly.dev/rotate \
  -H "x-api-key: demo-key" \
  -H "Content-Type: application/json" \
  -d '{"entity":"demo_user","axis":[0,0,1],"angle":1.0472}'
```

In the Streamlit UI we expose a `♾️ Möbius Transform` button beneath the Metrics card:

```python
if st.button("♾️ Möbius Transform", help="Reproject the exponent lattice"):
    payload = {"entity": ENTITY, "axis": (0.0, 0.0, 1.0), "angle": 1.0472}
    resp = requests.post(f"{API}/rotate", json=payload, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    st.success(
        f"Rotated lattice. Δenergy = {data['energy_cycles']}, "
        f"checksum {data['original_checksum']} → {data['rotated_checksum']}."
    )
```

Behind the scenes `/rotate` pulls the factor vector, runs the quaternion Möbius rotation,
anchors the new vector via `anchor_batch`, and returns before/after checksums plus energy
cycles. Triggering it from the CTA keeps the lattice fresh without shell access.

## Traversal & inference observability

The commercial surface now exposes `/traverse` and `/inference/state` so operators can audit
how the ledger walks across primes and which inference tasks are active. Both Streamlit apps
surface the data in dedicated tabs:

- **Traversal Paths** renders the top weighted walks returned by `/traverse`, with per-node
  labels and weights for quick debugging of slot coverage.
- **Inference Status** shows the active job, queued work, recent completions, and telemetry
  emitted by `/inference/state`.

Older deployments that do not implement the new endpoints simply hide the tabs and emit a
“not available” notice. Prompt construction also threads the traversal and inference summaries
into the augmented prompt so the LLM can acknowledge long-running jobs without manual probing.

## RocksDB probe verification

Point the UI at your live ledger directory by exporting `ROCKSDB_DATA_PATH` (defaults to
`/app/rocksdb-data`). The **Memory & Inference** tab now exposes a **RocksDB probe** form that
invokes `tests.rocksdb_probe.run_probe()` directly from the Streamlit surface:

1. Enter the entity ID, a synthetic prompt (pre-filled with “kangaroo neon laser”), and the prime
   pattern you expect the ledger to traverse (e.g. `2*3*5*7`).
2. Click **Run RocksDB probe** to anchor the prompt via the embedded RocksDB client and walk the
   keyspace using the multiplicative pattern.
3. Expand the JSON results panel and compare the returned key/value pairs with
   `MemoryService.memory_lookup()` output (also surfaced via the sidebar “Raw Ledger” expander).

This workflow makes it easy to validate whether the multiplicative primes you supply at anchor time
produce the same storage hits you see via the API before running a full end-to-end chat session.

## Manual smoke test – S2 promotion button

1. Launch `chat_demo_app.py` (`make demo`) and authenticate as any demo account so the Connectivity Debug tab is visible.
2. Open **Connectivity Debug** → click **Promote to S2 tier**.
3. Confirm a success toast appears, the recall mode switcher updates to slots/S2, and no request exceptions are surfaced in the sidebar log.
