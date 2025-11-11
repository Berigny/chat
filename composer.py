"""Response composer helpers."""

from __future__ import annotations

import hashlib
from typing import Dict, List


def render_snippet(snippet: Dict) -> str:
    text = (snippet or {}).get("text") or ""
    snippet_type = (snippet or {}).get("type")
    source_url = (snippet or {}).get("source_url")
    if snippet_type == "quote" and source_url:
        return f"“{text}”"
    return text


def compose_summary(shards: List[Dict], question: str) -> str:
    if not shards:
        return "I couldn’t find enough grounded material to answer that yet."

    parts: List[str] = []
    for shard in shards:
        theses = shard.get("theses") or []
        parts.extend(theses)
        for snippet in shard.get("snippets") or []:
            parts.append(render_snippet(snippet))

    summary = " ".join(part.strip() for part in parts if part).strip()
    if not summary:
        summary = "I couldn’t find enough grounded material to answer that yet."

    provenance = shards[0].get("provenance", {}) if shards else {}
    title = provenance.get("title", "Unknown source")
    author = provenance.get("author", "Unknown author")
    year = provenance.get("year", "n.d.")
    signature = hashlib.sha1(f"{question}|{summary}".encode("utf-8")).hexdigest()[:12]
    footer = f"Sources: {title} ({author} {year}), {len(shards)} shards, sig {signature}"
    return f"{summary}\n\n{footer}"

