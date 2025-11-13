"""Prompt composition helpers that rely on memory selections."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from .memory_service import MemoryService, strip_ledger_noise


S1_PRIMES = {2, 3, 5, 7}
S2_PRIMES = {11, 13, 17, 19}


__all__ = ["PromptService", "create_prompt_service", "LEDGER_SNIPPET_LIMIT"]

logger = logging.getLogger(__name__)

LEDGER_SNIPPET_LIMIT = 5
LEDGER_SNIPPET_CHARS = 250
ATTACHMENT_SNIPPET_CHARS = 400
ATTACHMENT_LIMIT = 3


def _recent_chat_block(history: Sequence[tuple[str, str]], *, max_entries: int = 15) -> str | None:
    if not history:
        return None
    lines: list[str] = []
    for role, content in history[-max_entries:]:
        snippet = (content or "").strip().replace("\n", " ")
        if not snippet:
            continue
        if len(snippet) > 300:
            snippet = f"{snippet[:300]}…"
        lines.append(f"{role}: {snippet}")
    return "\n".join(lines) if lines else None


def _trim_snippet(text: str, limit: int) -> str:
    snippet = (text or "").strip().replace("\n", " ")
    if len(snippet) > limit:
        snippet = f"{snippet[:limit].rstrip()}…"
    return snippet


def _estimate_tokens(text: str) -> int:
    """Rudimentary token estimate (~4 chars/token)."""

    return max(1, len(text) // 4)


@dataclass
class PromptService:
    """Compose prompts that incorporate ledger memories and chat history."""

    memory_service: MemoryService

    def build_augmented_prompt(
        self,
        *,
        entity: str | None,
        question: str,
        schema: Mapping[int, Mapping[str, object]],
        chat_history: Sequence[tuple[str, str]],
        ledger_id: str | None = None,
        attachments: Iterable[Mapping[str, object]] | None = None,
        assembly: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
        time_window_hours: int = 168,
        since: int | None = None,
        until: int | None = None,
        quote_safe: bool = False,
    ) -> str:
        prompt_lines = [
            "You are a helpful assistant with access to a perfect, exact memory ledger.",
            "Your goal is to provide the most relevant, insightful, and natural response.",
            "Use the provided ledger memories and conversation history to understand the full context.",
            "You are free to synthesize information, draw conclusions, and ask clarifying questions.",
            "Your response should be helpful and conversational, not a rigid report.",
        ]

        if entity and assembly is None and hasattr(self.memory_service, "assemble_context"):
            assembly = self.memory_service.assemble_context(
                entity,
                ledger_id=ledger_id,
                k=LEDGER_SNIPPET_LIMIT,
                quote_safe=quote_safe,
                since=since,
            )

        if entity:
            memories: list[dict] = []
            if not assembly or not any(assembly.get(section) for section in ("summaries", "bodies", "claims")):
                memories = self.memory_service.select_context(
                    entity,
                    question,
                    schema,
                    ledger_id=ledger_id,
                    limit=LEDGER_SNIPPET_LIMIT,
                    time_window_hours=time_window_hours,
                    since=since,
                    until=until,
                )
        else:
            memories = []

        summary_entries: list[Mapping[str, object]] = []
        body_entries: list[Mapping[str, object]] = []
        claim_entries: list[Mapping[str, object]] = []
        if isinstance(assembly, Mapping):
            summary_entries = [dict(entry) for entry in assembly.get("summaries", []) if isinstance(entry, Mapping)]
            body_entries = [dict(entry) for entry in assembly.get("bodies", []) if isinstance(entry, Mapping)]
            claim_entries = [dict(entry) for entry in assembly.get("claims", []) if isinstance(entry, Mapping)]

        summary_entries.sort(key=lambda entry: entry.get("timestamp") or 0, reverse=True)
        body_entries.sort(key=lambda entry: entry.get("timestamp") or 0, reverse=True)
        claim_entries.sort(key=lambda entry: entry.get("timestamp") or 0, reverse=True)

        summary_lines: list[str] = []
        for entry in summary_entries[:LEDGER_SNIPPET_LIMIT]:
            summary = _trim_snippet(str(entry.get("summary") or ""), LEDGER_SNIPPET_CHARS)
            if not summary:
                continue
            prime = entry.get("prime") if isinstance(entry.get("prime"), int) else None
            title = entry.get("title") or (f"Prime {prime}" if prime else None)
            tags = [tag for tag in entry.get("tags", ()) if isinstance(tag, str) and tag]
            tag_block = f" [tags: {', '.join(tags[:3])}]" if tags else ""
            timestamp = entry.get("timestamp") if isinstance(entry.get("timestamp"), (int, float)) else None
            ts_label = (
                datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M")
                if timestamp
                else None
            )
            label_parts = [part for part in (ts_label, title) if part]
            label = " | ".join(label_parts) if label_parts else title or ts_label or "Ledger"
            summary_lines.append(f"- {label}{tag_block}: {summary}")

        body_lines: list[str] = []
        for entry in body_entries[:LEDGER_SNIPPET_LIMIT]:
            if entry.get("quote_safe") is False:
                continue
            prime = entry.get("prime") if isinstance(entry.get("prime"), int) else None
            title = entry.get("title") or entry.get("summary") or (f"Prime {prime}" if prime else None)
            body_source = entry.get("body")
            if isinstance(body_source, Sequence) and not isinstance(body_source, (str, bytes)):
                body_chunks = [chunk for chunk in body_source if isinstance(chunk, str)]
            elif isinstance(body_source, str):
                body_chunks = [body_source]
            else:
                body_chunks = []
            chunk_text = None
            for chunk in body_chunks:
                if isinstance(chunk, str) and chunk.strip():
                    chunk_text = chunk.strip()
                    break
            if not chunk_text:
                summary_text = entry.get("summary") if isinstance(entry.get("summary"), str) else None
                chunk_text = summary_text.strip() if summary_text else None
            if not chunk_text:
                continue
            snippet = _trim_snippet(chunk_text, LEDGER_SNIPPET_CHARS)
            if not snippet:
                continue
            tags = [tag for tag in entry.get("tags", ()) if isinstance(tag, str) and tag]
            tag_block = f" [tags: {', '.join(tags[:3])}]" if tags else ""
            body_lines.append(f"- {title or f'Prime {prime}'}{tag_block}: {snippet}")

        claim_lines: list[str] = []
        for entry in claim_entries[:LEDGER_SNIPPET_LIMIT]:
            claim_text = entry.get("claim") or entry.get("summary")
            if not isinstance(claim_text, str):
                continue
            snippet = _trim_snippet(claim_text, LEDGER_SNIPPET_CHARS)
            if not snippet:
                continue
            prime = entry.get("prime") if isinstance(entry.get("prime"), int) else None
            title = entry.get("title") or (f"Prime {prime}" if prime else "Claim")
            tags = [tag for tag in entry.get("tags", ()) if isinstance(tag, str) and tag]
            tag_block = f" [tags: {', '.join(tags[:3])}]" if tags else ""
            claim_lines.append(f"- {title}{tag_block}: {snippet}")

        structured_rendered = False
        if summary_lines:
            structured_rendered = True
            prompt_lines.append("\n--- Ledger S2 Summaries ---")
            prompt_lines.extend(summary_lines)
        if claim_lines:
            structured_rendered = True
            prompt_lines.append("\n--- Ledger S2 Claims ---")
            prompt_lines.extend(claim_lines)
        if body_lines:
            structured_rendered = True
            prompt_lines.append("\n--- Ledger Bodies ---")
            prompt_lines.extend(body_lines)

        if not structured_rendered and memories:
            legacy_s1: list[str] = []
            legacy_s2: list[str] = []
            legacy_bodies: list[str] = []
            for entry in memories:
                prime = entry.get("prime") if isinstance(entry.get("prime"), int) else None
                raw_summary = (
                    entry.get("summary")
                    or entry.get("text")
                    or entry.get("snippet")
                    or entry.get("_structured_text")
                    or entry.get("_sanitized_text")
                )
                if isinstance(raw_summary, Sequence) and not isinstance(raw_summary, (str, bytes)):
                    for chunk in raw_summary:
                        if isinstance(chunk, str) and chunk.strip():
                            raw_summary = chunk
                            break
                summary = _trim_snippet(str(raw_summary or ""), LEDGER_SNIPPET_CHARS)
                title = entry.get("title") or (f"Prime {prime}" if prime else None)
                tags = [tag for tag in entry.get("tags", ()) if tag]
                tag_block = f" [tags: {', '.join(tags[:3])}]" if tags else ""
                timestamp = entry.get("timestamp")
                ts_label = (
                    datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M")
                    if timestamp
                    else None
                )
                label_parts = [part for part in (ts_label, title) if part]
                label = " | ".join(label_parts) if label_parts else title or ts_label or "Ledger"
                if summary:
                    line = f"- {label}{tag_block}: {summary}"
                    if prime in S1_PRIMES:
                        legacy_s1.append(line)
                    elif prime in S2_PRIMES:
                        legacy_s2.append(line)
                    else:
                        legacy_s2.append(line)

                body_source = entry.get("body")
                if isinstance(body_source, Sequence) and not isinstance(body_source, (str, bytes)):
                    body_chunks = [chunk for chunk in body_source if isinstance(chunk, str)]
                elif isinstance(body_source, str):
                    body_chunks = [body_source]
                else:
                    body_chunks = []
                for chunk in body_chunks[:LEDGER_SNIPPET_LIMIT]:
                    snippet = _trim_snippet(chunk, LEDGER_SNIPPET_CHARS)
                    if snippet:
                        label = title or f"Prime {prime}" or "Ledger"
                        legacy_bodies.append(f"- {label}: {snippet}")

            if legacy_s1:
                structured_rendered = True
                prompt_lines.append("\n--- Ledger S1 Slots ---")
                prompt_lines.extend(legacy_s1[:LEDGER_SNIPPET_LIMIT])
            if legacy_s2:
                structured_rendered = True
                prompt_lines.append("\n--- Ledger S2 Summaries ---")
                prompt_lines.extend(legacy_s2[:LEDGER_SNIPPET_LIMIT])
            if legacy_bodies:
                structured_rendered = True
                prompt_lines.append("\n--- Ledger Bodies ---")
                prompt_lines.extend(legacy_bodies[:LEDGER_SNIPPET_LIMIT])

        if not structured_rendered:
            prompt_lines.append("\n--- Ledger Memories ---")
            if memories:
                for entry in memories[:LEDGER_SNIPPET_LIMIT]:
                    sanitized = entry.get("_sanitized_text") or strip_ledger_noise((entry.get("text") or "").strip())
                    if not sanitized:
                        continue
                    timestamp = entry.get("timestamp")
                    if timestamp:
                        ts = (
                            datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M")
                            if timestamp
                            else "No timestamp"
                        )
                    else:
                        ts = "No timestamp"
                    prompt_lines.append(f"[{ts}] {_trim_snippet(sanitized, LEDGER_SNIPPET_CHARS)}")
            else:
                prompt_lines.append("(No specific memories matched the query, but the full ledger is available.)")

        chat_block = _recent_chat_block(chat_history)
        if chat_block:
            prompt_lines.append("\n--- Recent Conversation ---")
            prompt_lines.append(chat_block)

        if attachments:
            prompt_lines.append("\n--- Attachments ---")
            for attachment in list(attachments)[:ATTACHMENT_LIMIT]:
                name = attachment.get("name", "attachment")
                snippet = _trim_snippet(attachment.get("text", ""), ATTACHMENT_SNIPPET_CHARS)
                if snippet:
                    prompt_lines.append(f"[{name}] {snippet}")

        prompt_lines.append("\n--- Your Turn ---")
        prompt_lines.append(f"User's request: {question}")
        prompt_lines.append("Your response:")
        prompt = "\n".join(prompt_lines)
        token_estimate = _estimate_tokens(prompt)
        if token_estimate > 15000:
            logger.warning("Augmented prompt approx %s tokens; risk of context overflow.", token_estimate)
        else:
            logger.debug("Augmented prompt approx %s tokens.", token_estimate)
        return prompt

    def build_capabilities_block(
        self,
        *,
        entity: str | None,
        schema: Mapping[int, Mapping[str, object]],
        chat_history: Sequence[tuple[str, str]],
        prime_semantics: str,
        ledger_id: str | None = None,
        last_anchor_error: str | None = None,
        context_limit: int = 5,
    ) -> str:
        anchor_note = (
            "Latest anchor failed; new text may not be stored."
            if last_anchor_error
            else "Latest anchor succeeded."
        )

        lines = [
            "Capabilities & Instructions:",
            "- Cite only the memories listed below; do not invent new quotes.",
            "- If the ledger has no entry for the requested topic, say so explicitly.",
            "- Anchors succeed only when the ledger accepts the factors; report failures if they occur.",
            "",
            anchor_note,
            "",
            prime_semantics,
        ]

        if entity:
            ledger_block = self.memory_service.render_context_block(
                entity,
                schema,
                ledger_id=ledger_id,
                limit=context_limit,
            )
        else:
            ledger_block = ""

        if ledger_block:
            lines.extend(["", "Recent ledger memories:", ledger_block])

        recent_chat = _recent_chat_block(chat_history)
        if recent_chat:
            lines.extend(["", "Recent chat summary:", recent_chat])

        return "\n".join(lines)


def create_prompt_service(memory_service: MemoryService) -> PromptService:
    return PromptService(memory_service=memory_service)
