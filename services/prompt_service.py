"""Prompt composition helpers that rely on memory selections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from .memory_service import MemoryService, strip_ledger_noise


__all__ = ["PromptService", "create_prompt_service"]


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
        time_window_hours: int = 168,
        since: int | None = None,
        until: int | None = None,
    ) -> str:
        prompt_lines = [
            "You are a helpful assistant with access to a perfect, exact memory ledger.",
            "Your goal is to provide the most relevant, insightful, and natural response.",
            "Use the provided ledger memories and conversation history to understand the full context.",
            "You are free to synthesize information, draw conclusions, and ask clarifying questions.",
            "Your response should be helpful and conversational, not a rigid report.",
        ]

        if entity:
            memories = self.memory_service.select_context(
                entity,
                question,
                schema,
                ledger_id=ledger_id,
                limit=10,
                time_window_hours=time_window_hours,
                since=since,
                until=until,
            )
        else:
            memories = []

        if memories:
            prompt_lines.append("\n--- Ledger Memories (most relevant first) ---")
            for entry in memories:
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
                prompt_lines.append(f"[{ts}] {sanitized}")
        else:
            prompt_lines.append("\n--- Ledger Memories ---")
            prompt_lines.append("(No specific memories matched the query, but the full ledger is available.)")

        chat_block = _recent_chat_block(chat_history)
        if chat_block:
            prompt_lines.append("\n--- Recent Conversation ---")
            prompt_lines.append(chat_block)

        if attachments:
            prompt_lines.append("\n--- Attachments ---")
            for attachment in attachments:
                name = attachment.get("name", "attachment")
                snippet = (attachment.get("text") or "").strip()[:1000]
                if snippet:
                    prompt_lines.append(f"[{name}] {snippet}...")

        prompt_lines.append("\n--- Your Turn ---")
        prompt_lines.append(f"User's request: {question}")
        prompt_lines.append("Your response:")
        return "\n".join(prompt_lines)


def create_prompt_service(memory_service: MemoryService) -> PromptService:
    return PromptService(memory_service=memory_service)
