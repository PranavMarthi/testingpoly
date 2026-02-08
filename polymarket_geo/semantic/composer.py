from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComposedText:
    title_text: str
    description_text: str
    choices_text: str
    combined_text: str
    snippets: dict[str, str]


class TextComposer:
    """Builds per-field and combined text payloads for semantic retrieval."""

    @staticmethod
    def compose(title: str, description: str | None = None, choices: list[str] | None = None) -> ComposedText:
        t = (title or "").strip()
        d = (description or "").strip()
        c_items = [c.strip() for c in (choices or []) if c and c.strip()]
        c = " | ".join(c_items)

        parts = [f"title: {t}"]
        if d:
            parts.append(f"description: {d}")
        if c:
            parts.append(f"choices: {c}")

        return ComposedText(
            title_text=t,
            description_text=d,
            choices_text=c,
            combined_text="\n".join(parts),
            snippets={
                "title": t[:160],
                "description": d[:160],
                "choices": c[:160],
            },
        )
