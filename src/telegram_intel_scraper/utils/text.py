from __future__ import annotations
import re


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def title_heuristic(text: str, max_len: int = 90) -> str:
    t = normalize_whitespace(text)
    if not t:
        return "Telegram message"
    first_line = t.split("\n", 1)[0].strip()
    if len(first_line) <= max_len:
        return first_line
    return first_line[: max_len - 1].rstrip() + "…"
