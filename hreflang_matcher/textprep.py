from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

# Reihenfolge = Priorität des Vorschlags.
TEXT_COL_HINTS = ["main content", "content extraction", "body", "text", "title", "h1", "meta description"]
NO_TEXT_REASON = "kein Text in den gewählten Spalten"
_WS = re.compile(r"\s+")


def suggest_text_columns(df: pd.DataFrame) -> list[str]:
    """Spalten, deren Name einen Texthinweis enthält, in Hinweis-Reihenfolge, ohne Duplikate."""
    out: list[str] = []
    for hint in TEXT_COL_HINTS:
        for c in df.columns:
            if hint in str(c).lower() and c not in out:
                out.append(c)
    return out


def truncate(text, max_chars: int) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = _WS.sub(" ", str(text)).strip()
    return s[:max_chars]


def build_texts(df: pd.DataFrame, cols: list[str], max_chars: int) -> tuple[dict[int, str], dict[int, str]]:
    """Positionaler Zeilenindex -> zusammengesetzter, gekürzter Text; leere Zeilen -> Grund."""
    texts: dict[int, str] = {}
    errors: dict[int, str] = {}
    for pos, (_, r) in enumerate(df.iterrows()):
        parts = [truncate(r[c], max_chars) for c in cols]
        joined = "\n\n".join(p for p in parts if p)[:max_chars]
        if joined:
            texts[pos] = joined
        else:
            errors[pos] = NO_TEXT_REASON
    return texts, errors


def estimate_tokens(texts: Iterable[str]) -> int:
    return sum(len(t) for t in texts) // 4
