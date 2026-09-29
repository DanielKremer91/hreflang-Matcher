from __future__ import annotations

import hashlib

MODE_FILE = "file"
MODE_GROUP = "group"


def source_key(mode: str, file_id: str | None, raw: bytes, code: str | None) -> str:
    """Stabiler Widget-Key-Präfix je Sprachvariante, unabhängig von der Listenposition.

    Modus "file" (eine Datei je Sprache): Key aus der Upload-ID, ersatzweise aus dem Datei-Hash.
    Modus "group" (Gesamtdatei per Muster): Key aus dem hreflang-Code der Mustertabelle.
    """
    if mode == MODE_FILE:
        ident = file_id or hashlib.md5(raw).hexdigest()[:12]
        return f"file_{ident}"
    if mode == MODE_GROUP:
        if not code:
            raise ValueError("Für Muster-Gruppen wird ein hreflang-Code benötigt.")
        return f"grp_{code}"
    raise ValueError(f"Unbekannter Modus: {mode}")
