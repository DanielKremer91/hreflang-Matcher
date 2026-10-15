from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import requests
import trafilatura

from .models import FetchResult

USER_AGENT = "ONE-hreflang-Matcher/1.0 (+https://onebeyondsearch.com)"
NO_TEXT = "kein Text extrahiert"


def extract_main_text(html: str, url: str | None = None) -> str:
    """Haupttext ohne Navigation/Footer/Kommentare/Tabellen; leer, wenn nichts extrahiert wurde."""
    if not html:
        return ""
    text = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=False, favor_precision=True, fast=True
    )
    return (text or "").strip()


def fetch_one(url: str, timeout: float = 10.0, session: requests.Session | None = None) -> tuple[str, str | None]:
    """Lädt eine Seite und extrahiert den Haupttext. Rückgabe (Text, None) oder ("", Grund)."""
    own_session = session is None
    session = session or requests.Session()
    try:
        try:
            resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, allow_redirects=True)
        except requests.Timeout:
            return "", "Timeout"
        except requests.RequestException as e:
            return "", f"Verbindungsfehler: {e.__class__.__name__}"
        if resp.status_code != 200:
            return "", f"HTTP {resp.status_code}"
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype and "html" not in ctype:
            return "", f"kein HTML ({ctype})"
        text = extract_main_text(resp.text, url)
        if not text:
            return "", NO_TEXT
        return text, None
    finally:
        if own_session:
            session.close()


def fetch_main_texts(
    urls: dict[int, str],
    concurrency: int = 5,
    timeout: float = 10.0,
    on_progress: Callable[[int, int], None] | None = None,
    session_factory: Callable[[], requests.Session] | None = None,
) -> FetchResult:
    """Parallel (max. `concurrency`) laden; Ergebnis je positionalem Zeilenindex."""
    session_factory = session_factory or requests.Session
    result = FetchResult()
    items = list(urls.items())
    total = len(items)
    done = 0

    def run(item: tuple[int, str]) -> tuple[int, str, str | None]:
        idx, url = item
        session = session_factory()
        try:
            text, err = fetch_one(url, timeout=timeout, session=session)
        except Exception as e:  # eine defekte Seite darf nie den ganzen Lauf abbrechen
            text, err = "", f"Fehler: {e.__class__.__name__}"
        finally:
            session.close()
        return idx, text, err

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for idx, text, err in pool.map(run, items):
            if err is None:
                result.texts[idx] = text
            else:
                result.errors[idx] = err
            done += 1
            if on_progress:
                on_progress(done, total)
    return result
