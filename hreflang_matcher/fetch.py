from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import requests
import trafilatura

from .models import FetchResult

USER_AGENT = "ONE-hreflang-Matcher/1.0 (+https://onebeyondsearch.com)"
NO_TEXT = "kein Text extrahiert"
FETCH_CACHE_LIMIT = 5_000


def extract_main_text(html: str | bytes, url: str | None = None) -> str:
    """Haupttext ohne Navigation/Footer/Kommentare/Tabellen; leer, wenn nichts extrahiert wurde.

    Bytes werden bevorzugt: Trafilatura erkennt das Encoding aus Meta-Tag und Inhalt. requests würde
    `text/html` ohne charset als ISO-8859-1 dekodieren und Umlaute zerstören."""
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
        text = extract_main_text(resp.content, url)
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
    cache: dict[str, str] | None = None,
) -> FetchResult:
    """Parallel (max. `concurrency`) laden; Ergebnis je positionalem Zeilenindex.

    `cache` (URL -> Haupttext) überspringt bereits geladene Seiten; jeder erfolgreiche Abruf wird sofort
    eingetragen, damit ein abgebrochener Lauf keine Arbeit verliert (FIFO, max. FETCH_CACHE_LIMIT)."""
    session_factory = session_factory or requests.Session
    cache = {} if cache is None else cache
    result = FetchResult()
    items: list[tuple[int, str]] = []
    for idx, url in urls.items():
        hit = cache.get(url)
        if hit is not None:
            result.texts[idx] = hit
        else:
            items.append((idx, url))
    total = len(urls)
    done = total - len(items)

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

    # Ohne `with`: Bei einem Abbruch (z. B. Streamlit-Rerun) nicht auf wartende Abrufe warten.
    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
    try:
        futures = [pool.submit(run, item) for item in items]
        for fut in as_completed(futures):
            idx, text, err = fut.result()
            if err is None:
                result.texts[idx] = text
                cache[urls[idx]] = text
            else:
                result.errors[idx] = err
            done += 1
            if on_progress:
                on_progress(done, total)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        while len(cache) > FETCH_CACHE_LIMIT:
            cache.pop(next(iter(cache)))
    return result
