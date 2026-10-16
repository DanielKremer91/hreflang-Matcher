from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from urllib.parse import urljoin

import requests
import trafilatura
import urllib3

from . import netguard
from .models import FetchResult

USER_AGENT = "ONE-hreflang-Matcher/1.0 (+https://onebeyondsearch.com)"
NO_TEXT = "kein Text extrahiert"
FETCH_CACHE_LIMIT = 5_000
FETCH_CACHE_TEXT_CHARS = 60_000   # = Maximum des Kürzungs-Sliders; hält den Seiten-Cache bei ca. 60 KB je Seite
MAX_BYTES = 5_000_000
MAX_REDIRECTS = 5
TIMEOUT = "Timeout"
BAD_REDIRECT = "Weiterleitung zu unzulässiger Adresse"
TOO_MANY_REDIRECTS = "zu viele Weiterleitungen"
_REDIRECT_CODES = {301, 302, 303, 307, 308}


def _too_big(max_bytes: int) -> str:
    return f"Seite zu groß (> {max_bytes / 1_000_000:g} MB)"


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


def fetch_one(
    url: str,
    timeout: float = 10.0,
    session: requests.Session | None = None,
    allow_private: bool = False,
    max_bytes: int = MAX_BYTES,
) -> tuple[str, str | None]:
    """Lädt eine Seite und extrahiert den Haupttext. Rückgabe (Text, None) oder ("", Grund).

    Jede Adresse (auch jede Weiterleitung, max. MAX_REDIRECTS) wird vorab per netguard geprüft; interne
    Adressen sind gesperrt, Loopback nur mit `allow_private`. Antwort max. `max_bytes`, Gesamtzeit je URL
    max. 3 × `timeout`."""
    own_session = session is None
    session = session or requests.Session()
    deadline = time.monotonic() + 3 * timeout
    current = url
    try:
        for hop in range(MAX_REDIRECTS + 1):
            reason = netguard.check_url(current, allow_loopback=allow_private)
            if reason:
                return "", reason if hop == 0 else BAD_REDIRECT
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "", TIMEOUT
            try:
                resp = session.get(current, headers={"User-Agent": USER_AGENT}, timeout=min(timeout, remaining),
                                   allow_redirects=False, stream=True)
            except requests.Timeout:
                return "", TIMEOUT
            except requests.RequestException as e:
                return "", f"Verbindungsfehler: {e.__class__.__name__}"
            with resp:
                location = resp.headers.get("Location")
                if resp.status_code in _REDIRECT_CODES and location:
                    current = urljoin(current, location)
                    continue
                return _read_page(resp, current, max_bytes, deadline)
        return "", TOO_MANY_REDIRECTS
    finally:
        if own_session:
            session.close()


def _iter_body(resp: requests.Response, chunk_size: int = 65536):
    """Body stückweise, sobald Daten da sind. read1 wartet nicht auf volle Blöcke; so greift die
    Gesamtfrist auch bei Servern, die die Antwort tröpfchenweise schicken."""
    read1 = getattr(resp.raw, "read1", None)
    if read1 is None:   # urllib3 < 2.3
        yield from resp.iter_content(chunk_size=chunk_size)
        return
    while True:
        chunk = read1(chunk_size, decode_content=True)
        if not chunk:
            return
        yield chunk


def _read_page(resp: requests.Response, url: str, max_bytes: int, deadline: float) -> tuple[str, str | None]:
    """Status und Content-Type prüfen, Body bis `max_bytes` lesen, Haupttext extrahieren."""
    if resp.status_code != 200:
        return "", f"HTTP {resp.status_code}"
    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype and "html" not in ctype:
        return "", f"kein HTML ({ctype})"
    try:
        if int(resp.headers.get("Content-Length") or 0) > max_bytes:
            return "", _too_big(max_bytes)
    except ValueError:
        pass
    chunks: list[bytes] = []
    size = 0
    try:
        for chunk in _iter_body(resp):
            size += len(chunk)
            if size > max_bytes:
                return "", _too_big(max_bytes)
            if time.monotonic() > deadline:
                return "", TIMEOUT
            chunks.append(chunk)
    except (requests.Timeout, urllib3.exceptions.ReadTimeoutError, TimeoutError):
        return "", TIMEOUT
    except (requests.RequestException, urllib3.exceptions.HTTPError, OSError) as e:
        return "", f"Verbindungsfehler: {e.__class__.__name__}"
    text = extract_main_text(b"".join(chunks), url)
    if not text:
        return "", NO_TEXT
    return text, None


def fetch_main_texts(
    urls: dict[int, str],
    concurrency: int = 5,
    timeout: float = 10.0,
    on_progress: Callable[[int, int], None] | None = None,
    session_factory: Callable[[], requests.Session] | None = None,
    cache: dict[str, str] | None = None,
    allow_private: bool = False,
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
            text, err = fetch_one(url, timeout=timeout, session=session, allow_private=allow_private)
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
                cache[urls[idx]] = text[:FETCH_CACHE_TEXT_CHARS]
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
