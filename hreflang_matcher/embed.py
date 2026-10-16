from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
import requests

from . import netguard
from .models import EmbeddingConfig, EmbedResult

PRESETS: dict[str, dict] = {
    "OpenAI": {"base_url": "https://api.openai.com/v1",
               "models": ["text-embedding-3-small", "text-embedding-3-large"], "local": False, "needs_key": True},
    "Ollama (lokal)": {"base_url": "http://localhost:11434/v1", "models": ["bge-m3"], "local": True, "needs_key": False},
    "LM Studio (lokal)": {"base_url": "http://localhost:1234/v1", "models": [], "local": True, "needs_key": False},
    "Eigener OpenAI-kompatibler Endpunkt": {"base_url": "", "models": [], "local": False, "needs_key": False},
}
CACHE_LIMIT = 20_000
RETRY_SLEEPS = (2, 4, 8)
FAIL_PREFIX = "Embedding-Anfrage fehlgeschlagen: "
_LENGTH_RE = re.compile(r"context length|maximum context|too long|max.*tokens|input length", re.I)


class EmbeddingError(Exception):
    """Nicht wiederholbarer Fehler der Embedding-Anfrage."""


class RetryableError(EmbeddingError):
    """429, 5xx oder Netzwerkfehler."""


def cache_key(cfg: EmbeddingConfig, text: str) -> str:
    return hashlib.sha1(f"{cfg.model}|{cfg.base_url}|{text}".encode("utf-8")).hexdigest()


def _short(resp: requests.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        body = None
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        msg = err.get("message") or resp.text
    elif isinstance(err, str):
        msg = err
    else:
        msg = resp.text
    return str(msg).strip().replace("\n", " ")[:200]


def _post_once(cfg: EmbeddingConfig, texts: list[str], session: requests.Session) -> list[np.ndarray]:
    url = cfg.base_url.rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    try:
        resp = session.post(url, json={"model": cfg.model, "input": texts}, headers=headers, timeout=cfg.timeout)
    except requests.Timeout as e:
        raise RetryableError(f"Timeout: {e}") from e
    except (requests.exceptions.MissingSchema, requests.exceptions.InvalidSchema,
            requests.exceptions.InvalidURL) as e:
        raise EmbeddingError(f"Ungültige Base-URL: {cfg.base_url!r}") from e
    except requests.RequestException as e:
        raise RetryableError(f"Verbindungsfehler: {e.__class__.__name__}") from e
    if resp.status_code == 429 or resp.status_code >= 500:
        raise RetryableError(f"HTTP {resp.status_code} {_short(resp)}")
    if resp.status_code >= 400:
        raise EmbeddingError(f"HTTP {resp.status_code} {_short(resp)}")
    try:
        data = resp.json()["data"]
        data = sorted(data, key=lambda d: d.get("index", 0))
        vecs = [np.asarray(d["embedding"], dtype=np.float32) for d in data]
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise EmbeddingError(f"Unerwartete Antwort des Endpunkts: {e}") from e
    if len(vecs) != len(texts):
        raise EmbeddingError(f"Endpunkt lieferte {len(vecs)} Vektoren für {len(texts)} Texte")
    return vecs


def embed_batch(
    cfg: EmbeddingConfig,
    texts: list[str],
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[np.ndarray]:
    """Ein Batch mit Retry (429/5xx/Netz) und einmaliger Halbierung bei Längenfehlern (400).

    Die Base-URL wird einmal je Batch per netguard geprüft: Localhost bleibt erlaubt (Ollama, LM Studio),
    private Netze und Link-Local (z. B. Cloud-Metadaten 169.254.169.254) nicht."""
    reason = netguard.check_url(cfg.base_url, allow_loopback=True)
    if reason:
        raise EmbeddingError(f"Ungültige Base-URL {cfg.base_url!r}: {reason}")
    own_session = session is None
    session = session or requests.Session()
    attempt = 0
    halved = False
    try:
        while True:
            try:
                return _post_once(cfg, texts, session)
            except RetryableError:
                if attempt >= len(RETRY_SLEEPS):
                    raise
                sleep(RETRY_SLEEPS[attempt])
                attempt += 1
            except EmbeddingError as e:
                if not halved and str(e).startswith("HTTP 400") and _LENGTH_RE.search(str(e)):
                    half = max(1, cfg.max_chars // 2)
                    texts = [t[:half] for t in texts]
                    halved = True
                    continue
                raise
    finally:
        if own_session:
            session.close()


def embed_texts(
    texts: dict[int, str],
    cfg: EmbeddingConfig,
    cache: dict[str, np.ndarray] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    session_factory: Callable[[], requests.Session] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_workers: int = 5,
) -> EmbedResult:
    """Vektoren je Zeilenindex; Cache-Treffer ohne Anfrage; Fehler je Zeile statt Abbruch."""
    cache = {} if cache is None else cache
    session_factory = session_factory or requests.Session
    result = EmbedResult()
    total = len(texts)
    done = 0
    missing: list[tuple[int, str]] = []  # je eindeutigem Text die erste Zeile
    groups: dict[str, list[int]] = {}  # Cache-Schlüssel -> alle Zeilen mit diesem Text
    for idx, text in texts.items():
        key = cache_key(cfg, text)
        hit = cache.get(key)
        if hit is not None:
            result.vectors[idx] = hit
            done += 1
        elif key in groups:
            groups[key].append(idx)
        else:
            groups[key] = [idx]
            missing.append((idx, text))
    if on_progress:
        on_progress(done, total)

    batches = [missing[i:i + cfg.batch_size] for i in range(0, len(missing), cfg.batch_size)]

    def consume(batch: list[tuple[int, str]], vecs: list[np.ndarray] | None, err: str | None) -> None:
        nonlocal done
        for pos, (_, text) in enumerate(batch):
            key = cache_key(cfg, text)
            for idx in groups[key]:
                if err is not None:
                    result.errors[idx] = FAIL_PREFIX + err
                else:
                    result.vectors[idx] = vecs[pos]
                done += 1
            if err is None:
                cache[key] = vecs[pos]
        if on_progress:
            on_progress(done, total)

    def run(batch: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], list[np.ndarray] | None, str | None]:
        session = session_factory()
        try:
            return batch, embed_batch(cfg, [t for _, t in batch], session, sleep), None
        except Exception as e:  # ein defekter Batch darf nie den ganzen Lauf abbrechen
            return batch, None, str(e)
        finally:
            session.close()

    # Ohne `with`: Bei einem Abbruch (z. B. Streamlit-Rerun) nicht auf wartende Batches warten.
    # Fertige Batches stehen da bereits im Cache.
    pool = ThreadPoolExecutor(max_workers=max(1, max_workers))
    try:
        if batches:
            # Erster Batch synchron: Ist der Endpunkt nicht erreichbar oder der Key falsch, erfährt der
            # Aufrufer das nach einem Retry-Zyklus statt nach einem je Batch.
            first = batches[0]
            session = session_factory()
            try:
                vecs = embed_batch(cfg, [t for _, t in first], session, sleep)
            except EmbeddingError:
                raise
            except Exception as e:
                consume(first, None, str(e))
            else:
                consume(first, vecs, None)
            finally:
                session.close()
        futures = [pool.submit(run, b) for b in batches[1:]]
        for fut in as_completed(futures):
            consume(*fut.result())
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        while len(cache) > CACHE_LIMIT:
            cache.pop(next(iter(cache)))

    if result.vectors:
        counts = Counter(int(v.size) for v in result.vectors.values())
        result.dim = max(d for d, c in counts.items() if c == max(counts.values()))
    return result


def test_connection(cfg: EmbeddingConfig) -> int:
    """Schickt einen Ein-Wort-Text; gibt die Dimension zurück oder wirft EmbeddingError."""
    return int(embed_batch(cfg, ["test"], sleep=lambda s: None)[0].size)
