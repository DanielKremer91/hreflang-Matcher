from __future__ import annotations

import json
import re
from collections import Counter
from io import BytesIO
from urllib.parse import parse_qsl, urlencode, urlparse

import numpy as np
import pandas as pd

from .models import LanguageSet

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "pk_campaign", "pk_kwd",
}

_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

URL_COL_NAMES = ["address", "url", "urls", "adresse", "page", "seite", "landing page", "landingpage"]
EMB_COL_HINTS = ["embed", "vector", "vektor"]
STATUS_COL_NAMES = ["status code", "statuscode"]
INDEX_COL_NAMES = ["indexability", "indexierbarkeit"]
# Screaming Frog schreibt den Status der Embedding-Anfrage in diese Spalte.
NOTE_COL_NAMES = ["prompt request status", "request status", "prompt status"]


def normalize_url(url) -> str:
    """Normalisiert eine URL für das Matching. Ergebnis ohne Schema, z. B. 'example.com/de/seite'."""
    s = str(url or "").strip()
    if not s or s.lower() == "nan":
        return ""
    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s
    try:
        p = urlparse(s)
        host = (p.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = p.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        qs = sorted(
            (k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS
        )
        query = urlencode(qs)
        return host + path + ("?" + query if query else "")
    except ValueError:
        return str(url or "").strip()


def parse_vector(value) -> np.ndarray | None:
    """Parst einen Embedding-Wert (JSON-Array, Zahlenliste, list) zu float32. None wenn nicht lesbar."""
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim != 1 or arr.size == 0:
                return None
            return arr
        except (ValueError, TypeError):
            return None
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = np.asarray(json.loads(s), dtype=np.float32)
            if arr.ndim == 1 and arr.size > 0:
                return arr
        except (ValueError, TypeError):
            pass
    nums = _FLOAT_RE.findall(s)
    if not nums:
        return None
    return np.asarray([float(x) for x in nums], dtype=np.float32)


def _read_csv(raw: bytes, sep, encoding: str) -> pd.DataFrame:
    """Liest eine CSV und zählt fehlerhafte Zeilen (zu viele Felder) in df.attrs["bad_lines"]."""
    bad_lines: list[list[str]] = []
    df = pd.read_csv(
        BytesIO(raw), sep=sep, engine="python", encoding=encoding,
        # Rückgabe None verwirft die Zeile; gezählt wird sie trotzdem, damit nichts stumm verloren geht.
        on_bad_lines=lambda bad: (bad_lines.append(bad), None)[1],
    )
    df.columns = [str(c).strip() for c in df.columns]
    df.attrs["bad_lines"] = len(bad_lines)
    return df


def read_any_file(name: str, raw: bytes) -> pd.DataFrame:
    """Liest CSV (Encoding- und Separator-Erkennung) oder Excel aus Bytes. Wirft ValueError.

    Bei CSV steht die Zahl übersprungener fehlerhafter Zeilen in df.attrs["bad_lines"].
    """
    name = (name or "").lower()
    if not raw:
        raise ValueError("Datei ist leer.")
    if name.endswith(".csv") or name.endswith(".txt") or name.endswith(".tsv"):
        last_err = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                df = _read_csv(raw, None, enc)
                if df.shape[1] >= 1:
                    return df
            except UnicodeDecodeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                for sep in (";", ",", "\t"):
                    try:
                        return _read_csv(raw, sep, enc)
                    except Exception as e2:
                        last_err = e2
        raise ValueError(f"CSV konnte nicht gelesen werden: {last_err}")
    try:
        df = pd.read_excel(BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Excel konnte nicht gelesen werden: {e}") from e
    df.columns = [str(c).strip() for c in df.columns]
    df.attrs["bad_lines"] = 0
    return df


def _looks_like_url_series(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if len(sample) == 0:
        return False
    hits = sum(1 for v in sample if re.match(r"^https?://", v.strip(), re.I) or ("/" in v and "." in v))
    return hits >= max(1, int(len(sample) * 0.2))


def detect_url_column(df: pd.DataFrame) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for key in URL_COL_NAMES:
        if key in lower and _looks_like_url_series(df[lower[key]]):
            return lower[key]
    for c in df.columns:
        if any(k in str(c).lower() for k in URL_COL_NAMES) and _looks_like_url_series(df[c]):
            return c
    for c in df.columns:
        if _looks_like_url_series(df[c]):
            return c
    return None


def _looks_like_vector_series(series: pd.Series, min_len: int = 8, min_ratio: float = 0.8) -> bool:
    sample = series.dropna().astype(str).head(20)
    if len(sample) == 0:
        return False
    good = 0
    for v in sample:
        vec = parse_vector(v)
        if vec is not None and vec.size >= min_len:
            good += 1
    return good >= 1 and good / len(sample) >= min_ratio


def detect_embedding_column(df: pd.DataFrame) -> str | None:
    # Named columns: lenient ratio, broken rows are reported per row later.
    for c in df.columns:
        if any(h in str(c).lower() for h in EMB_COL_HINTS) and _looks_like_vector_series(df[c], min_ratio=0.2):
            return c
    # Content-only detection: strict default ratio.
    for c in df.columns:
        if _looks_like_vector_series(df[c]):
            return c
    return None


def _detect_by_names(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def detect_status_column(df: pd.DataFrame) -> str | None:
    return _detect_by_names(df, STATUS_COL_NAMES)


def detect_indexability_column(df: pd.DataFrame) -> str | None:
    return _detect_by_names(df, INDEX_COL_NAMES)


def detect_note_column(df: pd.DataFrame) -> str | None:
    """Spalte mit dem Status der Embedding-Anfrage (z. B. Screaming Frog 'Prompt Request Status')."""
    return _detect_by_names(df, NOTE_COL_NAMES)


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().lower() in ("", "nan", "none")


def is_indexable(value) -> bool:
    if value is None:
        return False
    v = str(value).strip().lower()
    return v.startswith("indexable") or v.startswith("indexierbar")


def build_language_set(
    df: pd.DataFrame,
    code: str,
    label: str,
    url_col: str,
    emb_col: str,
    filter_indexable: bool,
    status_col: str | None = None,
    index_col: str | None = None,
    note_col: str | None = None,
) -> LanguageSet:
    """Baut ein LanguageSet: filtert, parst Vektoren, verwirft Ausreißer, dedupliziert, L2-normalisiert.

    note_col: optionale Spalte mit dem Status der Embedding-Anfrage; ihr Wert wird bei leeren
    Embeddings in den Verwerfungsgrund übernommen."""
    rows: list[tuple[str, str, np.ndarray]] = []
    dropped: list[tuple[str, str]] = []

    for _, r in df.iterrows():
        url_raw = "" if pd.isna(r[url_col]) else str(r[url_col]).strip()
        if not url_raw:
            dropped.append(("", "leere URL"))
            continue
        if filter_indexable and status_col is not None:
            code_val = pd.to_numeric(r[status_col], errors="coerce")
            if pd.isna(code_val) or int(code_val) != 200:
                dropped.append((url_raw, f"Status Code {'' if pd.isna(code_val) else int(code_val)}".strip()))
                continue
        if filter_indexable and index_col is not None:
            if not is_indexable(r[index_col]):
                dropped.append((url_raw, f"nicht indexierbar ({r[index_col]})"))
                continue
        if _is_blank(r[emb_col]):
            note = "" if note_col is None or _is_blank(r[note_col]) else str(r[note_col]).strip()
            dropped.append((url_raw, f"Embedding fehlt ({note_col}: {note})" if note else "Embedding fehlt"))
            continue
        vec = parse_vector(r[emb_col])
        if vec is None:
            dropped.append((url_raw, "Embedding nicht lesbar"))
            continue
        rows.append((url_raw, normalize_url(url_raw), vec))

    dim = 0
    if rows:
        counts = Counter(int(v.size) for _, _, v in rows)
        best = max(counts.values())
        dim = max(d for d, c in counts.items() if c == best)

    urls: list[str] = []
    norm_urls: list[str] = []
    vecs: list[np.ndarray] = []
    seen: dict[str, str] = {}
    for url_raw, norm, vec in rows:
        if vec.size != dim:
            dropped.append((url_raw, f"Dimension {vec.size} statt {dim} (vermutlich abgeschnitten)"))
            continue
        if norm in seen:
            dropped.append((url_raw, f"Duplikat nach Normalisierung von {seen[norm]}"))
            continue
        n = float(np.linalg.norm(vec))
        if not np.isfinite(n) or n == 0.0:
            dropped.append((url_raw, "Nullvektor oder ungültige Werte"))
            continue
        seen[norm] = url_raw
        urls.append(url_raw)
        norm_urls.append(norm)
        vecs.append((vec / n).astype(np.float32))

    vectors = np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)
    dropped_df = pd.DataFrame(dropped, columns=["URL", "Grund"])
    return LanguageSet(code=code, label=label, urls=urls, norm_urls=norm_urls, vectors=vectors, dropped=dropped_df)


def check_dimensions(sets: list[LanguageSet]) -> None:
    dims = {s.code: s.dim for s in sets if len(s) > 0}
    if len(set(dims.values())) > 1:
        detail = ", ".join(f"{c}: {d}" for c, d in dims.items())
        raise ValueError(
            "Die Embedding-Dimensionen unterscheiden sich zwischen den Dateien "
            f"({detail}). Alle Dateien müssen mit demselben Modell erzeugt sein. "
            "Bei Excel-Exporten prüfe das Zellenlimit von 32.767 Zeichen."
        )
