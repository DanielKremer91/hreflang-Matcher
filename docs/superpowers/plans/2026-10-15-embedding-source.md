# Embedding-Quelle im Tool – Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Das Tool kann Embeddings selbst berechnen: aus Textspalten der hochgeladenen Datei oder aus Seiteninhalten (Trafilatura), über einen OpenAI-kompatiblen Endpunkt. Der bestehende Weg (Embeddings in der Datei) bleibt unverändert.

**Architecture:** Drei neue streamlit-freie Module (`textprep.py`, `embed.py`, `fetch.py`) liefern Texte bzw. Vektoren; `io_utils.language_set_from_vectors` baut daraus dieselben `LanguageSet`-Objekte wie heute, sodass Matching und Ausgabe unverändert bleiben. `app.py` bekommt ein Radio „Embedding-Quelle“, einen Einstellungs-Expander und einen expliziten Button „Embeddings berechnen“ mit Session-Cache.

**Tech Stack:** Python ≥3.10 (Entwicklung mit 3.14 in `.venv`), streamlit, pandas, numpy, openpyxl, requests, trafilatura, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-embedding-source-design.md` (ergänzt `docs/superpowers/specs/2026-09-29-hreflang-matcher-design.md`)

## Global Constraints

- Bestehendes Verhalten bei Quelle „Embeddings sind in der Datei enthalten“ bleibt exakt gleich; alle 132 bestehenden Tests müssen weiter passen.
- Neue Abhängigkeiten nur `requests>=2.31` und `trafilatura>=2.0`. Kein torch, keine SDKs. `pyproject.toml` bekommt `requires-python = ">=3.10"`.
- Module unter `hreflang_matcher/` importieren **nie** streamlit. Jede Datei beginnt mit `from __future__ import annotations`.
- Keine API-Anfrage ohne Klick auf „Embeddings berechnen“ (bzw. „Verbindung testen“). Slider, Spaltenwechsel, Reruns lösen nichts aus.
- Embedding-Endpunkt: `POST {base_url}/embeddings`, Body `{"model": ..., "input": [...]}`, Header `Authorization: Bearer <key>` nur wenn Key gesetzt; Antwort `data[i].embedding` in Eingabereihenfolge.
- Batches von 64, max. 5 parallele Batches; Retry bei 429/5xx/Netzfehler 3× mit 2, 4, 8 s; 4xx (außer 429) → Grund `Embedding-Anfrage fehlgeschlagen: HTTP <code> <kurz>`; bei 400 mit Längenfehler einmal mit halbiertem `max_chars` wiederholen.
- Trafilatura: User-Agent `ONE-hreflang-Matcher/1.0 (+https://onebeyondsearch.com)`, 5 parallele Abrufe, Timeout 10 s, nur `text/html` mit 200; Gründe `HTTP <status>`, `kein HTML (<content-type>)`, `Timeout`, `Verbindungsfehler: <kurz>`, `kein Text extrahiert`.
- Session-Cache der Vektoren: Schlüssel `sha1(model + "|" + base_url + "|" + text)`, max. 50 000 Einträge, FIFO.
- UI-Texte Deutsch; Presets exakt: „OpenAI“, „Ollama (lokal)“, „LM Studio (lokal)“, „Eigener OpenAI-kompatibler Endpunkt“.
- Commit-Messages wie angegeben, jeweils mit Endzeile `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Nie `.venv/` oder `.superpowers/` committen.
- Arbeitsverzeichnis `/Users/Daniel/Downloads/one-hreflang-matcher`, Branch `feature/embedding-source`, Tests mit `.venv/bin/python -m pytest -q -W error`.

---

## File Structure

| Datei | Verantwortung |
|---|---|
| `requirements.txt`, `pyproject.toml`, `README.md` | Abhängigkeiten, Python-Floor, Doku |
| `hreflang_matcher/models.py` | + `EmbeddingConfig`, `EmbedResult`, `FetchResult` |
| `hreflang_matcher/textprep.py` | Textspalten vorschlagen, Texte bauen/kürzen, Token-Schätzung |
| `hreflang_matcher/io_utils.py` | + `row_filter_reasons`, `language_set_from_vectors`, `embeddings_csv`; `build_language_set` nutzt die gemeinsamen Helfer |
| `hreflang_matcher/embed.py` | Presets, OpenAI-kompatibler Client, Batching, Retry, Cache |
| `hreflang_matcher/fetch.py` | Seiten laden, Trafilatura-Extraktion, parallel, Gründe |
| `app.py` | Radio Embedding-Quelle, Textspalten/URL-only je Quelle, Einstellungs-Expander, Button, Cache, Export |
| `tests/test_textprep.py`, `tests/test_embed.py`, `tests/test_fetch.py`, `tests/test_io_utils.py`, `tests/test_app.py` | Tests |

---

### Task 1: Abhängigkeiten, Datenmodell, Textaufbereitung

**Files:**
- Modify: `requirements.txt`, `pyproject.toml`, `README.md`, `hreflang_matcher/models.py`
- Create: `hreflang_matcher/textprep.py`
- Test: `tests/test_textprep.py`

**Interfaces:**
- Produces: `models.EmbeddingConfig(base_url, model, api_key="", max_chars=20000, batch_size=64, timeout=60.0)`, `models.EmbedResult(vectors: dict[int, np.ndarray], errors: dict[int, str], dim: int)`, `models.FetchResult(texts: dict[int, str], errors: dict[int, str])`
- Produces: `textprep.suggest_text_columns(df) -> list[str]`, `textprep.truncate(text, max_chars) -> str`, `textprep.build_texts(df, cols, max_chars) -> tuple[dict[int, str], dict[int, str]]` (positionaler Zeilenindex → Text, positionaler Zeilenindex → Grund für leere Zeilen), `textprep.estimate_tokens(texts: Iterable[str]) -> int`, `textprep.NO_TEXT_REASON = "kein Text in den gewählten Spalten"`

- [ ] **Step 1: Abhängigkeiten und Python-Floor**

`requirements.txt` ergänzen:
```
requests>=2.31
trafilatura>=2.0
```
`pyproject.toml`:
```toml
[project]
name = "one-hreflang-matcher"
version = "0.2.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
```
`README.md`, Abschnitt „Getestete Umgebungen“ ersetzen durch:
```markdown
## Getestete Umgebungen

- Python 3.14 mit streamlit 1.66, pandas 3.0, numpy 2.5, trafilatura 2.3, requests 2.34 (Streamlit-Cloud-Standard)

Python 3.10 oder neuer wird benötigt (trafilatura und requests). Die Testsuite läuft ohne Warnungen durch.
```
Run: `uv pip install -q --python .venv/bin/python -r requirements-dev.txt && .venv/bin/python -c "import requests, trafilatura; print('ok')"` → `ok` (die venv wurde mit uv angelegt und hat kein pip).

- [ ] **Step 2: Datenmodell ergänzen**

In `hreflang_matcher/models.py` `field` importieren (`from dataclasses import dataclass, field`) und am Ende anhängen:
```python
@dataclass
class EmbeddingConfig:
    """Verbindung zu einem OpenAI-kompatiblen Embedding-Endpunkt."""

    base_url: str              # z. B. "https://api.openai.com/v1", "http://localhost:11434/v1"
    model: str                 # z. B. "text-embedding-3-small", "bge-m3"
    api_key: str = ""          # leer bei lokalen Servern
    max_chars: int = 20000     # Kürzung je Text vor dem Versand
    batch_size: int = 64
    timeout: float = 60.0


@dataclass
class EmbedResult:
    vectors: dict[int, np.ndarray] = field(default_factory=dict)   # Zeilenindex -> float32-Vektor
    errors: dict[int, str] = field(default_factory=dict)           # Zeilenindex -> Grund
    dim: int = 0


@dataclass
class FetchResult:
    texts: dict[int, str] = field(default_factory=dict)            # Zeilenindex -> Haupttext
    errors: dict[int, str] = field(default_factory=dict)           # Zeilenindex -> Grund
```

- [ ] **Step 3: Fehlschlagende Tests für textprep**

`tests/test_textprep.py`:
```python
import pandas as pd

from hreflang_matcher import textprep


class TestSuggestTextColumns:
    def test_hint_order_main_content_first(self):
        df = pd.DataFrame(columns=["Address", "Title 1", "H1-1", "Main Content Extraction 1", "Status Code"])
        assert textprep.suggest_text_columns(df) == ["Main Content Extraction 1", "Title 1", "H1-1"]

    def test_german_and_meta(self):
        df = pd.DataFrame(columns=["Adresse", "Meta Description 1", "Body Text", "Inlinks"])
        assert textprep.suggest_text_columns(df) == ["Body Text", "Meta Description 1"]

    def test_none(self):
        assert textprep.suggest_text_columns(pd.DataFrame(columns=["Address", "Inlinks"])) == []


class TestTruncate:
    def test_cuts_and_normalizes(self):
        assert textprep.truncate("  a   b\n\n\tc  ", 100) == "a b c"
        assert textprep.truncate("abcdef", 3) == "abc"
        assert textprep.truncate(None, 3) == ""


class TestBuildTexts:
    def test_joins_in_column_order_and_skips_empty(self):
        df = pd.DataFrame({
            "Address": ["u1", "u2", "u3"],
            "Title": ["Titel 1", None, ""],
            "Main": ["Inhalt   eins", "Inhalt zwei", float("nan")],
        })
        texts, errors = textprep.build_texts(df, ["Title", "Main"], max_chars=100)
        assert texts == {0: "Titel 1\n\nInhalt eins", 1: "Inhalt zwei"}
        assert errors == {2: textprep.NO_TEXT_REASON}

    def test_truncates_total(self):
        df = pd.DataFrame({"T": ["x" * 50], "M": ["y" * 50]})
        texts, _ = textprep.build_texts(df, ["T", "M"], max_chars=60)
        assert len(texts[0]) == 60 and texts[0].startswith("x" * 50)

    def test_positional_index_not_label(self):
        df = pd.DataFrame({"T": ["a", "b"]}, index=[10, 20])
        texts, _ = textprep.build_texts(df, ["T"], max_chars=10)
        assert texts == {0: "a", 1: "b"}


def test_estimate_tokens():
    assert textprep.estimate_tokens(["abcd" * 10, "ab"]) == 10
```

- [ ] **Step 4: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_textprep.py -q` → ModuleNotFoundError.

- [ ] **Step 5: textprep implementieren**

`hreflang_matcher/textprep.py`:
```python
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
```

- [ ] **Step 6: Tests laufen lassen**

Run: `.venv/bin/python -m pytest -q -W error` → alle PASS (132 + 8).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pyproject.toml README.md hreflang_matcher/models.py hreflang_matcher/textprep.py tests/test_textprep.py
git commit -m "feat: embedding config models and text preparation; require Python 3.10+

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: LanguageSet aus fertigen Vektoren, gemeinsamer Filterpfad, CSV-Export

**Files:**
- Modify: `hreflang_matcher/io_utils.py` (`build_language_set` refaktorieren, neue Funktionen anhängen)
- Test: `tests/test_io_utils.py`

**Interfaces:**
- Consumes: `models.LanguageSet`, `io_utils.normalize_url`, `io_utils.is_indexable`, `io_utils._is_blank`
- Produces:
  - `row_filter_reasons(df, url_col, filter_indexable, status_col=None, index_col=None) -> tuple[dict[int, str], dict[int, str]]` → (positionaler Index → Original-URL für behaltene Zeilen, positionaler Index → Grund für verworfene Zeilen). Gründe exakt wie bisher: `leere URL`, `Status Code <n>`, `nicht indexierbar (<wert>)`.
  - `finalize_language_set(code, label, rows: list[tuple[str, str, np.ndarray]], dropped: list[tuple[str, str]]) -> LanguageSet` (Dimensionsmehrheit, Duplikate, Nullvektoren, L2-Normalisierung – die bisherige zweite Hälfte von `build_language_set`).
  - `language_set_from_vectors(df, code, label, url_col, vectors: dict[int, np.ndarray], errors: dict[int, str], filter_indexable, status_col=None, index_col=None) -> LanguageSet`
  - `embeddings_csv(ls: LanguageSet) -> bytes` (Spalten `Address`, `Embedding` als JSON-Array mit 6 Nachkommastellen, utf-8-sig)
- `build_language_set` behält Signatur und Verhalten, nutzt intern `row_filter_reasons` und `finalize_language_set`.

- [ ] **Step 1: Fehlschlagende Tests anhängen**

An `tests/test_io_utils.py` anhängen (`_vec` und `LanguageSet` sind dort bereits verfügbar; `import json` am Dateianfang ergänzen):
```python
class TestRowFilterReasons:
    def _df(self):
        return pd.DataFrame({
            "Address": ["https://a.de/1", "", "https://a.de/3", "https://a.de/4"],
            "Status Code": [200, 200, 404, 200],
            "Indexability": ["Indexable", "Indexable", "Indexable", "Non-Indexable"],
        }, index=[7, 8, 9, 10])

    def test_positional_with_filter(self):
        keep, drop = io_utils.row_filter_reasons(self._df(), "Address", True, "Status Code", "Indexability")
        assert keep == {0: "https://a.de/1"}
        assert drop == {1: "leere URL", 2: "Status Code 404", 3: "nicht indexierbar (Non-Indexable)"}

    def test_without_filter(self):
        keep, drop = io_utils.row_filter_reasons(self._df(), "Address", False, "Status Code", "Indexability")
        assert set(keep) == {0, 2, 3} and drop == {1: "leere URL"}


class TestLanguageSetFromVectors:
    def test_builds_with_errors_and_filters(self):
        df = pd.DataFrame({
            "Address": ["https://a.de/1/", "https://a.de/2", "https://a.de/1", "https://a.de/4", "https://a.de/5"],
            "Status Code": [200, 200, 200, 404, 200],
        })
        vectors = {0: np.ones(8, dtype=np.float32), 1: np.arange(8, dtype=np.float32), 2: np.ones(8, dtype=np.float32),
                   3: np.ones(8, dtype=np.float32)}
        errors = {4: "Embedding-Anfrage fehlgeschlagen: HTTP 401 Unauthorized"}
        ls = io_utils.language_set_from_vectors(df, "de", "x", "Address", vectors, errors, True, "Status Code")
        assert ls.urls == ["https://a.de/1/", "https://a.de/2"]
        np.testing.assert_allclose(np.linalg.norm(ls.vectors, axis=1), [1.0, 1.0], rtol=1e-5)
        reasons = dict(zip(ls.dropped["URL"], ls.dropped["Grund"]))
        assert "Duplikat" in reasons["https://a.de/1"]
        assert reasons["https://a.de/4"] == "Status Code 404"
        assert reasons["https://a.de/5"].startswith("Embedding-Anfrage fehlgeschlagen")

    def test_row_without_vector_or_error_is_reported(self):
        df = pd.DataFrame({"Address": ["https://a.de/1"]})
        ls = io_utils.language_set_from_vectors(df, "de", "x", "Address", {}, {}, False)
        assert len(ls) == 0 and ls.dropped["Grund"].tolist() == ["Embedding fehlt"]

    def test_dimension_mismatch_dropped(self):
        df = pd.DataFrame({"Address": ["https://a.de/1", "https://a.de/2", "https://a.de/3"]})
        vectors = {0: np.ones(8, dtype=np.float32), 1: np.ones(8, dtype=np.float32), 2: np.ones(5, dtype=np.float32)}
        ls = io_utils.language_set_from_vectors(df, "de", "x", "Address", vectors, {}, False)
        assert len(ls) == 2 and "Dimension 5 statt 8" in ls.dropped["Grund"].iloc[0]


class TestEmbeddingsCsv:
    def test_roundtrip_into_existing_mode(self):
        df = pd.DataFrame({"Address": ["https://a.de/1", "https://a.de/2"], "Embeddings": [_vec(8, 1), _vec(8, 2)]})
        ls = io_utils.build_language_set(df, "de", "x", "Address", "Embeddings", filter_indexable=False)
        raw = io_utils.embeddings_csv(ls)
        assert raw.startswith("﻿".encode("utf-8"))
        df2 = io_utils.read_any_file("embeddings_de.csv", raw)
        assert list(df2.columns) == ["Address", "Embedding"]
        assert io_utils.detect_embedding_column(df2) == "Embedding"
        ls2 = io_utils.build_language_set(df2, "de", "y", "Address", "Embedding", filter_indexable=False)
        assert ls2.urls == ls.urls
        np.testing.assert_allclose(ls2.vectors, ls.vectors, atol=2e-6)
        assert len(json.loads(df2["Embedding"].iloc[0])) == 8
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_io_utils.py -q` → AttributeError für die neuen Funktionen.

- [ ] **Step 3: io_utils refaktorieren und ergänzen**

`build_language_set` so umbauen, dass der Zeilenfilter und die Finalisierung in Helfer wandern. Ergebnis in `hreflang_matcher/io_utils.py`:
```python
def row_filter_reasons(
    df: pd.DataFrame,
    url_col: str,
    filter_indexable: bool,
    status_col: str | None = None,
    index_col: str | None = None,
) -> tuple[dict[int, str], dict[int, str]]:
    """Positionaler Zeilenindex -> behaltene Original-URL bzw. -> Verwerfungsgrund (URL/Status/Indexierbarkeit)."""
    keep: dict[int, str] = {}
    drop: dict[int, str] = {}
    for pos, (_, r) in enumerate(df.iterrows()):
        url_raw = "" if pd.isna(r[url_col]) else str(r[url_col]).strip()
        if not url_raw:
            drop[pos] = "leere URL"
            continue
        if filter_indexable and status_col is not None:
            code_val = pd.to_numeric(r[status_col], errors="coerce")
            if pd.isna(code_val) or int(code_val) != 200:
                drop[pos] = f"Status Code {'' if pd.isna(code_val) else int(code_val)}".strip()
                continue
        if filter_indexable and index_col is not None and not is_indexable(r[index_col]):
            drop[pos] = f"nicht indexierbar ({r[index_col]})"
            continue
        keep[pos] = url_raw
    return keep, drop


def finalize_language_set(
    code: str, label: str, rows: list[tuple[str, str, np.ndarray]], dropped: list[tuple[str, str]]
) -> LanguageSet:
    """Dimensionsmehrheit, Duplikate, Nullvektoren, L2-Normalisierung; baut das LanguageSet."""
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
    """Baut ein LanguageSet aus einer Embedding-Spalte: filtert, parst Vektoren, verwirft Ausreißer, dedupliziert.

    note_col: optionale Spalte mit dem Status der Embedding-Anfrage; ihr Wert wird bei leeren
    Embeddings in den Verwerfungsgrund übernommen."""
    keep, drop = row_filter_reasons(df, url_col, filter_indexable, status_col, index_col)
    rows: list[tuple[str, str, np.ndarray]] = []
    dropped: list[tuple[str, str]] = []
    url_values = df[url_col].tolist()
    for pos, reason in drop.items():
        u = url_values[pos]
        dropped.append(("" if pd.isna(u) else str(u).strip(), reason))
    emb_values = df[emb_col].tolist()
    note_values = df[note_col].tolist() if note_col is not None else None
    for pos, url_raw in keep.items():
        if _is_blank(emb_values[pos]):
            note = "" if note_values is None or _is_blank(note_values[pos]) else str(note_values[pos]).strip()
            dropped.append((url_raw, f"Embedding fehlt ({note_col}: {note})" if note else "Embedding fehlt"))
            continue
        vec = parse_vector(emb_values[pos])
        if vec is None:
            dropped.append((url_raw, "Embedding nicht lesbar"))
            continue
        rows.append((url_raw, normalize_url(url_raw), vec))
    return finalize_language_set(code, label, rows, dropped)


def language_set_from_vectors(
    df: pd.DataFrame,
    code: str,
    label: str,
    url_col: str,
    vectors: dict[int, np.ndarray],
    errors: dict[int, str],
    filter_indexable: bool,
    status_col: str | None = None,
    index_col: str | None = None,
) -> LanguageSet:
    """Baut ein LanguageSet aus bereits berechneten Vektoren (positionaler Zeilenindex -> Vektor)."""
    keep, drop = row_filter_reasons(df, url_col, filter_indexable, status_col, index_col)
    rows: list[tuple[str, str, np.ndarray]] = []
    dropped: list[tuple[str, str]] = []
    url_values = df[url_col].tolist()
    for pos, reason in drop.items():
        u = url_values[pos]
        dropped.append(("" if pd.isna(u) else str(u).strip(), reason))
    for pos, url_raw in keep.items():
        if pos in errors:
            dropped.append((url_raw, errors[pos]))
        elif pos in vectors:
            rows.append((url_raw, normalize_url(url_raw), np.asarray(vectors[pos], dtype=np.float32)))
        else:
            dropped.append((url_raw, "Embedding fehlt"))
    return finalize_language_set(code, label, rows, dropped)


def embeddings_csv(ls: LanguageSet) -> bytes:
    """CSV mit Address + Embedding (JSON-Array), im Modus 'Embeddings in der Datei' direkt nutzbar."""
    emb = ["[" + ", ".join(f"{x:.6f}" for x in v) + "]" for v in ls.vectors]
    df = pd.DataFrame({"Address": ls.urls, "Embedding": emb})
    return df.to_csv(index=False).encode("utf-8-sig")
```
Die alte Schleife in `build_language_set` (mit `for _, r in df.iterrows()` und der Finalisierung) wird vollständig durch diese Fassung ersetzt. Die Reihenfolge der `dropped`-Einträge ändert sich (Filtergründe zuerst); bestehende Tests prüfen Mengen, nicht Reihenfolge.

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest -q -W error` → alle PASS, inkl. der alten `TestBuildLanguageSet`.

- [ ] **Step 5: Commit**

```bash
git add hreflang_matcher/io_utils.py tests/test_io_utils.py
git commit -m "feat: build LanguageSets from precomputed vectors and export embeddings as CSV

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Embedding-Client (OpenAI-kompatibel)

**Files:**
- Create: `hreflang_matcher/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: `models.EmbeddingConfig`, `models.EmbedResult`
- Produces:
  - `embed.PRESETS: dict[str, dict]` mit Schlüsseln `base_url`, `models` (Liste), `local` (bool), `needs_key` (bool); Preset-Namen exakt „OpenAI“, „Ollama (lokal)“, „LM Studio (lokal)“, „Eigener OpenAI-kompatibler Endpunkt“.
  - `embed.cache_key(cfg, text) -> str`
  - `embed.EmbeddingError(Exception)` (nicht wiederholbar), `embed.RetryableError(EmbeddingError)`
  - `embed.embed_batch(cfg, texts: list[str], session=None, sleep=time.sleep) -> list[np.ndarray]` (Retry/Halbierung eingebaut; wirft `EmbeddingError` mit `HTTP <code> <kurz>`-Text)
  - `embed.embed_texts(texts: dict[int, str], cfg, cache: dict[str, np.ndarray] | None = None, on_progress=None, session_factory=None, sleep=time.sleep, max_workers=5) -> EmbedResult`
  - `embed.test_connection(cfg) -> int` (Dimension)
  - `embed.CACHE_LIMIT = 50_000`

- [ ] **Step 1: Fehlschlagende Tests mit lokalem Testserver**

`tests/test_embed.py`:
```python
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from hreflang_matcher import embed
from hreflang_matcher.models import EmbeddingConfig


def _vec_for(text: str, dim: int = 4) -> list[float]:
    h = sum(ord(c) for c in text)
    return [float((h + i) % 7) for i in range(dim)]


class _Handler(BaseHTTPRequestHandler):
    # Pro Serverinstanz: Liste geplanter Antworten [(status, body_dict | None)]; None = normale Antwort.
    script: list = []
    requests_seen: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests_seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        status, payload = (200, None)
        if type(self).script:
            status, payload = type(self).script.pop(0)
        if payload is None and status == 200:
            payload = {"data": [{"index": i, "embedding": _vec_for(t)} for i, t in enumerate(body["input"])]}
        if payload is None:
            payload = {"error": {"message": "boom"}}
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.script = []
    _Handler.requests_seen = []
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, _Handler
    srv.shutdown()


def _cfg(srv, key="", **kw):
    return EmbeddingConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="m", api_key=key, **kw)


def test_presets_exact():
    assert list(embed.PRESETS) == ["OpenAI", "Ollama (lokal)", "LM Studio (lokal)", "Eigener OpenAI-kompatibler Endpunkt"]
    assert embed.PRESETS["OpenAI"]["base_url"] == "https://api.openai.com/v1"
    assert embed.PRESETS["OpenAI"]["models"][0] == "text-embedding-3-small"
    assert embed.PRESETS["Ollama (lokal)"]["local"] is True and embed.PRESETS["Ollama (lokal)"]["models"] == ["bge-m3"]


def test_cache_key_depends_on_model_url_text():
    c1 = EmbeddingConfig("http://x/v1", "a")
    c2 = EmbeddingConfig("http://x/v1", "b")
    assert embed.cache_key(c1, "t") != embed.cache_key(c2, "t")
    assert embed.cache_key(c1, "t") == embed.cache_key(EmbeddingConfig("http://x/v1", "a", api_key="k"), "t")


def test_embed_batch_posts_to_embeddings_with_bearer(server):
    srv, h = server
    out = embed.embed_batch(_cfg(srv, key="sk-1"), ["hallo", "welt"])
    assert [v.tolist() for v in out] == [_vec_for("hallo"), _vec_for("welt")]
    assert out[0].dtype == np.float32
    req = h.requests_seen[0]
    assert req["path"] == "/v1/embeddings" and req["auth"] == "Bearer sk-1"
    assert req["body"] == {"model": "m", "input": ["hallo", "welt"]}


def test_embed_batch_without_key_sends_no_auth_header(server):
    srv, h = server
    embed.embed_batch(_cfg(srv), ["x"])
    assert h.requests_seen[0]["auth"] is None


def test_retry_on_429_then_500_then_ok(server):
    srv, h = server
    h.script[:] = [(429, None), (500, None)]
    sleeps = []
    out = embed.embed_batch(_cfg(srv), ["x"], sleep=sleeps.append)
    assert len(out) == 1 and sleeps == [2, 4] and len(h.requests_seen) == 3


def test_gives_up_after_three_retries(server):
    srv, h = server
    h.script[:] = [(503, None)] * 4
    with pytest.raises(embed.EmbeddingError) as e:
        embed.embed_batch(_cfg(srv), ["x"], sleep=lambda s: None)
    assert "HTTP 503" in str(e.value) and len(h.requests_seen) == 4


def test_401_is_not_retried(server):
    srv, h = server
    h.script[:] = [(401, {"error": {"message": "Incorrect API key"}})]
    with pytest.raises(embed.EmbeddingError) as e:
        embed.embed_batch(_cfg(srv), ["x"], sleep=lambda s: None)
    assert str(e.value).startswith("HTTP 401") and "Incorrect API key" in str(e.value)
    assert len(h.requests_seen) == 1


def test_400_length_error_retries_once_with_half_max_chars(server):
    srv, h = server
    h.script[:] = [(400, {"error": {"message": "This model's maximum context length is 8192 tokens"}})]
    long = "a" * 1000
    out = embed.embed_batch(_cfg(srv, max_chars=1000), [long], sleep=lambda s: None)
    assert len(out) == 1
    assert len(h.requests_seen) == 2 and len(h.requests_seen[1]["body"]["input"][0]) == 500


def test_embed_texts_batches_cache_and_errors(server):
    srv, h = server
    cfg = _cfg(srv, batch_size=2)
    cache: dict = {}
    texts = {0: "a", 1: "b", 2: "c"}
    progress = []
    res = embed.embed_texts(texts, cfg, cache=cache, on_progress=lambda d, t: progress.append((d, t)), max_workers=1)
    assert set(res.vectors) == {0, 1, 2} and res.errors == {} and res.dim == 4
    assert len(h.requests_seen) == 2 and progress[-1] == (3, 3)
    assert len(cache) == 3
    # zweiter Lauf: alles aus dem Cache, keine Anfrage
    res2 = embed.embed_texts(texts, cfg, cache=cache, max_workers=1)
    assert len(h.requests_seen) == 2 and res2.vectors[1].tolist() == _vec_for("b")


def test_embed_texts_reports_batch_failure_per_row(server):
    srv, h = server
    h.script[:] = [(401, {"error": {"message": "bad key"}})]
    res = embed.embed_texts({0: "a", 1: "b"}, _cfg(srv, batch_size=2), max_workers=1, sleep=lambda s: None)
    assert res.vectors == {} and res.dim == 0
    assert res.errors == {0: "Embedding-Anfrage fehlgeschlagen: HTTP 401 bad key",
                          1: "Embedding-Anfrage fehlgeschlagen: HTTP 401 bad key"}


def test_cache_fifo_limit(monkeypatch, server):
    srv, _ = server
    monkeypatch.setattr(embed, "CACHE_LIMIT", 3)
    cache: dict = {}
    embed.embed_texts({i: f"t{i}" for i in range(5)}, _cfg(srv, batch_size=5), cache=cache, max_workers=1)
    assert len(cache) == 3 and embed.cache_key(_cfg(srv), "t4") in cache and embed.cache_key(_cfg(srv), "t0") not in cache


def test_connection_returns_dim(server):
    srv, _ = server
    assert embed.test_connection(_cfg(srv)) == 4


def test_connection_error_is_embedding_error():
    cfg = EmbeddingConfig(base_url="http://127.0.0.1:9/v1", model="m", timeout=1)
    with pytest.raises(embed.EmbeddingError) as e:
        embed.embed_batch(cfg, ["x"], sleep=lambda s: None)
    assert "Verbindungsfehler" in str(e.value)
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_embed.py -q` → ModuleNotFoundError.

- [ ] **Step 3: embed.py implementieren**

```python
from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import requests

from .models import EmbeddingConfig, EmbedResult

PRESETS: dict[str, dict] = {
    "OpenAI": {"base_url": "https://api.openai.com/v1",
               "models": ["text-embedding-3-small", "text-embedding-3-large"], "local": False, "needs_key": True},
    "Ollama (lokal)": {"base_url": "http://localhost:11434/v1", "models": ["bge-m3"], "local": True, "needs_key": False},
    "LM Studio (lokal)": {"base_url": "http://localhost:1234/v1", "models": [], "local": True, "needs_key": False},
    "Eigener OpenAI-kompatibler Endpunkt": {"base_url": "", "models": [], "local": False, "needs_key": False},
}
CACHE_LIMIT = 50_000
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
        msg = resp.json().get("error", {}).get("message") or resp.text
    except ValueError:
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
    except (ValueError, KeyError, TypeError) as e:
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
    """Ein Batch mit Retry (429/5xx/Netz) und einmaliger Halbierung bei Längenfehlern (400)."""
    session = session or requests.Session()
    attempt = 0
    halved = False
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
    missing: list[tuple[int, str]] = []
    for idx, text in texts.items():
        hit = cache.get(cache_key(cfg, text))
        if hit is not None:
            result.vectors[idx] = hit
            done += 1
        else:
            missing.append((idx, text))
    if on_progress:
        on_progress(done, total)

    batches = [missing[i:i + cfg.batch_size] for i in range(0, len(missing), cfg.batch_size)]

    def run(batch: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], list[np.ndarray] | None, str | None]:
        try:
            return batch, embed_batch(cfg, [t for _, t in batch], session_factory(), sleep), None
        except EmbeddingError as e:
            return batch, None, str(e)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for batch, vecs, err in pool.map(run, batches):
            for pos, (idx, text) in enumerate(batch):
                if err is not None:
                    result.errors[idx] = FAIL_PREFIX + err
                else:
                    result.vectors[idx] = vecs[pos]
                    cache[cache_key(cfg, text)] = vecs[pos]
            done += len(batch)
            if on_progress:
                on_progress(done, total)

    while len(cache) > CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    if result.vectors:
        counts = Counter(int(v.size) for v in result.vectors.values())
        result.dim = max(d for d, c in counts.items() if c == max(counts.values()))
    return result


def test_connection(cfg: EmbeddingConfig) -> int:
    """Schickt einen Ein-Wort-Text; gibt die Dimension zurück oder wirft EmbeddingError."""
    return int(embed_batch(cfg, ["test"], sleep=lambda s: None)[0].size)
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest -q -W error` → alle PASS.

- [ ] **Step 5: Commit**

```bash
git add hreflang_matcher/embed.py tests/test_embed.py
git commit -m "feat: OpenAI-compatible embedding client with batching, retry and cache

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Seiten laden und Haupttext extrahieren (Trafilatura)

**Files:**
- Create: `hreflang_matcher/fetch.py`
- Test: `tests/test_fetch.py`

**Interfaces:**
- Consumes: `models.FetchResult`
- Produces:
  - `fetch.USER_AGENT = "ONE-hreflang-Matcher/1.0 (+https://onebeyondsearch.com)"`
  - `fetch.extract_main_text(html: str, url: str | None = None) -> str` (leer, wenn nichts extrahiert)
  - `fetch.fetch_one(url: str, timeout: float = 10.0, session=None) -> tuple[str, str | None]` → (Text, Grund); genau eines von beiden ist leer bzw. None
  - `fetch.fetch_main_texts(urls: dict[int, str], concurrency: int = 5, timeout: float = 10.0, on_progress=None, session_factory=None) -> FetchResult`

- [ ] **Step 1: Fehlschlagende Tests mit Fixture und lokalem Server**

`tests/test_fetch.py`:
```python
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hreflang_matcher import fetch

ARTICLE_HTML = """<!doctype html><html lang="de"><head><title>Küche renovieren</title></head><body>
<nav><a href="/">Start</a> <a href="/kontakt">Kontakt</a></nav>
<main><article>
<h1>Küche renovieren: Wie gelingt die perfekte Modernisierung?</h1>
<p>Alte Küche, frische Optik: Ist Ihre Küche in die Jahre gekommen, muss sie nicht zwingend eine neue her.
Mit kreativen Ideen und gezielten Maßnahmen lässt sich Ihre Küche kostengünstig modernisieren.</p>
<p>Erfahren Sie, wie Sie durch die Renovierung von Fronten, Arbeitsplatten und Wänden einen völlig neuen Look
schaffen können. Wir zeigen Möglichkeiten der Küchenrenovierung, Tipps und Inspiration sowie die Kosten.</p>
<p>Küchengeräte tauschen, Wände und Böden erneuern oder nur die Fronten lackieren: Jede Maßnahme hat ihren Preis
und ihren Effekt. Planen Sie Schritt für Schritt und holen Sie sich Angebote ein.</p>
</article></main>
<footer><p>Impressum · Datenschutz</p></footer></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ok":
            body = ARTICLE_HTML.encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        elif self.path == "/pdf":
            body = b"%PDF-1.4"
            self.send_response(200); self.send_header("Content-Type", "application/pdf")
        elif self.path == "/empty":
            body = b"<html><body><nav>Start Kontakt</nav></body></html>"
            self.send_response(200); self.send_header("Content-Type", "text/html")
        elif self.path == "/slow":
            time.sleep(1.5)
            body = b"<html></html>"
            self.send_response(200); self.send_header("Content-Type", "text/html")
        else:
            body = b"nope"
            self.send_response(404); self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def base():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_extract_main_text_drops_nav_and_footer():
    text = fetch.extract_main_text(ARTICLE_HTML, "https://example.com/de/kueche")
    assert "Küche renovieren" in text and "kostengünstig modernisieren" in text
    assert "Impressum" not in text and "Kontakt" not in text


def test_extract_main_text_empty_for_boilerplate_only():
    assert fetch.extract_main_text("<html><body><nav>Start</nav></body></html>") == ""


def test_fetch_one_ok(base):
    text, err = fetch.fetch_one(base + "/ok")
    assert err is None and "Modernisierung" in text


@pytest.mark.parametrize("path,reason", [
    ("/missing", "HTTP 404"), ("/pdf", "kein HTML (application/pdf)"), ("/empty", "kein Text extrahiert"),
])
def test_fetch_one_reasons(base, path, reason):
    text, err = fetch.fetch_one(base + path)
    assert text == "" and err == reason


def test_fetch_one_timeout(base):
    text, err = fetch.fetch_one(base + "/slow", timeout=0.3)
    assert text == "" and err == "Timeout"


def test_fetch_one_connection_error():
    text, err = fetch.fetch_one("http://127.0.0.1:9/x", timeout=0.5)
    assert text == "" and err.startswith("Verbindungsfehler:")


def test_fetch_main_texts_parallel_keeps_indices(base):
    urls = {0: base + "/ok", 1: base + "/missing", 2: base + "/ok", 3: base + "/pdf"}
    progress = []
    res = fetch.fetch_main_texts(urls, concurrency=3, timeout=5, on_progress=lambda d, t: progress.append((d, t)))
    assert set(res.texts) == {0, 2} and res.errors == {1: "HTTP 404", 3: "kein HTML (application/pdf)"}
    assert progress[-1] == (4, 4)


def test_user_agent_constant():
    assert fetch.USER_AGENT == "ONE-hreflang-Matcher/1.0 (+https://onebeyondsearch.com)"
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_fetch.py -q` → ModuleNotFoundError.

- [ ] **Step 3: fetch.py implementieren**

```python
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
    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=False, favor_precision=True)
    return (text or "").strip()


def fetch_one(url: str, timeout: float = 10.0, session: requests.Session | None = None) -> tuple[str, str | None]:
    """Lädt eine Seite und extrahiert den Haupttext. Rückgabe (Text, None) oder ("", Grund)."""
    session = session or requests.Session()
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
        text, err = fetch_one(url, timeout=timeout, session=session_factory())
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
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest -q -W error` → alle PASS. Falls `test_extract_main_text_drops_nav_and_footer` an Trafilaturas Mindestlänge scheitert, die Fixture um einen weiteren Absatz verlängern, nicht die Extraktionsparameter ändern.

- [ ] **Step 5: Commit**

```bash
git add hreflang_matcher/fetch.py tests/test_fetch.py
git commit -m "feat: fetch pages and extract main text with Trafilatura

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Integration in app.py (Embedding-Quelle, Einstellungen, Berechnung, Export)

**Files:**
- Modify: `app.py` (Imports; Konstanten nach `FILE_TYPES`; Abschnitt „1. Upload“ nach dem `mode`-Radio; Schleife in „2. Sprachzuordnung“; Block vor der Validierung `codes = [...]`; Hilfetext `HELP_MD`)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `textprep.suggest_text_columns`, `textprep.build_texts`, `textprep.truncate`, `textprep.estimate_tokens`; `embed.PRESETS`, `embed.embed_texts`, `embed.test_connection`, `embed.EmbeddingError`; `fetch.fetch_main_texts`; `io_utils.row_filter_reasons`, `io_utils.language_set_from_vectors`, `io_utils.embeddings_csv`; `models.EmbeddingConfig`
- Produces: UI-Konstanten `EMB_IN_FILE`, `EMB_FROM_TEXT`, `EMB_FROM_PAGES`; Session-Keys `emb_cache` (dict), `computed` ({"sets": list[LanguageSet], "sig": tuple}); Funktionen `render_embedding_settings() -> EmbeddingConfig`, `compute_language_sets(pending, emb_source, cfg, filter_idx) -> list[LanguageSet]`, `render_set_metrics(ls)`

- [ ] **Step 1: AppTest-Wrapper erweitern und fehlschlagende Tests schreiben**

In `tests/test_app.py` den `_WRAPPER`-String so erweitern, dass `embed.embed_texts` und `embed.test_connection` gefälscht werden (kein Netz). Neuer Wrapper (ersetzt den bestehenden vollständig):
```python
_WRAPPER = f'''
import hashlib
import numpy as np
import streamlit as st
import hreflang_matcher.embed as _emb
from hreflang_matcher.models import EmbedResult

class _FakeUpload:
    def __init__(self, name, data, file_id):
        self.name, self._data, self.file_id = name, data, file_id
    def getvalue(self):
        return self._data

_orig_uploader = st.file_uploader
def _fake_uploader(*args, **kwargs):
    if kwargs.get("accept_multiple_files"):
        return [_FakeUpload(*t) for t in st.session_state.get("_fake_files", [])]
    single = st.session_state.get("_fake_single")
    return _FakeUpload(*single) if single else None
st.file_uploader = _fake_uploader

def _fake_vec(text):
    h = hashlib.sha1(text.encode("utf-8")).digest()
    return np.asarray([b / 255.0 for b in h[:8]], dtype=np.float32)

def _fake_embed_texts(texts, cfg, cache=None, on_progress=None, **kw):
    st.session_state["_embed_calls"] = st.session_state.get("_embed_calls", 0) + 1
    st.session_state["_embed_cfg"] = (cfg.base_url, cfg.model, cfg.max_chars)
    res = EmbedResult(vectors={{i: _fake_vec(t) for i, t in texts.items()}}, dim=8)
    if on_progress:
        on_progress(len(texts), len(texts))
    return res

_orig_embed, _orig_test = _emb.embed_texts, _emb.test_connection
_emb.embed_texts = _fake_embed_texts
_emb.test_connection = lambda cfg: 8
with open({APP!r}, encoding="utf-8") as _fh:
    _src = _fh.read()
try:
    exec(compile(_src, {APP!r}, "exec"))
finally:
    st.file_uploader = _orig_uploader
    _emb.embed_texts, _emb.test_connection = _orig_embed, _orig_test
'''
```
Hinweis: Im f-String müssen die Dict-Klammern der Comprehension verdoppelt werden (`{{i: ...}}`), wie oben gezeigt.

Hilfsfunktion und Tests anhängen:
```python
def _text_csv(lang: str) -> bytes:
    rows = ["Address;Title;Main Content Extraction 1"]
    for i in range(3):
        rows.append(f"https://a.com/{lang}/p{i}/;Titel {lang} {i};Inhalt {lang} Seite {i} " + "lorem " * 20)
    return "\n".join(rows).encode("utf-8")


def _radio(at, label):
    return next(r for r in at.radio if r.label == label)


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_text_source_requires_explicit_compute_click():
    at = AppTest.from_string(_WRAPPER, default_timeout=60)
    at.session_state["_fake_files"] = [("de.csv", _text_csv("de"), "id-de"), ("fr.csv", _text_csv("fr"), "id-fr")]
    at.run()
    _radio(at, "Embedding-Quelle").set_value("Embeddings aus Textspalten berechnen").run()
    assert not at.exception, at.exception
    assert at.session_state.get("_embed_calls", 0) == 0
    assert any("Embeddings berechnen" in i.value for i in at.info)
    assert not any(b.label == "Let's Go" for b in at.button)


def test_text_source_full_run_to_results_and_export():
    at = AppTest.from_string(_WRAPPER, default_timeout=60)
    at.session_state["_fake_files"] = [("de.csv", _text_csv("de"), "id-de"), ("fr.csv", _text_csv("fr"), "id-fr")]
    at.run()
    _radio(at, "Embedding-Quelle").set_value("Embeddings aus Textspalten berechnen").run()
    next(t for t in at.text_input if t.label == "API-Key").set_value("sk-test").run()
    _button(at, "Embeddings berechnen").click().run()
    assert not at.exception, at.exception
    assert at.session_state["_embed_calls"] == 2            # ein Aufruf je Sprachvariante
    assert at.session_state["_embed_cfg"] == ("https://api.openai.com/v1", "text-embedding-3-small", 20000)
    comp = at.session_state["computed"]
    assert [s.code for s in comp["sets"]] == ["de", "fr"] and all(len(s) == 3 and s.dim == 8 for s in comp["sets"])
    _button(at, "Let's Go").click().run()
    assert not at.exception, at.exception
    assert any(h.value == "4. Ergebnisse" for h in at.subheader)
    assert at.session_state["_embed_calls"] == 2            # Let's Go löst keine neue Berechnung aus


def test_text_source_settings_change_marks_embeddings_stale():
    at = AppTest.from_string(_WRAPPER, default_timeout=60)
    at.session_state["_fake_files"] = [("de.csv", _text_csv("de"), "id-de"), ("fr.csv", _text_csv("fr"), "id-fr")]
    at.run()
    _radio(at, "Embedding-Quelle").set_value("Embeddings aus Textspalten berechnen").run()
    next(t for t in at.text_input if t.label == "API-Key").set_value("sk-test").run()
    _button(at, "Embeddings berechnen").click().run()
    next(s for s in at.slider if s.label.startswith("Max. Zeichen")).set_value(4000).run()
    assert not at.exception, at.exception
    assert any("veraltet" in w.value for w in at.warning)
    assert not any(b.label == "Let's Go" for b in at.button)
    assert at.session_state["_embed_calls"] == 2


def test_in_file_source_unchanged():
    at = AppTest.from_string(_WRAPPER, default_timeout=60)
    at.session_state["_fake_files"] = [("de.csv", _csv("de"), "id-de"), ("fr.csv", _csv("fr"), "id-fr")]
    at.run()
    assert _radio(at, "Embedding-Quelle").value == "Embeddings sind in der Datei enthalten"
    assert not any(b.label == "Embeddings berechnen" for b in at.button)
    _button(at, "Let's Go").click().run()
    assert not at.exception, at.exception
    assert at.session_state.get("_embed_calls", 0) == 0
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_app.py -q` → die vier neuen Tests FAIL (kein Radio „Embedding-Quelle“), die alten bestehen weiter (der Wrapper ändert nichts am alten Pfad).

- [ ] **Step 3: app.py ändern**

**3a. Imports** (Zeilen 8–9 ersetzen):
```python
from hreflang_matcher import embed, fetch, io_utils, lang_detect, matching, output, textprep
from hreflang_matcher.models import EmbeddingConfig, LanguageSet
```

**3b. Konstanten** nach `FILE_TYPES = [...]`:
```python
EMB_IN_FILE = "Embeddings sind in der Datei enthalten"
EMB_FROM_TEXT = "Embeddings aus Textspalten berechnen"
EMB_FROM_PAGES = "Embeddings aus den Seiteninhalten berechnen (Trafilatura)"
FETCH_WARN_URLS = 500
```

**3c. Helfer** nach `warn_bad_lines` einfügen:
```python
def render_set_metrics(ls: LanguageSet) -> None:
    m1, m2, m3 = st.columns(3)
    m1.metric("Gültige URLs", len(ls))
    m2.metric("Embedding-Dimension", ls.dim)
    m3.metric("Verworfen", len(ls.dropped))
    if len(ls.dropped):
        with st.expander(f"Verworfene Zeilen ({len(ls.dropped)})"):
            st.dataframe(ls.dropped, width="stretch", hide_index=True)
    if len(ls) == 0:
        st.error("Keine gültigen URLs mit Embeddings in dieser Sprachvariante.")


def render_embedding_settings() -> EmbeddingConfig:
    """Expander mit Preset, Base-URL, Modell, Key, Kürzung, Verbindungstest, Cache leeren."""
    with st.expander("⚙️ Embedding-Einstellungen", expanded=True):
        preset = st.selectbox("Preset", list(embed.PRESETS), key="emb_preset")
        p = embed.PRESETS[preset]
        c1, c2 = st.columns(2)
        with c1:
            base_url = st.text_input("Base-URL", value=p["base_url"], key=f"emb_base_{preset}",
                                     help="OpenAI-kompatibler Endpunkt; das Tool ruft {Base-URL}/embeddings auf.")
        with c2:
            if p["models"]:
                model = st.selectbox("Modell", p["models"], key=f"emb_model_{preset}")
            else:
                model = st.text_input("Modell", value="", key=f"emb_model_{preset}", help="Modellname laut Endpunkt, z. B. bge-m3")
        api_key = st.text_input("API-Key", type="password", key="emb_key",
                                help="Nur in dieser Session im Speicher. Bei lokalen Servern meist leer.")
        max_chars = st.slider("Max. Zeichen je Text (Kürzung vor dem Versand)", 2000, 60000, 20000, 1000, key="emb_max_chars")
        if p["local"]:
            st.info("Lokale Endpunkte funktionieren nur, wenn das Tool auf demselben Rechner läuft. "
                    "Streamlit Cloud erreicht dein Localhost nicht.")
        cfg = EmbeddingConfig(base_url=base_url.strip(), model=model.strip(), api_key=api_key.strip(), max_chars=int(max_chars))
        b1, b2 = st.columns(2)
        if b1.button("Verbindung testen", key="emb_test"):
            if p["needs_key"] and not cfg.api_key:
                st.error("Für dieses Preset wird ein API-Key benötigt.")
            else:
                try:
                    st.success(f"Verbindung OK, Dimension {embed.test_connection(cfg)}")
                except embed.EmbeddingError as e:
                    st.error(f"Verbindung fehlgeschlagen: {e}")
        if b2.button("Embedding-Cache leeren", key="emb_clear"):
            st.session_state["emb_cache"] = {}
            st.session_state.pop("computed", None)
            st.success("Cache geleert.")
    return cfg


def compute_language_sets(pending: list[dict], emb_source: str, cfg: EmbeddingConfig, filter_idx: bool) -> list[LanguageSet]:
    """Texte bauen oder abrufen, Embeddings berechnen, LanguageSets bilden. Wirft EmbeddingError bei Totalausfall."""
    cache = st.session_state.setdefault("emb_cache", {})
    out: list[LanguageSet] = []
    for p in pending:
        df = p["df"]
        st.write(f"**{p['label']}**")
        bar = st.progress(0.0)
        keep, drop = io_utils.row_filter_reasons(df, p["url_col"], filter_idx, p["status_col"], p["index_col"])
        errors: dict[int, str] = {}
        if emb_source == EMB_FROM_TEXT:
            texts, errs = textprep.build_texts(df, p["text_cols"], cfg.max_chars)
            texts = {i: t for i, t in texts.items() if i in keep}
            errors.update({i: r for i, r in errs.items() if i in keep})
        else:
            bar.progress(0.0, text="Seiten werden geladen …")
            fr = fetch.fetch_main_texts(
                {i: keep[i] for i in keep}, concurrency=5, timeout=10.0,
                on_progress=lambda d, t: bar.progress(0.5 * d / max(t, 1), text=f"Seiten geladen: {d}/{t}"),
            )
            texts = {i: textprep.truncate(t, cfg.max_chars) for i, t in fr.texts.items()}
            errors.update(fr.errors)
        base = 0.5 if emb_source == EMB_FROM_PAGES else 0.0
        res = embed.embed_texts(
            texts, cfg, cache=cache,
            on_progress=lambda d, t: bar.progress(base + (1 - base) * d / max(t, 1), text=f"Embeddings: {d}/{t}"),
        )
        if texts and not res.vectors:
            raise embed.EmbeddingError(next(iter(res.errors.values()), "keine Vektoren erhalten"))
        errors.update(res.errors)
        bar.progress(1.0, text="fertig")
        out.append(io_utils.language_set_from_vectors(
            df, p["code"], p["label"], p["url_col"], res.vectors, errors, filter_idx, p["status_col"], p["index_col"],
        ))
    return out
```

**3d. Abschnitt 1** – direkt nach `mode = st.radio(...)` einfügen:
```python
emb_source = st.radio(
    "Embedding-Quelle",
    [EMB_IN_FILE, EMB_FROM_TEXT, EMB_FROM_PAGES],
    index=0,
    help="Standard: Embeddings stehen bereits in der Datei. Alternativ berechnet das Tool sie aus Textspalten "
         "oder lädt die Seiten selbst und extrahiert den Haupttext.",
)
```
Die beiden `help=`-Texte der Uploader bleiben; nur bei `EMB_IN_FILE` ist die Embedding-Spalte Pflicht, deshalb den Mehrfach-Uploader-Help so ändern: `"Mindestens eine URL-Spalte; bei Quelle „in der Datei enthalten“ zusätzlich eine Embedding-Spalte. Weitere Spalten werden ignoriert."`

Nach der `filter_idx`-Checkbox:
```python
cfg: EmbeddingConfig | None = None
if emb_source != EMB_IN_FILE:
    cfg = render_embedding_settings()
```

**3e. Abschnitt 2 – Schleife.** Vor der Schleife `pending: list[dict] = []` ergänzen. Innerhalb des Containers den Block von `c1, c2, c3 = st.columns([1, 2, 2])` bis zum Ende des Containers durch Folgendes ersetzen (Code-Teil in `c1` bleibt wie bisher):
```python
        c1, c2, c3 = st.columns([1, 2, 2])
        with c1:
            if src["code_default"] is not None:
                code = src["code_default"]
                st.text_input("hreflang-Code", value=code, disabled=True, key=f"{key}_code",
                              help="Wird in der Mustertabelle oben festgelegt.")
            else:
                code_default = lang_detect.suggest_code(df[url_guess].astype(str).tolist()) if url_guess else ""
                code = st.text_input("hreflang-Code", value=code_default, key=f"{key}_code", help="z. B. de, de-AT, fr-CH")
        with c2:
            url_col = st.selectbox("URL-Spalte", cols, index=cols.index(url_guess) if url_guess in cols else 0, key=f"{key}_url")
        emb_col = None
        text_cols: list[str] = []
        with c3:
            if emb_source == EMB_IN_FILE:
                emb_col = st.selectbox("Embedding-Spalte", cols, index=cols.index(emb_guess) if emb_guess in cols else 0, key=f"{key}_emb")
            elif emb_source == EMB_FROM_TEXT:
                suggested = textprep.suggest_text_columns(df)
                text_cols = st.multiselect("Textspalten (Reihenfolge = Reihenfolge im Text)", cols, default=suggested,
                                           key=f"{key}_textcols")
            else:
                st.caption(f"{len(df)} URLs werden geladen und mit Trafilatura extrahiert.")
                if len(df) > FETCH_WARN_URLS:
                    st.warning(f"{len(df)} URLs: Der Abruf dauert lange und die Session muss offen bleiben. "
                               "Für große Sites ist der Crawl mit Screaming Frog der bessere Weg.")

        if emb_source == EMB_IN_FILE and emb_guess is None:
            st.warning("Keine Embedding-Spalte automatisch erkannt. Bitte manuell auswählen.")
        if emb_source == EMB_FROM_TEXT:
            if not text_cols:
                st.warning("Bitte mindestens eine Textspalte wählen.")
            else:
                _t, _ = textprep.build_texts(df, text_cols, cfg.max_chars)
                avg = int(sum(len(t) for t in _t.values()) / max(len(_t), 1))
                st.caption(f"{len(_t)} Texte, Ø {avg} Zeichen, ca. {textprep.estimate_tokens(_t.values())} Tokens")
        if filter_idx:
            missing = [n for n, c in (("Status-Code", status_col), ("Indexierbarkeit", index_col)) if c is None]
            if missing:
                st.warning(f"Spalte {' und '.join(missing)} nicht gefunden, dieser Teilfilter wird übersprungen.")

        code_n = lang_detect.normalize_code(code)
        if not lang_detect.is_valid_code(code_n):
            st.error("Ungültiger hreflang-Code. Erlaubt: Sprache (de) oder Sprache-Region (de-AT).")
            invalid_codes += 1
            continue

        if emb_source == EMB_IN_FILE:
            ls = build_cached(src["name"], src["raw"], code_n, src["label"], url_col, emb_col, filter_idx,
                              status_col, index_col, src["row_idx"], io_utils.detect_note_column(df))
            render_set_metrics(ls)
            sets.append(ls)
            set_meta.append({"key": key, "file_key": src["file_key"], "code": ls.code, "len": len(ls), "dim": ls.dim,
                             "url_col": url_col, "emb_col": emb_col})
        else:
            pending.append({"df": df, "code": code_n, "label": src["label"], "url_col": url_col, "text_cols": text_cols,
                            "status_col": status_col, "index_col": index_col, "key": key, "file_key": src["file_key"]})
```

**3f. Berechnung** – nach der Schleife, vor `codes = [s.code for s in sets]` einfügen:
```python
if emb_source != EMB_IN_FILE and invalid_codes == 0:
    if emb_source == EMB_FROM_TEXT and any(not p["text_cols"] for p in pending):
        st.stop()
    emb_sig = (emb_source, cfg.base_url, cfg.model, cfg.max_chars, filter_idx,
               tuple((p["key"], p["file_key"], p["code"], tuple(p["text_cols"]), p["url_col"]) for p in pending))
    n_rows = sum(len(p["df"]) for p in pending)
    hint = " text-embedding-3-small kostet ca. 0,02 $ je 1 Mio. Tokens." if "openai.com" in cfg.base_url else ""
    st.caption(f"{n_rows} Zeilen in {len(pending)} Sprachvariante(n). Die Berechnung startet nur über den Button.{hint}")
    if st.button("Embeddings berechnen", type="primary", key="emb_compute"):
        if embed.PRESETS[st.session_state.get("emb_preset", "OpenAI")]["needs_key"] and not cfg.api_key:
            st.error("Für dieses Preset wird ein API-Key benötigt.")
        elif not cfg.base_url or not cfg.model:
            st.error("Base-URL und Modell müssen gesetzt sein.")
        else:
            try:
                with st.spinner("Embeddings werden berechnet …"):
                    st.session_state["computed"] = {"sets": compute_language_sets(pending, emb_source, cfg, filter_idx),
                                                    "sig": emb_sig}
                st.session_state.pop("result", None)
            except embed.EmbeddingError as e:
                st.session_state.pop("computed", None)
                st.error(f"Verbindung zum Endpunkt fehlgeschlagen ({cfg.base_url}): {e}")
    comp = st.session_state.get("computed")
    if not comp:
        st.info("Bitte auf „Embeddings berechnen“ klicken, um fortzufahren.")
        st.stop()
    if comp["sig"] != emb_sig:
        st.warning("Die Embeddings sind veraltet (Spalten, Dateien oder Einstellungen haben sich geändert). "
                   "Bitte erneut auf „Embeddings berechnen“ klicken.")
        st.stop()
    for ls, p in zip(comp["sets"], pending):
        with st.container(border=True):
            st.markdown(f"**{p['label']}** – berechnet mit {cfg.model}")
            render_set_metrics(ls)
            st.download_button(f"📥 Embeddings als CSV ({ls.code})", io_utils.embeddings_csv(ls),
                               f"embeddings_{ls.code}.csv", "text/csv", key=f"dl_emb_{p['key']}", on_click="ignore",
                               help="Im Modus „Embeddings sind in der Datei enthalten“ direkt wieder verwendbar.")
        sets.append(ls)
        set_meta.append({"key": p["key"], "file_key": p["file_key"], "code": ls.code, "len": len(ls), "dim": ls.dim,
                         "url_col": p["url_col"], "emb_col": f"berechnet:{cfg.model}"})
```

**3g. Hilfetext** – in `HELP_MD` nach dem Absatz „**Eingabe:** …“ einfügen:
```markdown
**Embedding-Quelle:** Standard ist „in der Datei enthalten“ (z. B. aus Screaming Frog). Alternativ berechnet das
Tool die Embeddings selbst: aus gewählten Textspalten (Main Content, Title, H1 …) oder aus den Seiteninhalten,
die es mit Trafilatura extrahiert. Dafür brauchst du einen OpenAI-kompatiblen Endpunkt: OpenAI mit API-Key,
oder lokal Ollama/LM Studio (nur wenn das Tool auf demselben Rechner läuft). Texte werden vor dem Versand
gekürzt, die Berechnung startet nur per Button, und die fertigen Embeddings kannst du als CSV sichern und
beim nächsten Mal ohne API-Kosten hochladen.
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest tests/test_app.py -q -W error` → alle PASS (alte + 4 neue). Dann `.venv/bin/python -m pytest -q -W error` → alle PASS.

- [ ] **Step 5: Headless-Start**

Run: `.venv/bin/python -m streamlit run app.py --server.headless true --server.port 8598 > /tmp/st8598.log 2>&1 & PID=$!; sleep 8; curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8598; kill $PID; grep -ci "traceback" /tmp/st8598.log` → `200` und `0`.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: compute embeddings in the app from text columns or page content

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: README, Spec-Abgleich, Abschluss

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README ergänzen**

Im Abschnitt „Eingabedaten“ nach dem Punkt zu leeren Embeddings:
```markdown
- Alternativ berechnet das Tool die Embeddings selbst („Embedding-Quelle“): aus Textspalten
  (z. B. Main Content Extraction, Title, H1) oder aus den Seiteninhalten per Trafilatura. Dafür ist
  ein OpenAI-kompatibler Endpunkt nötig: OpenAI (API-Key), ein gehosteter Dienst oder lokal
  Ollama/LM Studio (nur bei lokalem Start des Tools). Berechnete Embeddings lassen sich als CSV
  exportieren und später ohne API-Kosten wieder hochladen.
```
Im Abschnitt „Ablauf“ Punkt 1 ergänzen: „Bei Quelle „berechnen“: Textspalten wählen oder URLs laden, Endpunkt einstellen, „Embeddings berechnen“ klicken.“

Im Abschnitt „Grenzen“ ergänzen:
```markdown
- Trafilatura-Modus: realistisch bis einige hundert URLs je Lauf; ab 500 URLs warnt das Tool.
  Seiten, die JavaScript zum Rendern brauchen, liefern wenig oder keinen Text.
```

- [ ] **Step 2: Spec-Abgleich**

Jede Anforderung aus `docs/superpowers/specs/2026-10-15-embedding-source-design.md` Abschnitt 5.1–5.9 gegen `app.py`, `embed.py`, `fetch.py`, `textprep.py`, `io_utils.py` prüfen. Abweichungen im Code beheben, nicht in der Spec.

- [ ] **Step 3: Gesamte Suite und Lokal-Check**

Run: `.venv/bin/python -m pytest -q -W error` → alle PASS, keine Warnungen.
Run (optional, nur wenn Ollama lokal läuft): App starten, Quelle „Textspalten“, Preset „Ollama (lokal)“, „Verbindung testen“ → „Verbindung OK, Dimension 1024“.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: describe in-tool embedding computation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
