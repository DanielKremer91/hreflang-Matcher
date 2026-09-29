# ONE hreflang Matcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eine Streamlit-App, die Sprachvarianten einer Website per Embedding-Cosinus zu hreflang-Clustern zuordnet und HTML-Tags plus CSV exportiert.

**Architecture:** Reine Fachlogik in `hreflang_matcher/` (io_utils, lang_detect, matching, output) ohne Streamlit-Import, mit pytest getestet. `app.py` orchestriert nur UI und Session-State. Matching läuft blockweise in numpy gegen eine Pivot-Sprache mit greedy 1:1-Zuordnung.

**Tech Stack:** Python 3.10+, streamlit, pandas, numpy, openpyxl, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-hreflang-matcher-design.md`

## Global Constraints

- Abhängigkeiten nur: `streamlit>=1.36`, `pandas>=2.0`, `numpy>=1.26`, `openpyxl>=3.1`, dev: `pytest>=8`. Kein torch, keine sentence-transformers, kein faiss.
- Module unter `hreflang_matcher/` importieren **nie** `streamlit`.
- Alle UI-Texte Deutsch. Konfidenzwerte exakt `"sicher"` und `"prüfen"`. Methodenwerte exakt `"slug"` und `"embedding"`.
- Vektoren werden bei abweichender Dimension **verworfen**, nie gepaddet.
- Ausgabe-URLs sind immer Original-URLs, normalisierte URLs nur intern.
- Niemals die volle n×m-Ähnlichkeitsmatrix im Speicher halten; Zeilenblock 1024, Spaltenblock 4096.
- Commits: `git add` der genannten Dateien, Commit-Message wie angegeben, Endzeile `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Arbeitsverzeichnis für alle Befehle: `/Users/Daniel/Downloads/one-hreflang-matcher`. Tests laufen mit `.venv/bin/python -m pytest`.

---

## File Structure

| Datei | Verantwortung |
|---|---|
| `requirements.txt` | Laufzeit-Abhängigkeiten für Streamlit Cloud |
| `requirements-dev.txt` | `-r requirements.txt` plus pytest |
| `pyproject.toml` | pytest-Konfiguration (`testpaths = ["tests"]`) |
| `.gitignore` | `.venv/`, `__pycache__/`, `.pytest_cache/` |
| `.streamlit/config.toml` | Theme mit Primärfarbe `#d7263d` |
| `hreflang_matcher/__init__.py` | leer |
| `hreflang_matcher/models.py` | Dataclasses `LanguageSet`, `Match`, `MatchResult`, `Cluster`, `PatternSuggestion` |
| `hreflang_matcher/io_utils.py` | Datei lesen, Spalten erkennen, Vektoren parsen, URL normalisieren, `LanguageSet` bauen, Dimensions-Check |
| `hreflang_matcher/lang_detect.py` | hreflang-Code-Normalisierung und -Vorschlag, Sprachkollisionen, Sprachsegment entfernen, Muster-Vorschlag und Split |
| `hreflang_matcher/matching.py` | Top-k-Cosinus blockweise, Slug-Match, Embedding-Match mit greedy 1:1, `run_matching` |
| `hreflang_matcher/output.py` | Cluster, Mapping-Tabelle, Unmatched-Tabelle, hreflang-Code-Format, HTML-Blöcke |
| `app.py` | Streamlit-UI |
| `tests/test_io_utils.py`, `tests/test_lang_detect.py`, `tests/test_matching.py`, `tests/test_output.py`, `tests/test_app.py` | pytest |
| `README.md` | Nutzung, Deployment, Datenformat |

---

### Task 1: Projektgerüst, Datenmodell, URL-Normalisierung, Vektor-Parsing

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `.gitignore`, `.streamlit/config.toml`
- Create: `hreflang_matcher/__init__.py`, `hreflang_matcher/models.py`, `hreflang_matcher/io_utils.py`
- Test: `tests/__init__.py`, `tests/test_io_utils.py`

**Interfaces:**
- Produces: `models.LanguageSet(code, label, urls, norm_urls, vectors, dropped)`, `models.Match(...)`, `models.MatchResult(...)`, `models.Cluster(...)`, `models.PatternSuggestion(...)` (exakte Felder unten)
- Produces: `io_utils.normalize_url(url: str) -> str`, `io_utils.parse_vector(value) -> np.ndarray | None`

- [ ] **Step 1: Gerüstdateien anlegen**

`requirements.txt`:
```
streamlit>=1.36
pandas>=2.0
numpy>=1.26
openpyxl>=3.1
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest>=8
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:
```
.venv/
__pycache__/
.pytest_cache/
*.pyc
.DS_Store
```

`.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#d7263d"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f2f2f2"
textColor = "#000000"
```

`hreflang_matcher/__init__.py` und `tests/__init__.py`: leer.

- [ ] **Step 2: Virtuelle Umgebung anlegen und Abhängigkeiten installieren**

Run:
```bash
cd /Users/Daniel/Downloads/one-hreflang-matcher && python3 -m venv .venv && .venv/bin/pip install -q -r requirements-dev.txt && .venv/bin/python -c "import streamlit, pandas, numpy, openpyxl; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Datenmodell schreiben**

`hreflang_matcher/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LanguageSet:
    """Alle gültigen URLs einer Sprachvariante mit L2-normalisierten Vektoren."""

    code: str                 # hreflang-Code, z. B. "de", "de-AT"
    label: str                # Anzeigename (Dateiname oder Muster)
    urls: list[str]           # Original-URLs
    norm_urls: list[str]      # normalisierte URLs, gleiche Reihenfolge
    vectors: np.ndarray       # Form (n, dim), float32, L2-normalisiert
    dropped: pd.DataFrame     # Spalten: "URL", "Grund"

    def __len__(self) -> int:
        return len(self.urls)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0


@dataclass
class Match:
    pivot_url: str
    other_url: str
    other_code: str
    score: float              # 1.0 bei Slug-Match, sonst Cosinus
    method: str               # "slug" | "embedding"
    reciprocal: bool
    second_url: str | None    # zweitbester Kandidat derselben Sprache
    second_score: float | None
    margin: float | None      # score - second_score
    confidence: str           # "sicher" | "prüfen"
    reason: str               # leer bei "sicher"


@dataclass
class MatchResult:
    pivot_code: str
    matches: list[Match]
    unmatched: dict[str, list[str]]        # code -> Original-URLs ohne Zuordnung
    stats: dict[str, dict[str, int]]       # code -> Zähler


@dataclass
class Cluster:
    pivot_url: str
    members: dict[str, str]                # code -> Original-URL, inkl. Pivot
    matches: dict[str, Match]              # code -> Match, ohne Pivot
    confidence: str                        # "sicher" | "prüfen"


@dataclass
class PatternSuggestion:
    pattern: str              # Regex auf die Original-URL
    code: str                 # vorgeschlagener hreflang-Code, evtl. ""
    example: str
    count: int
```

- [ ] **Step 4: Fehlschlagende Tests für normalize_url und parse_vector schreiben**

`tests/test_io_utils.py`:
```python
import numpy as np
import pandas as pd
import pytest

from hreflang_matcher import io_utils


class TestNormalizeUrl:
    def test_strips_scheme_www_trailing_slash_and_tracking(self):
        assert io_utils.normalize_url("https://www.Example.com/de/seite/?utm_source=x") == "example.com/de/seite"

    def test_keeps_root_slash(self):
        assert io_utils.normalize_url("https://example.com/") == "example.com/"
        assert io_utils.normalize_url("https://example.com") == "example.com/"

    def test_sorts_query_and_drops_fragment(self):
        assert io_utils.normalize_url("http://example.com/p?b=2&a=1#top") == "example.com/p?a=1&b=2"

    def test_adds_scheme_when_missing(self):
        assert io_utils.normalize_url("example.com/x/") == "example.com/x"

    def test_empty(self):
        assert io_utils.normalize_url("") == ""
        assert io_utils.normalize_url(None) == ""


class TestParseVector:
    def test_json_array(self):
        v = io_utils.parse_vector("[0.1, -0.2, 3e-3]")
        assert v is not None
        np.testing.assert_allclose(v, [0.1, -0.2, 0.003], rtol=1e-5)
        assert v.dtype == np.float32

    def test_comma_separated_without_brackets(self):
        v = io_utils.parse_vector("0.1,0.2,0.3")
        np.testing.assert_allclose(v, [0.1, 0.2, 0.3], rtol=1e-5)

    def test_semicolon_and_whitespace(self):
        v = io_utils.parse_vector("0.1; 0.2 0.3|0.4")
        assert len(v) == 4

    def test_list_input(self):
        v = io_utils.parse_vector([1, 2, 3])
        assert v.shape == (3,)

    def test_invalid_returns_none(self):
        assert io_utils.parse_vector("") is None
        assert io_utils.parse_vector(None) is None
        assert io_utils.parse_vector(float("nan")) is None
        assert io_utils.parse_vector("abc") is None
```

- [ ] **Step 5: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_io_utils.py -v`
Expected: FAIL mit `AttributeError: module 'hreflang_matcher.io_utils' has no attribute` bzw. ImportError.

- [ ] **Step 6: io_utils mit normalize_url und parse_vector anlegen**

`hreflang_matcher/io_utils.py`:
```python
from __future__ import annotations

import json
import re
from urllib.parse import parse_qsl, urlencode, urlparse

import numpy as np
import pandas as pd

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "pk_campaign", "pk_kwd",
}

_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def normalize_url(url) -> str:
    """Normalisiert eine URL für das Matching. Ergebnis ohne Schema, z. B. 'example.com/de/seite'."""
    s = str(url or "").strip()
    if not s or s.lower() == "nan":
        return ""
    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s
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


def parse_vector(value) -> np.ndarray | None:
    """Parst einen Embedding-Wert (JSON-Array, Zahlenliste, list) zu float32. None wenn nicht lesbar."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32)
        return arr if arr.size > 0 else None
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
```

- [ ] **Step 7: Tests laufen lassen**

Run: `.venv/bin/python -m pytest tests/test_io_utils.py -v`
Expected: alle PASS.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt requirements-dev.txt pyproject.toml .gitignore .streamlit/config.toml hreflang_matcher/__init__.py hreflang_matcher/models.py hreflang_matcher/io_utils.py tests/__init__.py tests/test_io_utils.py
git commit -m "feat: project scaffold, data model, url normalization and vector parsing

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Datei lesen, Spaltenerkennung, LanguageSet bauen, Dimensions-Check

**Files:**
- Modify: `hreflang_matcher/io_utils.py`
- Test: `tests/test_io_utils.py`

**Interfaces:**
- Consumes: `models.LanguageSet`, `io_utils.normalize_url`, `io_utils.parse_vector`
- Produces:
  - `read_any_file(name: str, raw: bytes) -> pd.DataFrame` (wirft `ValueError` bei unlesbarer Datei)
  - `detect_url_column(df) -> str | None`
  - `detect_embedding_column(df) -> str | None`
  - `detect_status_column(df) -> str | None`
  - `detect_indexability_column(df) -> str | None`
  - `is_indexable(value) -> bool`
  - `build_language_set(df, code, label, url_col, emb_col, filter_indexable, status_col=None, index_col=None) -> LanguageSet`
  - `check_dimensions(sets: list[LanguageSet]) -> None` (wirft `ValueError` mit Liste je Set)

- [ ] **Step 1: Fehlschlagende Tests schreiben (an `tests/test_io_utils.py` anhängen)**

```python
def _vec(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return "[" + ", ".join(f"{x:.4f}" for x in rng.normal(size=n)) + "]"


class TestReadAnyFile:
    def test_csv_utf8_comma(self):
        raw = "Address,Embeddings\nhttps://a.de/x,\"[0.1, 0.2]\"\n".encode("utf-8-sig")
        df = io_utils.read_any_file("crawl.csv", raw)
        assert list(df.columns) == ["Address", "Embeddings"]
        assert len(df) == 1

    def test_csv_semicolon_latin1(self):
        raw = "Adresse;Embedding\nhttps://a.de/ü;\"[0.1, 0.2]\"\n".encode("latin1")
        df = io_utils.read_any_file("crawl.csv", raw)
        assert list(df.columns) == ["Adresse", "Embedding"]

    def test_xlsx(self, tmp_path):
        p = tmp_path / "c.xlsx"
        pd.DataFrame({"Address": ["https://a.de/x"], "Embeddings": ["[0.1, 0.2]"]}).to_excel(p, index=False)
        df = io_utils.read_any_file("c.xlsx", p.read_bytes())
        assert "Address" in df.columns

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            io_utils.read_any_file("x.xlsx", b"not an excel file")


class TestDetectColumns:
    def test_url_by_name(self):
        df = pd.DataFrame({"Foo": ["a"], "Address": ["https://a.de/x"]})
        assert io_utils.detect_url_column(df) == "Address"

    def test_url_by_content(self):
        df = pd.DataFrame({"Spalte1": ["https://a.de/x", "https://a.de/y", "https://a.de/z"], "Spalte2": [1, 2, 3]})
        assert io_utils.detect_url_column(df) == "Spalte1"

    def test_url_none(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert io_utils.detect_url_column(df) is None

    def test_embedding_by_name(self):
        df = pd.DataFrame({"Address": ["u"], "OpenAI Embedding 1": [_vec()]})
        assert io_utils.detect_embedding_column(df) == "OpenAI Embedding 1"

    def test_embedding_by_content(self):
        df = pd.DataFrame({"Address": ["u"] * 3, "Extraktion 1": [_vec(), _vec(), _vec()]})
        assert io_utils.detect_embedding_column(df) == "Extraktion 1"

    def test_embedding_ignores_short_numeric(self):
        df = pd.DataFrame({"Address": ["u"] * 3, "Status Code": ["200", "200", "404"]})
        assert io_utils.detect_embedding_column(df) is None

    def test_status_and_indexability_german_and_english(self):
        df = pd.DataFrame({"Statuscode": [200], "Indexierbarkeit": ["Indexierbar"]})
        assert io_utils.detect_status_column(df) == "Statuscode"
        assert io_utils.detect_indexability_column(df) == "Indexierbarkeit"
        df2 = pd.DataFrame({"Status Code": [200], "Indexability": ["Indexable"]})
        assert io_utils.detect_status_column(df2) == "Status Code"
        assert io_utils.detect_indexability_column(df2) == "Indexability"


class TestIsIndexable:
    @pytest.mark.parametrize("v,exp", [
        ("Indexable", True), ("indexable", True), ("Indexierbar", True),
        ("Non-Indexable", False), ("Nicht indexierbar", False), ("", False), (None, False),
    ])
    def test_values(self, v, exp):
        assert io_utils.is_indexable(v) is exp


class TestBuildLanguageSet:
    def _df(self):
        return pd.DataFrame({
            "Address": [
                "https://a.de/1/", "https://a.de/2", "https://a.de/1",  # 3 ist Duplikat von 1
                "https://a.de/3", "https://a.de/4", "https://a.de/5", "",
            ],
            "Embeddings": [_vec(8, 1), _vec(8, 2), _vec(8, 3), _vec(5, 4), "kaputt", _vec(8, 6), _vec(8, 7)],
            "Status Code": [200, 200, 200, 200, 200, 404, 200],
            "Indexability": ["Indexable"] * 7,
        })

    def test_builds_and_drops_with_reasons(self):
        ls = io_utils.build_language_set(
            self._df(), "de", "de.csv", "Address", "Embeddings",
            filter_indexable=True, status_col="Status Code", index_col="Indexability",
        )
        assert ls.urls == ["https://a.de/1/", "https://a.de/2"]
        assert ls.norm_urls == ["a.de/1", "a.de/2"]
        assert ls.vectors.shape == (2, 8)
        np.testing.assert_allclose(np.linalg.norm(ls.vectors, axis=1), [1.0, 1.0], rtol=1e-5)
        reasons = dict(zip(ls.dropped["URL"], ls.dropped["Grund"]))
        assert "Duplikat" in reasons["https://a.de/1"]
        assert "Dimension 5 statt 8" in reasons["https://a.de/3"]
        assert "nicht lesbar" in reasons["https://a.de/4"]
        assert "Status Code 404" in reasons["https://a.de/5"]
        assert "leere URL" in reasons[""]

    def test_no_filter_keeps_404(self):
        ls = io_utils.build_language_set(
            self._df(), "de", "de.csv", "Address", "Embeddings",
            filter_indexable=False, status_col="Status Code", index_col="Indexability",
        )
        assert "https://a.de/5" in ls.urls

    def test_non_indexable_dropped(self):
        df = self._df()
        df.loc[1, "Indexability"] = "Non-Indexable"
        ls = io_utils.build_language_set(
            df, "de", "de.csv", "Address", "Embeddings",
            filter_indexable=True, status_col="Status Code", index_col="Indexability",
        )
        assert "https://a.de/2" not in ls.urls

    def test_empty_result_has_2d_vectors(self):
        df = pd.DataFrame({"Address": ["https://a.de/x"], "Embeddings": ["kaputt"]})
        ls = io_utils.build_language_set(df, "de", "x", "Address", "Embeddings", filter_indexable=False)
        assert ls.vectors.shape == (0, 0)
        assert len(ls) == 0


class TestCheckDimensions:
    def _ls(self, code, dim):
        from hreflang_matcher.models import LanguageSet
        return LanguageSet(code, code, ["u"], ["u"], np.ones((1, dim), dtype=np.float32), pd.DataFrame(columns=["URL", "Grund"]))

    def test_equal_ok(self):
        io_utils.check_dimensions([self._ls("de", 8), self._ls("fr", 8)])

    def test_unequal_raises_with_details(self):
        with pytest.raises(ValueError) as e:
            io_utils.check_dimensions([self._ls("de", 8), self._ls("fr", 16)])
        assert "de: 8" in str(e.value) and "fr: 16" in str(e.value)
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_io_utils.py -v`
Expected: neue Tests FAIL mit AttributeError.

- [ ] **Step 3: Implementierung an `hreflang_matcher/io_utils.py` anhängen**

```python
from collections import Counter
from io import BytesIO

from .models import LanguageSet

URL_COL_NAMES = ["address", "url", "urls", "adresse", "page", "seite", "landing page", "landingpage"]
EMB_COL_HINTS = ["embed", "vector", "vektor"]
STATUS_COL_NAMES = ["status code", "statuscode", "status"]
INDEX_COL_NAMES = ["indexability", "indexierbarkeit"]


def read_any_file(name: str, raw: bytes) -> pd.DataFrame:
    """Liest CSV (Encoding- und Separator-Erkennung) oder Excel aus Bytes. Wirft ValueError."""
    name = (name or "").lower()
    if not raw:
        raise ValueError("Datei ist leer.")
    if name.endswith(".csv") or name.endswith(".txt") or name.endswith(".tsv"):
        last_err = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                df = pd.read_csv(BytesIO(raw), sep=None, engine="python", encoding=enc, on_bad_lines="skip")
                if df.shape[1] >= 1:
                    df.columns = [str(c).strip() for c in df.columns]
                    return df
            except UnicodeDecodeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                for sep in (";", ",", "\t"):
                    try:
                        df = pd.read_csv(BytesIO(raw), sep=sep, engine="python", encoding=enc, on_bad_lines="skip")
                        df.columns = [str(c).strip() for c in df.columns]
                        return df
                    except Exception as e2:
                        last_err = e2
        raise ValueError(f"CSV konnte nicht gelesen werden: {last_err}")
    try:
        df = pd.read_excel(BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Excel konnte nicht gelesen werden: {e}") from e
    df.columns = [str(c).strip() for c in df.columns]
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


def _looks_like_vector_series(series: pd.Series, min_len: int = 8) -> bool:
    sample = series.dropna().astype(str).head(20)
    if len(sample) == 0:
        return False
    for v in sample:
        vec = parse_vector(v)
        if vec is None or vec.size < min_len:
            return False
    return True


def detect_embedding_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if any(h in str(c).lower() for h in EMB_COL_HINTS) and _looks_like_vector_series(df[c]):
            return c
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
) -> LanguageSet:
    """Baut ein LanguageSet: filtert, parst Vektoren, verwirft Ausreißer, dedupliziert, L2-normalisiert."""
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
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest tests/test_io_utils.py -v`
Expected: alle PASS.

- [ ] **Step 5: Commit**

```bash
git add hreflang_matcher/io_utils.py tests/test_io_utils.py
git commit -m "feat: file reading, column detection and LanguageSet builder

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Sprach-Codes, Kollisionen, Sprachsegment entfernen

**Files:**
- Create: `hreflang_matcher/lang_detect.py`
- Test: `tests/test_lang_detect.py`

**Interfaces:**
- Produces:
  - `normalize_code(code: str) -> str` (`"de-at"` → `"de-AT"`, `"DE"` → `"de"`)
  - `is_valid_code(code: str) -> bool`
  - `language_of(code: str) -> str`
  - `find_language_collisions(codes: list[str]) -> dict[str, list[str]]`
  - `suggest_code(urls: list[str]) -> str` (`""` wenn nichts erkannt)
  - `strip_language(norm_url: str, code: str) -> str` (Host durch `*` ersetzt, Sprachsegment entfernt)

- [ ] **Step 1: Fehlschlagende Tests schreiben**

`tests/test_lang_detect.py`:
```python
import pytest

from hreflang_matcher import lang_detect as ld


class TestCodes:
    @pytest.mark.parametrize("raw,exp", [("de-at", "de-AT"), ("DE", "de"), ("fr-CH", "fr-CH"), (" en ", "en")])
    def test_normalize(self, raw, exp):
        assert ld.normalize_code(raw) == exp

    @pytest.mark.parametrize("raw,exp", [("de", True), ("de-AT", True), ("de-at", True), ("x-default", False), ("deu", False), ("", False), ("de_AT", False)])
    def test_valid(self, raw, exp):
        assert ld.is_valid_code(raw) is exp

    def test_language_of(self):
        assert ld.language_of("de-AT") == "de"
        assert ld.language_of("FR") == "fr"

    def test_collisions(self):
        assert ld.find_language_collisions(["de", "de-AT", "fr"]) == {"de": ["de", "de-AT"]}
        assert ld.find_language_collisions(["de", "fr"]) == {}


class TestSuggestCode:
    def test_path_segment(self):
        urls = ["https://a.com/de/x", "https://a.com/de/y", "https://a.com/impressum"]
        assert ld.suggest_code(urls) == "de"

    def test_path_segment_with_region(self):
        assert ld.suggest_code(["https://a.com/de-at/x", "https://a.com/de-at/y"]) == "de-AT"

    def test_subdomain(self):
        assert ld.suggest_code(["https://fr.a.com/x", "https://fr.a.com/y"]) == "fr"

    def test_www_is_not_a_language(self):
        assert ld.suggest_code(["https://www.a.com/x"]) == ""

    def test_cctld(self):
        assert ld.suggest_code(["https://a.fr/x", "https://a.fr/y"]) == "fr"
        assert ld.suggest_code(["https://a.at/x"]) == ""   # Region ohne Sprache -> leer

    def test_nothing(self):
        assert ld.suggest_code(["https://a.com/x"]) == ""
        assert ld.suggest_code([]) == ""


class TestStripLanguage:
    def test_path_segment(self):
        assert ld.strip_language("a.com/de/seite", "de") == "*/seite"
        assert ld.strip_language("a.com/de-at/seite", "de-AT") == "*/seite"
        assert ld.strip_language("a.com/de/", "de") == "*/"

    def test_subdomain(self):
        assert ld.strip_language("fr.a.com/page", "fr") == "*/page"

    def test_cctld_only_host_replaced(self):
        assert ld.strip_language("a.fr/page", "fr") == "*/page"

    def test_other_segment_kept(self):
        assert ld.strip_language("a.com/produkte/x", "de") == "*/produkte/x"

    def test_root(self):
        assert ld.strip_language("a.com/", "de") == "*/"
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_lang_detect.py -v`
Expected: FAIL mit ModuleNotFoundError.

- [ ] **Step 3: Implementierung**

`hreflang_matcher/lang_detect.py`:
```python
from __future__ import annotations

import re
from collections import Counter
from urllib.parse import urlparse

from .models import PatternSuggestion

CODE_RE = re.compile(r"^([a-z]{2})(?:-([a-z]{2}))?$", re.I)

# ccTLD -> Sprache. Leere Werte sind Regionen ohne eindeutige Sprache.
CCTLD_LANG = {
    "de": "de", "fr": "fr", "it": "it", "es": "es", "nl": "nl", "pl": "pl", "pt": "pt",
    "cz": "cs", "dk": "da", "se": "sv", "no": "no", "fi": "fi", "ru": "ru", "jp": "ja",
    "cn": "zh", "uk": "en", "us": "en", "hu": "hu", "ro": "ro", "gr": "el", "tr": "tr",
    "at": "", "ch": "", "be": "", "ca": "", "com": "", "net": "", "org": "", "eu": "",
}


def normalize_code(code: str) -> str:
    s = (code or "").strip()
    m = CODE_RE.match(s)
    if not m:
        return s
    lang, reg = m.group(1).lower(), m.group(2)
    return f"{lang}-{reg.upper()}" if reg else lang


def is_valid_code(code: str) -> bool:
    s = (code or "").strip()
    return bool(CODE_RE.match(s)) and s.lower() != "x-default"


def language_of(code: str) -> str:
    return (code or "").strip().split("-")[0].lower()


def find_language_collisions(codes: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in codes:
        groups.setdefault(language_of(c), []).append(c)
    return {lang: cs for lang, cs in groups.items() if len(cs) > 1}


def _host_and_path(url: str) -> tuple[str, str]:
    s = str(url or "").strip()
    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s
    p = urlparse(s)
    return (p.hostname or "").lower(), (p.path or "/")


def _code_candidate(url: str) -> str:
    host, path = _host_and_path(url)
    segs = [x for x in path.split("/") if x]
    if segs and CODE_RE.match(segs[0]):
        return normalize_code(segs[0])
    parts = host.split(".")
    if len(parts) >= 3 and len(parts[0]) == 2 and parts[0] != "www":
        return parts[0].lower()
    tld = parts[-1] if parts else ""
    return CCTLD_LANG.get(tld, "")


def suggest_code(urls: list[str]) -> str:
    counts = Counter(c for c in (_code_candidate(u) for u in urls) if c)
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def strip_language(norm_url: str, code: str) -> str:
    """Entfernt Host und Sprachsegment aus einer normalisierten URL ('host/path'). Ergebnis '*/rest'."""
    s = norm_url or ""
    slash = s.find("/")
    path = s[slash:] if slash >= 0 else "/"
    lang = language_of(code)
    segs = path.split("/")  # segs[0] == "" wegen führendem "/"
    if len(segs) > 1 and segs[1]:
        first = segs[1].lower()
        if first == code.lower() or first == lang or (CODE_RE.match(first) and language_of(first) == lang):
            rest = "/" + "/".join(segs[2:])
            return "*" + rest
    return "*" + path
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest tests/test_lang_detect.py -v`
Expected: alle PASS.

- [ ] **Step 5: Commit**

```bash
git add hreflang_matcher/lang_detect.py tests/test_lang_detect.py
git commit -m "feat: hreflang code normalization, suggestion and language stripping

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Muster-Vorschlag und Split einer Gesamtdatei

**Files:**
- Modify: `hreflang_matcher/lang_detect.py`
- Test: `tests/test_lang_detect.py`

**Interfaces:**
- Consumes: `models.PatternSuggestion`, `normalize_code`, `CCTLD_LANG`
- Produces:
  - `suggest_patterns(urls: list[str]) -> list[PatternSuggestion]` (nach `count` absteigend)
  - `split_by_patterns(urls: list[str], patterns: list[tuple[str, str]]) -> tuple[dict[str, list[int]], list[int], list[str]]` → (code → Zeilenindizes, Rest-Indizes, ungültige Muster)

- [ ] **Step 1: Fehlschlagende Tests anhängen**

```python
class TestSuggestPatterns:
    def test_path_segments(self):
        urls = ["https://a.com/de/x", "https://a.com/de/y", "https://a.com/fr/x", "https://a.com/"]
        sug = ld.suggest_patterns(urls)
        by_code = {s.code: s for s in sug}
        assert by_code["de"].count == 2 and by_code["fr"].count == 1
        assert re.match(by_code["de"].pattern, "https://a.com/de/x")
        assert not re.match(by_code["de"].pattern, "https://a.com/design/x")
        assert sug[0].code == "de"

    def test_subdomain_and_tld(self):
        urls = ["https://fr.a.com/x", "https://a.de/y", "https://a.de/z"]
        sug = ld.suggest_patterns(urls)
        codes = {s.code for s in sug}
        assert {"fr", "de"} <= codes
        de = next(s for s in sug if s.code == "de")
        assert re.match(de.pattern, "https://a.de/y") and not re.match(de.pattern, "https://a.dev/y")

    def test_empty(self):
        assert ld.suggest_patterns([]) == []


class TestSplitByPatterns:
    def test_top_down_and_rest(self):
        urls = ["https://a.com/de/x", "https://a.com/de/y", "https://a.com/fr/x", "https://a.com/"]
        groups, rest, invalid = ld.split_by_patterns(urls, [(r"^https?://[^/]+/de(/|$)", "de"), (r"^https?://[^/]+/fr(/|$)", "fr")])
        assert groups == {"de": [0, 1], "fr": [2]}
        assert rest == [3]
        assert invalid == []

    def test_first_match_wins(self):
        urls = ["https://a.com/de/x"]
        groups, rest, _ = ld.split_by_patterns(urls, [(r"^https?://a\.com/", "all"), (r"/de/", "de")])
        assert groups == {"all": [0]}

    def test_invalid_regex_reported(self):
        groups, rest, invalid = ld.split_by_patterns(["https://a.com/x"], [("(", "de")])
        assert invalid == ["("] and rest == [0] and groups == {}

    def test_empty_pattern_or_code_skipped(self):
        groups, rest, invalid = ld.split_by_patterns(["https://a.com/x"], [("", "de"), (".*", "")])
        assert groups == {} and rest == [0]
```

Am Dateianfang `import re` ergänzen.

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_lang_detect.py -v`
Expected: neue Tests FAIL mit AttributeError.

- [ ] **Step 3: Implementierung anhängen**

```python
def suggest_patterns(urls: list[str]) -> list[PatternSuggestion]:
    """Schlägt Regex-Muster aus Pfadsegment, Subdomain und ccTLD vor, nach Häufigkeit sortiert."""
    buckets: dict[str, dict] = {}
    for u in urls:
        host, path = _host_and_path(u)
        if not host:
            continue
        segs = [x for x in path.split("/") if x]
        parts = host.split(".")
        key = None
        code = ""
        if segs and CODE_RE.match(segs[0]):
            seg = segs[0].lower()
            key = rf"^https?://[^/]+/{re.escape(seg)}(/|$)"
            code = normalize_code(seg)
        elif len(parts) >= 3 and len(parts[0]) == 2 and parts[0] != "www":
            sub = parts[0]
            key = rf"^https?://{re.escape(sub)}\."
            code = sub
        else:
            tld = parts[-1]
            if tld in CCTLD_LANG:
                key = rf"^https?://[^/]+\.{re.escape(tld)}(/|$)"
                code = CCTLD_LANG[tld]
        if key is None:
            continue
        b = buckets.setdefault(key, {"code": code, "example": u, "count": 0})
        b["count"] += 1
    out = [PatternSuggestion(pattern=k, code=v["code"], example=v["example"], count=v["count"]) for k, v in buckets.items()]
    out.sort(key=lambda s: (-s.count, s.pattern))
    return out


def split_by_patterns(
    urls: list[str], patterns: list[tuple[str, str]]
) -> tuple[dict[str, list[int]], list[int], list[str]]:
    """Ordnet jede URL dem ersten passenden Muster zu (top-down). Leere Muster/Codes werden übersprungen."""
    compiled: list[tuple[re.Pattern, str]] = []
    invalid: list[str] = []
    for pat, code in patterns:
        pat = (pat or "").strip()
        code = (code or "").strip()
        if not pat or not code:
            continue
        try:
            compiled.append((re.compile(pat, re.I), code))
        except re.error:
            invalid.append(pat)
    groups: dict[str, list[int]] = {}
    rest: list[int] = []
    for i, u in enumerate(urls):
        s = str(u or "")
        for rx, code in compiled:
            if rx.search(s):
                groups.setdefault(code, []).append(i)
                break
        else:
            rest.append(i)
    return groups, rest, invalid
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest tests/test_lang_detect.py -v`
Expected: alle PASS.

- [ ] **Step 5: Commit**

```bash
git add hreflang_matcher/lang_detect.py tests/test_lang_detect.py
git commit -m "feat: pattern suggestion and split for single-file uploads

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Blockweise Top-k-Cosinus

**Files:**
- Create: `hreflang_matcher/matching.py`
- Test: `tests/test_matching.py`

**Interfaces:**
- Produces: `topk_similarity(P: np.ndarray, L: np.ndarray, k: int = 5, row_block: int = 1024, col_block: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - Rückgabe `(top_idx (n,k') int64, top_val (n,k') float32, col_best_idx (m,) int64)` mit `k' = min(k, m)`. Je Zeile absteigend sortiert. `col_best_idx[j]` ist die Pivot-Zeile mit dem höchsten Score für Spalte j, `-1` wenn n == 0.
  - P und L müssen L2-normalisiert sein (Cosinus = Skalarprodukt).

- [ ] **Step 1: Fehlschlagende Tests schreiben**

`tests/test_matching.py`:
```python
import numpy as np
import pandas as pd
import pytest

from hreflang_matcher import matching
from hreflang_matcher.models import LanguageSet


def _norm(M):
    M = np.asarray(M, dtype=np.float32)
    return M / np.linalg.norm(M, axis=1, keepdims=True)


class TestTopkSimilarity:
    def test_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        P = _norm(rng.normal(size=(50, 16)))
        L = _norm(rng.normal(size=(70, 16)))
        S = P @ L.T
        top_idx, top_val, col_best = matching.topk_similarity(P, L, k=5, row_block=7, col_block=11)
        exp_idx = np.argsort(-S, axis=1)[:, :5]
        np.testing.assert_array_equal(top_idx, exp_idx)
        np.testing.assert_allclose(top_val, np.take_along_axis(S, exp_idx, axis=1), rtol=1e-5)
        np.testing.assert_array_equal(col_best, S.argmax(axis=0))

    def test_k_larger_than_m_is_capped(self):
        P = _norm(np.eye(3, 4))
        L = _norm(np.eye(2, 4))
        top_idx, top_val, col_best = matching.topk_similarity(P, L, k=5)
        assert top_idx.shape == (3, 2)
        assert top_idx[0, 0] == 0 and top_idx[1, 0] == 1
        assert top_val[2].tolist() == [0.0, 0.0]
        assert col_best.tolist() == [0, 1]

    def test_empty_inputs(self):
        P = np.zeros((0, 4), dtype=np.float32)
        L = _norm(np.eye(2, 4))
        top_idx, top_val, col_best = matching.topk_similarity(P, L, k=3)
        assert top_idx.shape == (0, 2) and col_best.shape == (2,) and (col_best == -1).all()
        top_idx, top_val, col_best = matching.topk_similarity(L, P, k=3)
        assert top_idx.shape == (2, 0) and col_best.shape == (0,)
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_matching.py -v`
Expected: FAIL mit ModuleNotFoundError.

- [ ] **Step 3: Implementierung**

`hreflang_matcher/matching.py`:
```python
from __future__ import annotations

import numpy as np

from .lang_detect import strip_language
from .models import LanguageSet, Match, MatchResult


def topk_similarity(
    P: np.ndarray, L: np.ndarray, k: int = 5, row_block: int = 1024, col_block: int = 4096
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k Cosinus je Pivot-Zeile und bester Pivot je L-Spalte, blockweise ohne volle Matrix."""
    n, m = int(P.shape[0]), int(L.shape[0])
    k = max(0, min(k, m))
    top_idx = np.full((n, k), -1, dtype=np.int64)
    top_val = np.full((n, k), -np.inf, dtype=np.float32)
    col_best_idx = np.full(m, -1, dtype=np.int64)
    col_best_val = np.full(m, -np.inf, dtype=np.float32)
    if n == 0 or m == 0:
        return top_idx, top_val, col_best_idx

    P = np.ascontiguousarray(P, dtype=np.float32)
    L = np.ascontiguousarray(L, dtype=np.float32)

    for r0 in range(0, n, row_block):
        r1 = min(r0 + row_block, n)
        rows_val = top_val[r0:r1].copy()
        rows_idx = top_idx[r0:r1].copy()
        for c0 in range(0, m, col_block):
            c1 = min(c0 + col_block, m)
            S = P[r0:r1] @ L[c0:c1].T                      # (b, c)
            # bester Pivot je Spalte
            blk_arg = S.argmax(axis=0)
            blk_max = S[blk_arg, np.arange(c1 - c0)]
            better = blk_max > col_best_val[c0:c1]
            col_best_val[c0:c1] = np.where(better, blk_max, col_best_val[c0:c1])
            col_best_idx[c0:c1] = np.where(better, blk_arg + r0, col_best_idx[c0:c1])
            # Top-k je Zeile mit bisherigem Stand zusammenführen
            kk = min(k, c1 - c0)
            if kk < S.shape[1]:
                part = np.argpartition(S, -kk, axis=1)[:, -kk:]
            else:
                part = np.tile(np.arange(S.shape[1]), (S.shape[0], 1))
            cand_val = np.take_along_axis(S, part, axis=1).astype(np.float32)
            cand_idx = (part + c0).astype(np.int64)
            all_val = np.concatenate([rows_val, cand_val], axis=1)
            all_idx = np.concatenate([rows_idx, cand_idx], axis=1)
            order = np.argsort(-all_val, axis=1, kind="stable")[:, :k]
            rows_val = np.take_along_axis(all_val, order, axis=1)
            rows_idx = np.take_along_axis(all_idx, order, axis=1)
        top_val[r0:r1] = rows_val
        top_idx[r0:r1] = rows_idx
    return top_idx, top_val, col_best_idx
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest tests/test_matching.py -v`
Expected: alle PASS.

- [ ] **Step 5: Commit**

```bash
git add hreflang_matcher/matching.py tests/test_matching.py
git commit -m "feat: blockwise top-k cosine similarity

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Slug-Match

**Files:**
- Modify: `hreflang_matcher/matching.py`
- Test: `tests/test_matching.py`

**Interfaces:**
- Consumes: `lang_detect.strip_language`, `models.LanguageSet`, `models.Match`
- Produces: `slug_match(pivot: LanguageSet, other: LanguageSet) -> tuple[list[Match], list[int], list[int]]` → (Matches, verbleibende Pivot-Indizes, verbleibende Other-Indizes)

- [ ] **Step 1: Fehlschlagende Tests anhängen**

```python
def _ls(code, urls, vectors=None):
    from hreflang_matcher.io_utils import normalize_url
    if vectors is None:
        vectors = np.eye(len(urls), max(len(urls), 4), dtype=np.float32)
    return LanguageSet(code, code, list(urls), [normalize_url(u) for u in urls], _norm(vectors),
                       pd.DataFrame(columns=["URL", "Grund"]))


class TestSlugMatch:
    def test_matches_identical_slug_across_language_segment(self):
        de = _ls("de", ["https://a.com/de/x/", "https://a.com/de/y", "https://a.com/de/only-de"])
        fr = _ls("fr", ["https://a.com/fr/y", "https://a.com/fr/x", "https://a.com/fr/only-fr"])
        matches, rem_p, rem_o = matching.slug_match(de, fr)
        pairs = {(m.pivot_url, m.other_url) for m in matches}
        assert pairs == {("https://a.com/de/x/", "https://a.com/fr/x"), ("https://a.com/de/y", "https://a.com/fr/y")}
        assert all(m.method == "slug" and m.score == 1.0 and m.confidence == "sicher" and m.reciprocal for m in matches)
        assert rem_p == [2] and rem_o == [2]

    def test_subdomain_and_tld(self):
        de = _ls("de", ["https://a.de/p"])
        fr = _ls("fr", ["https://fr.a.com/p"])
        matches, rem_p, rem_o = matching.slug_match(de, fr)
        assert len(matches) == 1 and rem_p == [] and rem_o == []

    def test_no_matches(self):
        de = _ls("de", ["https://a.com/de/x"])
        fr = _ls("fr", ["https://a.com/fr/y"])
        matches, rem_p, rem_o = matching.slug_match(de, fr)
        assert matches == [] and rem_p == [0] and rem_o == [0]
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_matching.py::TestSlugMatch -v`
Expected: FAIL mit AttributeError.

- [ ] **Step 3: Implementierung anhängen**

```python
def slug_match(pivot: LanguageSet, other: LanguageSet) -> tuple[list[Match], list[int], list[int]]:
    """Exakter Pfad-Match nach Entfernen des Sprachsegments. Score 1.0, Methode 'slug'."""
    other_by_key: dict[str, int] = {}
    for j, u in enumerate(other.norm_urls):
        other_by_key.setdefault(strip_language(u, other.code), j)
    matches: list[Match] = []
    used_other: set[int] = set()
    rem_p: list[int] = []
    for i, u in enumerate(pivot.norm_urls):
        j = other_by_key.get(strip_language(u, pivot.code))
        if j is not None and j not in used_other:
            used_other.add(j)
            matches.append(Match(
                pivot_url=pivot.urls[i], other_url=other.urls[j], other_code=other.code,
                score=1.0, method="slug", reciprocal=True,
                second_url=None, second_score=None, margin=None,
                confidence="sicher", reason="",
            ))
        else:
            rem_p.append(i)
    rem_o = [j for j in range(len(other.urls)) if j not in used_other]
    return matches, rem_p, rem_o
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest tests/test_matching.py -v`
Expected: alle PASS.

- [ ] **Step 5: Commit**

```bash
git add hreflang_matcher/matching.py tests/test_matching.py
git commit -m "feat: slug match stage

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Embedding-Match mit greedy 1:1, Reziprozität, Margin, Konfidenz

**Files:**
- Modify: `hreflang_matcher/matching.py`
- Test: `tests/test_matching.py`

**Interfaces:**
- Consumes: `topk_similarity`, `models.Match`, `models.LanguageSet`
- Produces: `embedding_match(pivot, other, rem_p: list[int], rem_o: list[int], threshold: float, band: float, min_margin: float, k: int = 5) -> list[Match]`

Konfidenzregeln (Spec 5.6): `sicher` nur wenn `score >= threshold + band` **und** reziprok **und** (`margin >= min_margin` oder kein Zweitbester). Sonst `prüfen` mit allen Gründen in `reason`, getrennt durch `"; "`. Gründe wörtlich: `"knapp über Threshold (0.812)"`, `"nicht reziprok"`, `"mehrdeutig (Margin 0.010)"`.

- [ ] **Step 1: Fehlschlagende Tests anhängen**

```python
class TestEmbeddingMatch:
    def _sets(self):
        # de0 ~ fr1 (0.99), de1 ~ fr0 (0.95), de2 ~ nichts, fr2 ~ nichts
        base = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        de = _ls("de", ["https://a.com/de/0", "https://a.com/de/1", "https://a.com/de/2"],
                 base[:3])
        fr_vecs = np.array([[0.05, 1, 0, 0], [1, 0.1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        fr = _ls("fr", ["https://a.com/fr/0", "https://a.com/fr/1", "https://a.com/fr/2"], fr_vecs)
        return de, fr

    def test_basic_assignment_and_unmatched(self):
        de, fr = self._sets()
        ms = matching.embedding_match(de, fr, [0, 1, 2], [0, 1, 2], threshold=0.8, band=0.0, min_margin=0.0)
        pairs = {(m.pivot_url, m.other_url): m for m in ms}
        assert set(pairs) == {("https://a.com/de/0", "https://a.com/fr/1"), ("https://a.com/de/1", "https://a.com/fr/0")}
        m = pairs[("https://a.com/de/0", "https://a.com/fr/1")]
        assert m.method == "embedding" and m.reciprocal and m.confidence == "sicher" and m.reason == ""
        assert m.second_url == "https://a.com/fr/0" and m.second_score is not None
        assert m.margin == pytest.approx(m.score - m.second_score, abs=1e-6)

    def test_threshold_excludes(self):
        de, fr = self._sets()
        ms = matching.embedding_match(de, fr, [0, 1, 2], [0, 1, 2], threshold=0.999, band=0.0, min_margin=0.0)
        assert ms == []

    def test_band_marks_review(self):
        de, fr = self._sets()
        ms = matching.embedding_match(de, fr, [0, 1, 2], [0, 1, 2], threshold=0.9, band=0.2, min_margin=0.0)
        assert all(m.confidence == "prüfen" and "knapp über Threshold" in m.reason for m in ms)

    def test_margin_marks_ambiguous_and_names_second(self):
        # fr0 und fr1 fast gleich nah an de0
        de = _ls("de", ["https://a.com/de/0"], np.array([[1, 0, 0, 0]], dtype=np.float32))
        fr = _ls("fr", ["https://a.com/fr/0", "https://a.com/fr/1"],
                 np.array([[1, 0.10, 0, 0], [1, 0.11, 0, 0]], dtype=np.float32))
        ms = matching.embedding_match(de, fr, [0], [0, 1], threshold=0.5, band=0.0, min_margin=0.05)
        assert len(ms) == 1
        m = ms[0]
        assert m.other_url == "https://a.com/fr/0" and m.second_url == "https://a.com/fr/1"
        assert m.confidence == "prüfen" and "mehrdeutig" in m.reason
        assert m.margin < 0.05

    def test_greedy_one_to_one_and_non_reciprocal(self):
        # de0 und de1 wollen beide fr0; de0 ist näher. de1 bekommt fr1 (nicht reziprok, weil fr1 de0 lieber mag).
        de = _ls("de", ["https://a.com/de/0", "https://a.com/de/1"],
                 np.array([[1, 0, 0, 0], [0.9, 0.3, 0, 0]], dtype=np.float32))
        fr = _ls("fr", ["https://a.com/fr/0", "https://a.com/fr/1"],
                 np.array([[1, 0, 0, 0], [0.6, 0.8, 0, 0]], dtype=np.float32))
        # de0·fr0 = 1.0, de1·fr0 = 0.949, de1·fr1 = 0.822, de0·fr1 = 0.6
        ms = matching.embedding_match(de, fr, [0, 1], [0, 1], threshold=0.5, band=0.0, min_margin=0.0)
        by_pivot = {m.pivot_url: m for m in ms}
        assert by_pivot["https://a.com/de/0"].other_url == "https://a.com/fr/0"
        assert by_pivot["https://a.com/de/1"].other_url == "https://a.com/fr/1"
        assert by_pivot["https://a.com/de/1"].reciprocal is False
        assert "nicht reziprok" in by_pivot["https://a.com/de/1"].reason
        urls_o = [m.other_url for m in ms]
        assert len(urls_o) == len(set(urls_o))

    def test_respects_remaining_indices(self):
        de, fr = self._sets()
        ms = matching.embedding_match(de, fr, [1], [0, 1, 2], threshold=0.8, band=0.0, min_margin=0.0)
        assert len(ms) == 1 and ms[0].pivot_url == "https://a.com/de/1"

    def test_empty_remaining(self):
        de, fr = self._sets()
        assert matching.embedding_match(de, fr, [], [0], threshold=0.5, band=0.0, min_margin=0.0) == []
        assert matching.embedding_match(de, fr, [0], [], threshold=0.5, band=0.0, min_margin=0.0) == []
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_matching.py::TestEmbeddingMatch -v`
Expected: FAIL mit AttributeError.

- [ ] **Step 3: Implementierung anhängen**

```python
def embedding_match(
    pivot: LanguageSet,
    other: LanguageSet,
    rem_p: list[int],
    rem_o: list[int],
    threshold: float,
    band: float,
    min_margin: float,
    k: int = 5,
) -> list[Match]:
    """Cosinus-Matching mit greedy 1:1-Zuordnung, Reziprozitäts- und Margin-Prüfung."""
    if not rem_p or not rem_o:
        return []
    P = pivot.vectors[rem_p]
    L = other.vectors[rem_o]
    top_idx, top_val, col_best = topk_similarity(P, L, k=k)

    cands: list[tuple[float, int, int]] = []
    for a in range(top_idx.shape[0]):
        for r in range(top_idx.shape[1]):
            v = float(top_val[a, r])
            if top_idx[a, r] >= 0 and v >= threshold:
                cands.append((v, a, int(top_idx[a, r])))
    cands.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[Match] = []
    for score, a, b in cands:
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)

        reciprocal = int(top_idx[a, 0]) == b and int(col_best[b]) == a

        second_url = second_score = margin = None
        for r in range(top_idx.shape[1]):
            j = int(top_idx[a, r])
            if j >= 0 and j != b and np.isfinite(top_val[a, r]):
                second_score = float(top_val[a, r])
                second_url = other.urls[rem_o[j]]
                margin = score - second_score
                break

        reasons: list[str] = []
        if score < threshold + band:
            reasons.append(f"knapp über Threshold ({score:.3f})")
        if not reciprocal:
            reasons.append("nicht reziprok")
        if margin is not None and margin < min_margin:
            reasons.append(f"mehrdeutig (Margin {margin:.3f})")

        matches.append(Match(
            pivot_url=pivot.urls[rem_p[a]], other_url=other.urls[rem_o[b]], other_code=other.code,
            score=round(score, 4), method="embedding", reciprocal=reciprocal,
            second_url=second_url,
            second_score=None if second_score is None else round(second_score, 4),
            margin=None if margin is None else round(margin, 4),
            confidence="sicher" if not reasons else "prüfen",
            reason="; ".join(reasons),
        ))
    return matches
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest tests/test_matching.py -v`
Expected: alle PASS.

- [ ] **Step 5: Commit**

```bash
git add hreflang_matcher/matching.py tests/test_matching.py
git commit -m "feat: embedding match with greedy 1:1, reciprocity and margin confidence

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: run_matching über alle Sprachen

**Files:**
- Modify: `hreflang_matcher/matching.py`
- Test: `tests/test_matching.py`

**Interfaces:**
- Consumes: `slug_match`, `embedding_match`, `models.MatchResult`
- Produces: `run_matching(sets: list[LanguageSet], pivot_code: str, threshold: float, band: float, min_margin: float, use_slug: bool, k: int = 5) -> MatchResult`
  - `unmatched[other_code]`: Other-URLs ohne Zuordnung. `unmatched[pivot_code]`: Pivot-URLs ohne Treffer in **irgendeiner** Sprache.
  - `stats[other_code]` Schlüssel: `"slug"`, `"embedding"`, `"sicher"`, `"prüfen"`, `"unmatched"`. `stats[pivot_code]` Schlüssel: `"total"`, `"unmatched"`.

- [ ] **Step 1: Fehlschlagende Tests anhängen**

```python
class TestRunMatching:
    def _sets(self):
        de = _ls("de", ["https://a.com/de/x", "https://a.com/de/y", "https://a.com/de/z"],
                 np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32))
        fr = _ls("fr", ["https://a.com/fr/x", "https://a.com/fr/other"],
                 np.array([[0, 0, 0, 1], [0, 1, 0.1, 0]], dtype=np.float32))   # fr/x nur per Slug findbar
        en = _ls("en", ["https://a.com/en/z"], np.array([[0, 0, 1, 0]], dtype=np.float32))
        return [de, fr, en]

    def test_with_slug(self):
        res = matching.run_matching(self._sets(), "de", threshold=0.8, band=0.0, min_margin=0.0, use_slug=True)
        assert res.pivot_code == "de"
        pairs = {(m.pivot_url, m.other_code): (m.other_url, m.method) for m in res.matches}
        assert pairs[("https://a.com/de/x", "fr")] == ("https://a.com/fr/x", "slug")
        assert pairs[("https://a.com/de/y", "fr")] == ("https://a.com/fr/other", "embedding")
        assert pairs[("https://a.com/de/z", "en")] == ("https://a.com/en/z", "slug")
        assert res.unmatched["fr"] == [] and res.unmatched["en"] == []
        assert res.unmatched["de"] == []
        assert res.stats["fr"] == {"slug": 1, "embedding": 1, "sicher": 2, "prüfen": 0, "unmatched": 0}
        assert res.stats["de"] == {"total": 3, "unmatched": 0}

    def test_without_slug(self):
        res = matching.run_matching(self._sets(), "de", threshold=0.8, band=0.0, min_margin=0.0, use_slug=False)
        assert all(m.method == "embedding" for m in res.matches)
        assert res.unmatched["fr"] == ["https://a.com/fr/x"]
        assert res.unmatched["de"] == ["https://a.com/de/x"]

    def test_unknown_pivot_raises(self):
        with pytest.raises(ValueError):
            matching.run_matching(self._sets(), "it", threshold=0.8, band=0.0, min_margin=0.0, use_slug=False)
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_matching.py::TestRunMatching -v`
Expected: FAIL mit AttributeError.

- [ ] **Step 3: Implementierung anhängen**

```python
def run_matching(
    sets: list[LanguageSet],
    pivot_code: str,
    threshold: float,
    band: float,
    min_margin: float,
    use_slug: bool,
    k: int = 5,
) -> MatchResult:
    """Matcht jede Nicht-Pivot-Sprache gegen die Pivot-Sprache."""
    pivot = next((s for s in sets if s.code == pivot_code), None)
    if pivot is None:
        raise ValueError(f"Pivot-Sprache '{pivot_code}' nicht in den Sprachsets.")

    matches: list[Match] = []
    unmatched: dict[str, list[str]] = {}
    stats: dict[str, dict[str, int]] = {}
    pivot_matched: set[str] = set()

    for other in sets:
        if other.code == pivot_code:
            continue
        if use_slug:
            slug, rem_p, rem_o = slug_match(pivot, other)
        else:
            slug, rem_p, rem_o = [], list(range(len(pivot))), list(range(len(other)))
        emb = embedding_match(pivot, other, rem_p, rem_o, threshold, band, min_margin, k)
        all_m = slug + emb
        matches.extend(all_m)
        matched_o = {m.other_url for m in all_m}
        pivot_matched |= {m.pivot_url for m in all_m}
        unmatched[other.code] = [u for u in other.urls if u not in matched_o]
        stats[other.code] = {
            "slug": len(slug),
            "embedding": len(emb),
            "sicher": sum(1 for m in all_m if m.confidence == "sicher"),
            "prüfen": sum(1 for m in all_m if m.confidence == "prüfen"),
            "unmatched": len(unmatched[other.code]),
        }

    unmatched[pivot_code] = [u for u in pivot.urls if u not in pivot_matched]
    stats[pivot_code] = {"total": len(pivot), "unmatched": len(unmatched[pivot_code])}
    return MatchResult(pivot_code=pivot_code, matches=matches, unmatched=unmatched, stats=stats)
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest -v`
Expected: alle PASS.

- [ ] **Step 5: Commit**

```bash
git add hreflang_matcher/matching.py tests/test_matching.py
git commit -m "feat: run_matching across all language sets

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Cluster, Mapping-Tabelle, Unmatched-Tabelle

**Files:**
- Create: `hreflang_matcher/output.py`
- Test: `tests/test_output.py`

**Interfaces:**
- Consumes: `models.MatchResult`, `models.Match`, `models.Cluster`, `models.LanguageSet`
- Produces:
  - `build_clusters(result: MatchResult, sets: list[LanguageSet]) -> list[Cluster]` (eine je Pivot-URL in Pivot-Reihenfolge, auch ohne Treffer)
  - `x_default_fallback(cluster: Cluster, x_default_code: str | None) -> bool`
  - `mapping_table(clusters, codes: list[str], pivot_code: str, x_default_code: str | None) -> pd.DataFrame`
  - `unmatched_table(result: MatchResult, sets: list[LanguageSet]) -> pd.DataFrame` (Spalten `Sprache`, `URL`, `Grund`)

Spaltenreihenfolge der Mapping-Tabelle: `Pivot-URL (<pivot>)`, dann je Nicht-Pivot-Code in der Reihenfolge von `codes`: `URL (<c>)`, `Score (<c>)`, `Methode (<c>)`, `Konfidenz (<c>)`, `Grund (<c>)`, `Zweitbeste URL (<c>)`, `Zweitbester Score (<c>)`; zuletzt `Cluster-Konfidenz`, `x-default-Fallback`.

- [ ] **Step 1: Fehlschlagende Tests schreiben**

`tests/test_output.py`:
```python
import numpy as np
import pandas as pd
import pytest

from hreflang_matcher import output
from hreflang_matcher.models import Cluster, LanguageSet, Match, MatchResult


def _ls(code, urls, dropped=None):
    dropped = dropped or []
    return LanguageSet(code, code, list(urls), list(urls), np.eye(len(urls), 4, dtype=np.float32),
                       pd.DataFrame(dropped, columns=["URL", "Grund"]))


def _m(p, o, code, score=0.95, conf="sicher", reason="", method="embedding", second=None, second_score=None, margin=None, reciprocal=True):
    return Match(p, o, code, score, method, reciprocal, second, second_score, margin, conf, reason)


@pytest.fixture
def scenario():
    de = _ls("de", ["https://a.com/de/1", "https://a.com/de/2", "https://a.com/de/3"], dropped=[("https://a.com/de/bad", "Embedding nicht lesbar")])
    fr = _ls("fr", ["https://a.com/fr/1", "https://a.com/fr/2", "https://a.com/fr/x"])
    en = _ls("en", ["https://a.com/en/1"])
    matches = [
        _m("https://a.com/de/1", "https://a.com/fr/1", "fr", method="slug", score=1.0),
        _m("https://a.com/de/1", "https://a.com/en/1", "en"),
        _m("https://a.com/de/2", "https://a.com/fr/2", "fr", score=0.82, conf="prüfen", reason="nicht reziprok",
           second="https://a.com/fr/x", second_score=0.81, margin=0.01, reciprocal=False),
    ]
    result = MatchResult("de", matches, {"fr": ["https://a.com/fr/x"], "en": [], "de": ["https://a.com/de/3"]}, {})
    return [de, fr, en], result


class TestBuildClusters:
    def test_clusters(self, scenario):
        sets, result = scenario
        cl = output.build_clusters(result, sets)
        assert [c.pivot_url for c in cl] == ["https://a.com/de/1", "https://a.com/de/2", "https://a.com/de/3"]
        assert cl[0].members == {"de": "https://a.com/de/1", "fr": "https://a.com/fr/1", "en": "https://a.com/en/1"}
        assert cl[0].confidence == "sicher"
        assert cl[1].members == {"de": "https://a.com/de/2", "fr": "https://a.com/fr/2"}
        assert cl[1].confidence == "prüfen"
        assert cl[2].members == {"de": "https://a.com/de/3"} and cl[2].matches == {}


class TestXDefaultFallback:
    def test_rules(self):
        c = Cluster("p", {"de": "p", "fr": "f"}, {}, "sicher")
        assert output.x_default_fallback(c, "de") is False
        assert output.x_default_fallback(c, "en") is True
        assert output.x_default_fallback(c, None) is False
        single = Cluster("p", {"de": "p"}, {}, "sicher")
        assert output.x_default_fallback(single, "en") is False


class TestMappingTable:
    def test_columns_and_values(self, scenario):
        sets, result = scenario
        cl = output.build_clusters(result, sets)
        df = output.mapping_table(cl, ["de", "fr", "en"], "de", "en")
        assert list(df.columns) == [
            "Pivot-URL (de)",
            "URL (fr)", "Score (fr)", "Methode (fr)", "Konfidenz (fr)", "Grund (fr)", "Zweitbeste URL (fr)", "Zweitbester Score (fr)",
            "URL (en)", "Score (en)", "Methode (en)", "Konfidenz (en)", "Grund (en)", "Zweitbeste URL (en)", "Zweitbester Score (en)",
            "Cluster-Konfidenz", "x-default-Fallback",
        ]
        assert len(df) == 3
        r0 = df.iloc[0]
        assert r0["URL (fr)"] == "https://a.com/fr/1" and r0["Methode (fr)"] == "slug" and r0["x-default-Fallback"] == "nein"
        r1 = df.iloc[1]
        assert r1["Konfidenz (fr)"] == "prüfen" and r1["Grund (fr)"] == "nicht reziprok"
        assert r1["Zweitbeste URL (fr)"] == "https://a.com/fr/x" and r1["Zweitbester Score (fr)"] == 0.81
        assert r1["URL (en)"] == "" and r1["x-default-Fallback"] == "ja"
        r2 = df.iloc[2]
        assert r2["URL (fr)"] == "" and r2["Cluster-Konfidenz"] == "sicher" and r2["x-default-Fallback"] == "nein"


class TestUnmatchedTable:
    def test_rows(self, scenario):
        sets, result = scenario
        df = output.unmatched_table(result, sets)
        assert list(df.columns) == ["Sprache", "URL", "Grund"]
        rows = set(map(tuple, df.values.tolist()))
        assert ("de", "https://a.com/de/bad", "Embedding nicht lesbar") in rows
        assert ("de", "https://a.com/de/3", "kein Treffer über Threshold") in rows
        assert ("fr", "https://a.com/fr/x", "kein Treffer über Threshold") in rows
        assert len(df) == 3
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_output.py -v`
Expected: FAIL mit ModuleNotFoundError.

- [ ] **Step 3: Implementierung**

`hreflang_matcher/output.py`:
```python
from __future__ import annotations

import html

import pandas as pd

from .lang_detect import language_of, normalize_code
from .models import Cluster, LanguageSet, Match, MatchResult

NO_MATCH_REASON = "kein Treffer über Threshold"


def build_clusters(result: MatchResult, sets: list[LanguageSet]) -> list[Cluster]:
    pivot = next(s for s in sets if s.code == result.pivot_code)
    by_pivot: dict[str, dict[str, Match]] = {}
    for m in result.matches:
        by_pivot.setdefault(m.pivot_url, {})[m.other_code] = m
    clusters: list[Cluster] = []
    for u in pivot.urls:
        ms = by_pivot.get(u, {})
        members = {result.pivot_code: u}
        for code, m in ms.items():
            members[code] = m.other_url
        conf = "prüfen" if any(m.confidence == "prüfen" for m in ms.values()) else "sicher"
        clusters.append(Cluster(pivot_url=u, members=members, matches=ms, confidence=conf))
    return clusters


def x_default_fallback(cluster: Cluster, x_default_code: str | None) -> bool:
    """True, wenn x-default gewünscht ist, der Cluster mehr als ein Mitglied hat und die x-default-Sprache fehlt."""
    if not x_default_code or len(cluster.members) < 2:
        return False
    return x_default_code not in cluster.members


def mapping_table(
    clusters: list[Cluster], codes: list[str], pivot_code: str, x_default_code: str | None
) -> pd.DataFrame:
    others = [c for c in codes if c != pivot_code]
    cols = [f"Pivot-URL ({pivot_code})"]
    for c in others:
        cols += [f"URL ({c})", f"Score ({c})", f"Methode ({c})", f"Konfidenz ({c})", f"Grund ({c})",
                 f"Zweitbeste URL ({c})", f"Zweitbester Score ({c})"]
    cols += ["Cluster-Konfidenz", "x-default-Fallback"]

    rows = []
    for cl in clusters:
        row = {cols[0]: cl.pivot_url}
        for c in others:
            m = cl.matches.get(c)
            row[f"URL ({c})"] = m.other_url if m else ""
            row[f"Score ({c})"] = m.score if m else None
            row[f"Methode ({c})"] = m.method if m else ""
            row[f"Konfidenz ({c})"] = m.confidence if m else ""
            row[f"Grund ({c})"] = m.reason if m else ""
            row[f"Zweitbeste URL ({c})"] = (m.second_url or "") if m else ""
            row[f"Zweitbester Score ({c})"] = m.second_score if m else None
        row["Cluster-Konfidenz"] = cl.confidence
        row["x-default-Fallback"] = "ja" if x_default_fallback(cl, x_default_code) else "nein"
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def unmatched_table(result: MatchResult, sets: list[LanguageSet]) -> pd.DataFrame:
    rows = []
    for s in sets:
        for _, r in s.dropped.iterrows():
            rows.append((s.code, r["URL"], r["Grund"]))
        for u in result.unmatched.get(s.code, []):
            rows.append((s.code, u, NO_MATCH_REASON))
    return pd.DataFrame(rows, columns=["Sprache", "URL", "Grund"])
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest tests/test_output.py -v`
Expected: alle PASS.

- [ ] **Step 5: Commit**

```bash
git add hreflang_matcher/output.py tests/test_output.py
git commit -m "feat: clusters, mapping table and unmatched table

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: hreflang-Code-Format und HTML-Blöcke

**Files:**
- Modify: `hreflang_matcher/output.py`
- Test: `tests/test_output.py`

**Interfaces:**
- Consumes: `build_clusters`, `x_default_fallback`, `lang_detect.language_of`, `lang_detect.normalize_code`
- Produces:
  - `format_hreflang_code(code: str, mode: str) -> str` mit `mode in {"language", "language-region"}`
  - `html_blocks(clusters: list[Cluster], pivot_code: str, x_default_code: str | None, code_mode: str, include_review: bool) -> str`

Regeln (Spec 5.7): Reihenfolge Pivot zuerst, übrige Codes alphabetisch, x-default zuletzt. Jedes Mitglied bekommt einen eigenen Block mit Kommentar `<!-- <url> -->` und identischer Tag-Liste. Cluster mit einem Mitglied: kein Block. Cluster „prüfen“ nur mit `include_review`. x-default fällt auf Pivot-URL zurück. URLs HTML-escaped. Blöcke durch Leerzeile getrennt, Datei endet mit `\n`.

- [ ] **Step 1: Fehlschlagende Tests anhängen**

```python
class TestFormatCode:
    def test_modes(self):
        assert output.format_hreflang_code("de-at", "language-region") == "de-AT"
        assert output.format_hreflang_code("de-at", "language") == "de"
        assert output.format_hreflang_code("fr", "language-region") == "fr"


class TestHtmlBlocks:
    def _clusters(self):
        c1 = Cluster("https://a.com/de/1?x=1&y=2", {"de": "https://a.com/de/1?x=1&y=2", "fr": "https://a.com/fr/1", "en": "https://a.com/en/1"}, {}, "sicher")
        c2 = Cluster("https://a.com/de/2", {"de": "https://a.com/de/2", "fr": "https://a.com/fr/2"}, {}, "prüfen")
        c3 = Cluster("https://a.com/de/3", {"de": "https://a.com/de/3"}, {}, "sicher")
        return [c1, c2, c3]

    def test_block_structure_order_and_escaping(self):
        out = output.html_blocks(self._clusters(), "de", "en", "language-region", include_review=False)
        blocks = out.strip().split("\n\n")
        assert len(blocks) == 3   # nur Cluster 1, ein Block je Mitglied
        b = blocks[0].split("\n")
        assert b[0] == "<!-- https://a.com/de/1?x=1&amp;y=2 -->"
        assert b[1] == '<link rel="alternate" hreflang="de" href="https://a.com/de/1?x=1&amp;y=2" />'
        assert b[2] == '<link rel="alternate" hreflang="en" href="https://a.com/en/1" />'
        assert b[3] == '<link rel="alternate" hreflang="fr" href="https://a.com/fr/1" />'
        assert b[4] == '<link rel="alternate" hreflang="x-default" href="https://a.com/en/1" />'
        assert blocks[1].startswith("<!-- https://a.com/en/1 -->") or blocks[1].startswith("<!-- https://a.com/fr/1 -->")
        assert blocks[1].split("\n")[1:] == b[1:]   # identische Tag-Liste
        assert out.endswith("\n")

    def test_include_review_and_xdefault_fallback(self):
        out = output.html_blocks(self._clusters(), "de", "en", "language-region", include_review=True)
        blocks = out.strip().split("\n\n")
        assert len(blocks) == 5
        c2_block = next(b for b in blocks if b.startswith("<!-- https://a.com/de/2 -->"))
        assert 'hreflang="x-default" href="https://a.com/de/2"' in c2_block

    def test_no_xdefault(self):
        out = output.html_blocks(self._clusters(), "de", None, "language-region", include_review=False)
        assert "x-default" not in out

    def test_language_mode(self):
        c = Cluster("p", {"de-AT": "https://a.at/p", "fr-CH": "https://a.ch/fr/p"}, {}, "sicher")
        out = output.html_blocks([c], "de-AT", None, "language", include_review=False)
        assert 'hreflang="de"' in out and 'hreflang="fr"' in out and "de-AT" not in out

    def test_empty(self):
        assert output.html_blocks([], "de", None, "language", include_review=True) == ""
```

- [ ] **Step 2: Tests laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_output.py -v`
Expected: neue Tests FAIL mit AttributeError.

- [ ] **Step 3: Implementierung anhängen**

```python
def format_hreflang_code(code: str, mode: str) -> str:
    return language_of(code) if mode == "language" else normalize_code(code)


def _tag(code_value: str, url: str) -> str:
    return f'<link rel="alternate" hreflang="{code_value}" href="{html.escape(url, quote=True)}" />'


def html_blocks(
    clusters: list[Cluster],
    pivot_code: str,
    x_default_code: str | None,
    code_mode: str,
    include_review: bool,
) -> str:
    blocks: list[str] = []
    for cl in clusters:
        if len(cl.members) < 2:
            continue
        if cl.confidence == "prüfen" and not include_review:
            continue
        ordered = [pivot_code] + sorted(c for c in cl.members if c != pivot_code)
        tags = [_tag(format_hreflang_code(c, code_mode), cl.members[c]) for c in ordered]
        if x_default_code:
            xd_url = cl.members.get(x_default_code, cl.pivot_url)
            tags.append(_tag("x-default", xd_url))
        for c in ordered:
            blocks.append("\n".join([f"<!-- {html.escape(cl.members[c], quote=True)} -->"] + tags))
    return "\n\n".join(blocks) + ("\n" if blocks else "")
```

- [ ] **Step 4: Tests laufen lassen**

Run: `.venv/bin/python -m pytest -v`
Expected: alle PASS.

- [ ] **Step 5: Commit**

```bash
git add hreflang_matcher/output.py tests/test_output.py
git commit -m "feat: hreflang code formatting and HTML block export

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Streamlit-App

**Files:**
- Create: `app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes (exakt wie in Tasks 1–10 definiert):
  - `io_utils.read_any_file(name, raw)`, `detect_url_column(df)`, `detect_embedding_column(df)`, `detect_status_column(df)`, `detect_indexability_column(df)`, `build_language_set(df, code, label, url_col, emb_col, filter_indexable, status_col, index_col)`, `check_dimensions(sets)`
  - `lang_detect.suggest_code(urls)`, `normalize_code(code)`, `is_valid_code(code)`, `find_language_collisions(codes)`, `suggest_patterns(urls)`, `split_by_patterns(urls, patterns)`
  - `matching.run_matching(sets, pivot_code, threshold, band, min_margin, use_slug)`
  - `output.build_clusters(result, sets)`, `mapping_table(clusters, codes, pivot_code, x_default_code)`, `unmatched_table(result, sets)`, `html_blocks(clusters, pivot_code, x_default_code, code_mode, include_review)`
- Produces: lauffähige App `streamlit run app.py`

- [ ] **Step 1: Smoke-Test schreiben**

`tests/test_app.py`:
```python
from streamlit.testing.v1 import AppTest


def test_app_renders_without_exception():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("ONE hreflang Matcher" in t.value for t in at.title)
    # Ohne Upload endet die App mit dem Hinweis zum Hochladen
    assert any("crawl-dateien" in i.value.lower() for i in at.info)
```

- [ ] **Step 2: Test laufen lassen, Fehlschlag prüfen**

Run: `.venv/bin/python -m pytest tests/test_app.py -v`
Expected: FAIL, weil `app.py` fehlt.

- [ ] **Step 3: app.py schreiben**

`app.py`:
```python
from __future__ import annotations

import pandas as pd
import streamlit as st

from hreflang_matcher import io_utils, lang_detect, matching, output
from hreflang_matcher.models import LanguageSet

# ============================================================
# Seite & Branding
# ============================================================
st.set_page_config(page_title="ONE hreflang Matcher", layout="wide")

LOGO_URL = "https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png"

st.markdown(
    """
<style>
div[data-testid="stDownloadButton"] > button {
    background-color: #d7263d !important;
    color: white !important;
    border: 1px solid #b51f33 !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background-color: #b51f33 !important;
    color: white !important;
    border: 1px solid #8f1828 !important;
}
div[data-testid="stDownloadButton"] > button:focus {
    box-shadow: 0 0 0 0.2rem rgba(215, 38, 61, 0.35) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

try:
    st.image(LOGO_URL, width=250)
except Exception:
    pass

st.title("ONE hreflang Matcher")

st.markdown(
    """
<div style="background-color: #f2f2f2; color: #000000; padding: 15px 20px; border-radius: 6px; font-size: 0.9em; max-width: 850px; margin-bottom: 1.5em; line-height: 1.5;">
  Entwickelt von <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">Daniel Kremer</a> von <a href="https://onebeyondsearch.com/" target="_blank">ONE Beyond Search</a> &nbsp;|&nbsp;
  Folge mir auf <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">LinkedIn</a> für mehr SEO-Insights und Tool-Updates
</div>
<hr>
""",
    unsafe_allow_html=True,
)

HELP_MD = """
**Ziel:** Das Tool ordnet die Sprach- und Regionsvarianten einer Website einander zu und erzeugt daraus
fertige hreflang-Cluster. Grundlage sind die Embeddings aus deinem Crawl (z. B. Screaming Frog mit
OpenAI-Embeddings). Verglichen wird immer **zwischen** Sprachen: Für jede URL der Pivot-Sprache wird in
jeder anderen Sprache die URL mit der höchsten Cosinus-Ähnlichkeit gesucht.

**Eingabe:** CSV oder Excel mit mindestens einer URL-Spalte und einer Embedding-Spalte. Alle weiteren
Spalten werden ignoriert. Entweder eine Datei je Sprachvariante oder eine Gesamtdatei, die per URL-Muster
aufgeteilt wird.

**Wichtig für gute Ergebnisse:**
- Die Embeddings müssen von einem **multilingualen Modell** stammen (z. B. OpenAI text-embedding-3).
  Rein englische Modelle liefern über Sprachgrenzen hinweg keine brauchbaren Ähnlichkeiten.
- Alle Dateien müssen mit **demselben Modell** erzeugt sein (gleiche Dimension).
- **Excel schneidet Zellen bei 32.767 Zeichen ab.** Bei 1536 Dimensionen ist das knapp, bei 3072 werden
  Vektoren zerstört. Nutze CSV. Abgeschnittene Vektoren werden erkannt und verworfen, nicht repariert.

**So funktioniert die Zuordnung:** Für jede Pivot-URL werden die fünf ähnlichsten URLs je Sprache über dem
Threshold gesammelt. Alle Kandidatenpaare werden nach Score sortiert und von oben abgearbeitet. Ein Paar wird
übernommen, wenn beide URLs noch frei sind. So bekommt jede URL genau einen Partner je Sprache (1:1).

**Konfidenz:** Ein Treffer ist *sicher*, wenn er deutlich über dem Threshold liegt (außerhalb des
Review-Bands), *reziprok* ist (beide URLs sind gegenseitig der beste Treffer) und der zweitbeste Kandidat
mindestens um die Margin schlechter ist. Sonst *prüfen*, mit Angabe des Grunds und des zweitbesten Kandidaten.

**Optional:** Exact-Slug-Match als Vorstufe (identischer Pfad nach Entfernen des Sprachsegments) und ein
Filter auf Status 200 und Indexable / Indexierbar.
"""

with st.expander("ℹ️ Was macht das Tool? (Erklärung & Tipps)", expanded=False):
    st.markdown(HELP_MD)

MODE_MULTI = "Eine Datei je Sprachvariante"
MODE_SINGLE = "Eine Gesamtdatei mit allen Sprachvarianten"
CODE_MODE_REGION = "Sprache-Region (z. B. de-AT)"
CODE_MODE_LANG = "Nur Sprache (z. B. de)"
NO_XDEFAULT = "kein x-default"
FILE_TYPES = ["csv", "xlsx", "xlsm", "xls"]


# ============================================================
# Gecachte Helfer
# ============================================================
@st.cache_data(show_spinner=False)
def read_cached(name: str, raw: bytes) -> pd.DataFrame:
    return io_utils.read_any_file(name, raw)


@st.cache_data(show_spinner=False)
def build_cached(
    name: str,
    raw: bytes,
    code: str,
    url_col: str,
    emb_col: str,
    filter_idx: bool,
    status_col: str | None,
    index_col: str | None,
    row_idx: tuple[int, ...] | None,
) -> LanguageSet:
    df = io_utils.read_any_file(name, raw)
    if row_idx is not None:
        df = df.iloc[list(row_idx)]
    return io_utils.build_language_set(df, code, name, url_col, emb_col, filter_idx, status_col, index_col)


def _clean(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


# ============================================================
# 1. Upload
# ============================================================
st.subheader("1. Crawl-Dateien hochladen")
mode = st.radio("Wie liegen deine Crawl-Daten vor?", [MODE_MULTI, MODE_SINGLE], horizontal=True)

sources: list[dict] = []

if mode == MODE_MULTI:
    files = st.file_uploader(
        "Eine Datei je Sprachvariante (CSV oder Excel), Mehrfachauswahl möglich",
        type=FILE_TYPES,
        accept_multiple_files=True,
        help="Mindestens eine URL-Spalte und eine Embedding-Spalte. Weitere Spalten werden ignoriert.",
    )
    for f in files or []:
        raw = f.getvalue()
        try:
            df = read_cached(f.name, raw)
        except ValueError as e:
            st.error(f"{f.name}: {e}")
            continue
        sources.append({"name": f.name, "raw": raw, "df": df, "row_idx": None, "label": f.name, "code_default": None})
else:
    f = st.file_uploader(
        "Gesamtdatei mit allen Sprachvarianten (CSV oder Excel)",
        type=FILE_TYPES,
        help="Die URLs werden per Muster (Regex) in Sprachvarianten aufgeteilt.",
    )
    if f is not None:
        raw = f.getvalue()
        try:
            df_all = read_cached(f.name, raw)
        except ValueError as e:
            st.error(f"{f.name}: {e}")
            df_all = None
        if df_all is not None:
            cols_all = list(df_all.columns)
            url_guess_all = io_utils.detect_url_column(df_all)
            url_col_all = st.selectbox(
                "URL-Spalte", cols_all,
                index=cols_all.index(url_guess_all) if url_guess_all in cols_all else 0,
                key="single_url_col",
            )
            urls_all = df_all[url_col_all].astype(str).tolist()
            st.markdown("**Aufteilung per URL-Muster** (Regex, wird von oben nach unten geprüft, erster Treffer gewinnt)")
            sugg = lang_detect.suggest_patterns(urls_all)
            default_rows = pd.DataFrame(
                [{"Muster": s.pattern, "hreflang-Code": s.code, "Beispiel": s.example, "URLs": s.count} for s in sugg],
                columns=["Muster", "hreflang-Code", "Beispiel", "URLs"],
            )
            edited = st.data_editor(
                default_rows, num_rows="dynamic", use_container_width=True,
                disabled=["Beispiel", "URLs"], key="pattern_editor",
            )
            patterns = [(_clean(r["Muster"]), _clean(r["hreflang-Code"])) for _, r in edited.iterrows()]
            groups, rest, invalid = lang_detect.split_by_patterns(urls_all, patterns)
            for p in invalid:
                st.warning(f"Ungültiges Regex-Muster wird ignoriert: `{p}`")
            if rest:
                st.warning(f"{len(rest)} URLs passen auf kein Muster und werden nicht gematcht.")
                with st.expander("URLs ohne Muster anzeigen"):
                    st.dataframe(pd.DataFrame({"URL": [urls_all[i] for i in rest]}), use_container_width=True, hide_index=True)
            for code, idxs in groups.items():
                sources.append({
                    "name": f.name, "raw": raw, "df": df_all.iloc[idxs], "row_idx": tuple(idxs),
                    "label": f"{f.name} · {code}", "code_default": code,
                })

if not sources:
    st.info("Bitte lade deine Crawl-Dateien hoch, um fortzufahren.")
    st.stop()

filter_idx = st.checkbox(
    "Nur URLs mit Status 200 und Indexable / Indexierbar ins Matching aufnehmen",
    value=True,
    help="Nutzt die Spalten 'Status Code'/'Statuscode' und 'Indexability'/'Indexierbarkeit', falls vorhanden.",
)

# ============================================================
# 2. Sprachzuordnung
# ============================================================
st.subheader("2. Sprachzuordnung und Spaltenerkennung")
sets: list[LanguageSet] = []

for i, src in enumerate(sources):
    df = src["df"]
    key = f"src{i}"
    with st.container(border=True):
        st.markdown(f"**{src['label']}** – {len(df)} Zeilen")
        cols = list(df.columns)
        url_guess = io_utils.detect_url_column(df)
        emb_guess = io_utils.detect_embedding_column(df)
        status_col = io_utils.detect_status_column(df)
        index_col = io_utils.detect_indexability_column(df)

        if src["code_default"] is not None:
            code_default = src["code_default"]
        else:
            code_default = lang_detect.suggest_code(df[url_guess].astype(str).tolist()) if url_guess else ""

        c1, c2, c3 = st.columns([1, 2, 2])
        with c1:
            code = st.text_input("hreflang-Code", value=code_default, key=f"{key}_code", help="z. B. de, de-AT, fr-CH")
        with c2:
            url_col = st.selectbox("URL-Spalte", cols, index=cols.index(url_guess) if url_guess in cols else 0, key=f"{key}_url")
        with c3:
            emb_col = st.selectbox("Embedding-Spalte", cols, index=cols.index(emb_guess) if emb_guess in cols else 0, key=f"{key}_emb")

        if emb_guess is None:
            st.warning("Keine Embedding-Spalte automatisch erkannt. Bitte manuell auswählen.")
        if filter_idx:
            missing = [n for n, c in (("Status-Code", status_col), ("Indexierbarkeit", index_col)) if c is None]
            if missing:
                st.caption(f"Hinweis: Spalte {' und '.join(missing)} nicht gefunden, dieser Teilfilter wird übersprungen.")

        code_n = lang_detect.normalize_code(code)
        if not lang_detect.is_valid_code(code_n):
            st.error("Ungültiger hreflang-Code. Erlaubt: Sprache (de) oder Sprache-Region (de-AT).")
            continue

        ls = build_cached(src["name"], src["raw"], code_n, url_col, emb_col, filter_idx, status_col, index_col, src["row_idx"])
        ls.label = src["label"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Gültige URLs", len(ls))
        m2.metric("Embedding-Dimension", ls.dim)
        m3.metric("Verworfen", len(ls.dropped))
        if len(ls.dropped):
            with st.expander(f"Verworfene Zeilen ({len(ls.dropped)})"):
                st.dataframe(ls.dropped, use_container_width=True, hide_index=True)
        if len(ls) == 0:
            st.error("Keine gültigen URLs mit Embeddings in dieser Datei.")
        sets.append(ls)

codes = [s.code for s in sets]
problems: list[str] = []
if len(sets) < 2:
    problems.append("Es werden mindestens zwei Sprachvarianten mit gültigem hreflang-Code benötigt.")
dupes = sorted({c for c in codes if codes.count(c) > 1})
if dupes:
    problems.append(f"Doppelte hreflang-Codes: {', '.join(dupes)}. Jede Sprachvariante braucht einen eigenen Code.")
if any(len(s) == 0 for s in sets):
    problems.append("Mindestens eine Sprachvariante hat keine gültigen URLs.")
try:
    io_utils.check_dimensions(sets)
except ValueError as e:
    problems.append(str(e))
if problems:
    for p in problems:
        st.error(p)
    st.stop()

# ============================================================
# 3. Einstellungen
# ============================================================
st.subheader("3. Einstellungen")
c1, c2 = st.columns(2)
with c1:
    pivot_code = st.selectbox("Pivot-Sprache (Ausgangspunkt für das Matching)", codes,
                              help="Jede andere Sprache wird gegen diese Sprache gematcht.")
    threshold = st.slider("Cosinus-Threshold (Mindest-Ähnlichkeit)", 0.0, 1.0, 0.80, 0.01)
    band = st.slider("Review-Band über dem Threshold", 0.0, 0.2, 0.05, 0.01,
                     help="Treffer, die weniger als diesen Wert über dem Threshold liegen, werden als 'prüfen' markiert.")
    min_margin = st.slider("Mindestabstand zum zweitbesten Kandidaten (Margin)", 0.0, 0.2, 0.02, 0.005,
                           help="Liegt der zweitbeste Kandidat derselben Sprache näher als dieser Wert am besten, gilt der Treffer als mehrdeutig ('prüfen').")
with c2:
    use_slug = st.checkbox("Exact-Slug-Match vor dem Embedding-Matching", value=False,
                           help="Identische Pfade nach Entfernen des Sprachsegments (z. B. /de/x/ ↔ /fr/x/) werden direkt zugeordnet.")
    collisions = lang_detect.find_language_collisions(codes)
    has_region = any("-" in c for c in codes)
    if collisions:
        detail = "; ".join(", ".join(v) for v in collisions.values())
        st.warning(f"Mehrere Varianten derselben Sprache erkannt ({detail}). Im Export wird Sprache-Region verwendet, sonst würden hreflang-Codes kollidieren.")
        mode_opts = [CODE_MODE_REGION]
    else:
        mode_opts = [CODE_MODE_REGION, CODE_MODE_LANG]
    default_idx = 0 if (has_region or collisions or len(mode_opts) == 1) else 1
    code_mode_label = st.radio("hreflang-Codes im Export", mode_opts, index=default_idx)
    code_mode = "language-region" if code_mode_label == CODE_MODE_REGION else "language"
    xd_opts = [NO_XDEFAULT] + codes
    xd_label = st.selectbox("x-default", xd_opts, index=1,
                            help="Fehlt die gewählte Sprache in einem Cluster, wird die Pivot-URL als x-default verwendet und der Cluster markiert.")
    x_default_code = None if xd_label == NO_XDEFAULT else xd_label
    include_review = st.checkbox("Cluster mit Konfidenz 'prüfen' in den HTML-Export aufnehmen", value=False)

# ============================================================
# 4. Start & Ergebnisse
# ============================================================
if st.button("Let's Go", type="primary"):
    with st.spinner("Matching läuft …"):
        result = matching.run_matching(sets, pivot_code, threshold, band, min_margin, use_slug)
        clusters = output.build_clusters(result, sets)
        codes_sorted = [pivot_code] + sorted(c for c in codes if c != pivot_code)
        st.session_state["result"] = {
            "mapping": output.mapping_table(clusters, codes_sorted, pivot_code, x_default_code),
            "unmatched": output.unmatched_table(result, sets),
            "html": output.html_blocks(clusters, pivot_code, x_default_code, code_mode, include_review),
            "stats": result.stats,
            "pivot": pivot_code,
            "codes": codes_sorted,
            "n_clusters": sum(1 for c in clusters if len(c.members) > 1),
            "n_review": sum(1 for c in clusters if len(c.members) > 1 and c.confidence == "prüfen"),
        }

res = st.session_state.get("result")
if res:
    st.subheader("4. Ergebnisse")
    mcols = st.columns(len(res["codes"]))
    for col, code in zip(mcols, res["codes"]):
        s = res["stats"].get(code, {})
        if code == res["pivot"]:
            col.metric(f"{code} (Pivot)", f"{s.get('total', 0) - s.get('unmatched', 0)} / {s.get('total', 0)} zugeordnet")
        else:
            col.metric(code, f"{s.get('slug', 0) + s.get('embedding', 0)} Treffer", f"{s.get('prüfen', 0)} prüfen", delta_color="inverse")
    st.caption(f"{res['n_clusters']} Cluster, davon {res['n_review']} mit Konfidenz „prüfen“.")

    st.markdown("#### Zuordnung (Mapping)")
    st.dataframe(res["mapping"], use_container_width=True, hide_index=True)
    st.download_button("📥 Mapping als CSV herunterladen", res["mapping"].to_csv(index=False).encode("utf-8-sig"),
                       "hreflang_mapping.csv", "text/csv", key="dl_map")

    st.markdown("#### URLs ohne Zuordnung")
    if res["unmatched"].empty:
        st.success("Alle URLs wurden zugeordnet.")
    else:
        st.dataframe(res["unmatched"], use_container_width=True, hide_index=True)
        st.download_button("📥 URLs ohne Zuordnung als CSV herunterladen", res["unmatched"].to_csv(index=False).encode("utf-8-sig"),
                           "hreflang_unmatched.csv", "text/csv", key="dl_unmatched")

    st.markdown("#### hreflang-HTML")
    if not res["html"]:
        st.info("Keine Cluster für den HTML-Export. Prüfe Threshold oder aktiviere „prüfen“-Cluster im Export.")
    else:
        blocks = res["html"].strip().split("\n\n")
        st.code("\n\n".join(blocks[:3]), language="html")
        st.caption(f"Vorschau der ersten {min(3, len(blocks))} von {len(blocks)} Blöcken. Die Datei enthält alle.")
        st.download_button("📥 hreflang-Tags als HTML-Datei herunterladen", res["html"].encode("utf-8"),
                           "hreflang_tags.html", "text/html", key="dl_html")
```

- [ ] **Step 4: Test laufen lassen**

Run: `.venv/bin/python -m pytest tests/test_app.py -v`
Expected: PASS.

- [ ] **Step 5: App manuell starten und Startseite prüfen**

Run: `.venv/bin/python -m streamlit run app.py --server.headless true --server.port 8599 & PID=$!; sleep 8; curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8599; kill $PID`
Expected: `200`

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: Streamlit UI for hreflang matcher

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: End-to-End-Pipeline-Test, README, Abschluss

**Files:**
- Create: `tests/test_pipeline.py`, `README.md`

**Interfaces:**
- Consumes: alle öffentlichen Funktionen aus Tasks 2, 3, 8, 9, 10

- [ ] **Step 1: Pipeline-Test schreiben (CSV-Bytes → LanguageSets → Matching → Ausgabe)**

`tests/test_pipeline.py`:
```python
import numpy as np

from hreflang_matcher import io_utils, lang_detect, matching, output


def _csv(rows: list[tuple[str, np.ndarray, int, str]]) -> bytes:
    lines = ["Address;OpenAI Embedding 1;Status Code;Indexability"]
    for url, vec, status, idx in rows:
        lines.append(f'{url};"[{", ".join(f"{x:.5f}" for x in vec)}]";{status};{idx}')
    return "\n".join(lines).encode("utf-8-sig")


def test_full_pipeline_two_languages():
    rng = np.random.default_rng(42)
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    base = q[:4].astype(np.float32)          # vier orthonormale Vektoren
    noise = lambda: rng.normal(scale=0.05, size=16).astype(np.float32)

    de_rows = [
        ("https://a.com/de/produkt-1/", base[0], 200, "Indexable"),
        ("https://a.com/de/produkt-2/", base[1], 200, "Indexable"),
        ("https://a.com/de/nur-deutsch/", base[2], 200, "Indexable"),
        ("https://a.com/de/geloescht/", base[3], 404, "Non-Indexable"),
    ]
    fr_rows = [
        ("https://a.com/fr/produit-2/", base[1] + noise(), 200, "Indexable"),
        ("https://a.com/fr/produit-1/", base[0] + noise(), 200, "Indexable"),
        ("https://a.com/fr/seulement-fr/", base[3], 200, "Indexable"),
    ]

    sets = []
    for name, rows in (("de.csv", de_rows), ("fr.csv", fr_rows)):
        df = io_utils.read_any_file(name, _csv(rows))
        url_col = io_utils.detect_url_column(df)
        emb_col = io_utils.detect_embedding_column(df)
        assert url_col == "Address" and emb_col == "OpenAI Embedding 1"
        code = lang_detect.suggest_code(df[url_col].astype(str).tolist())
        sets.append(io_utils.build_language_set(
            df, code, name, url_col, emb_col, filter_indexable=True,
            status_col=io_utils.detect_status_column(df), index_col=io_utils.detect_indexability_column(df),
        ))
    assert [s.code for s in sets] == ["de", "fr"]
    assert len(sets[0]) == 3   # 404 gefiltert
    io_utils.check_dimensions(sets)

    result = matching.run_matching(sets, "de", threshold=0.8, band=0.05, min_margin=0.02, use_slug=False)
    pairs = {(m.pivot_url, m.other_url) for m in result.matches}
    assert pairs == {
        ("https://a.com/de/produkt-1/", "https://a.com/fr/produit-1/"),
        ("https://a.com/de/produkt-2/", "https://a.com/fr/produit-2/"),
    }
    assert all(m.confidence == "sicher" for m in result.matches)
    assert result.unmatched["de"] == ["https://a.com/de/nur-deutsch/"]
    assert result.unmatched["fr"] == ["https://a.com/fr/seulement-fr/"]

    clusters = output.build_clusters(result, sets)
    mapping = output.mapping_table(clusters, ["de", "fr"], "de", "de")
    assert len(mapping) == 3
    unmatched = output.unmatched_table(result, sets)
    assert ("de", "https://a.com/de/geloescht/", "Status Code 404") in set(map(tuple, unmatched.values.tolist()))
    html = output.html_blocks(clusters, "de", "de", "language", include_review=False)
    assert html.count('hreflang="x-default"') == 4   # 2 Cluster × 2 Mitglieder
    assert 'href="https://a.com/fr/produit-1/"' in html
```

- [ ] **Step 2: Test laufen lassen**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS. Falls FAIL: Ursache im jeweiligen Modul beheben, nicht den Test anpassen.

- [ ] **Step 3: README schreiben**

`README.md`:
````markdown
# ONE hreflang Matcher

Streamlit-App, die Sprach- und Regionsvarianten einer Website auf Basis von Embeddings
(Cosinus-Ähnlichkeit) einander zuordnet und daraus hreflang-Cluster, eine Mapping-CSV
und fertige `<link rel="alternate" hreflang="…">`-Blöcke erzeugt.

## Lokal starten

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m streamlit run app.py
```

Tests: `.venv/bin/python -m pytest`

## Deployment auf Streamlit Community Cloud

Repository auf GitHub pushen, in Streamlit Cloud „New app“ wählen, `app.py` als Main file.
`requirements.txt` wird automatisch installiert. Es werden keine ML-Modelle geladen,
der Speicherbedarf bleibt gering.

## Eingabedaten

- CSV (empfohlen) oder Excel, eine Datei je Sprachvariante oder eine Gesamtdatei,
  die im Tool per URL-Muster aufgeteilt wird.
- Pflicht: eine URL-Spalte (z. B. `Address`) und eine Embedding-Spalte
  (JSON-Array oder Zahlenliste). Beide werden automatisch erkannt und sind korrigierbar.
- Optional: `Status Code`/`Statuscode` und `Indexability`/`Indexierbarkeit` für den Filter
  auf indexierbare 200er-URLs.
- Alle weiteren Spalten werden ignoriert.
- Embeddings müssen von einem multilingualen Modell stammen und in allen Dateien dieselbe
  Dimension haben. Excel schneidet Zellen bei 32.767 Zeichen ab, deshalb CSV nutzen.

## Ablauf

1. Upload, Erkennung von URL- und Embedding-Spalte, hreflang-Code-Vorschlag je Datei.
2. Pivot-Sprache wählen. Jede andere Sprache wird gegen die Pivot-Sprache gematcht.
3. Optional Exact-Slug-Match als Vorstufe, danach Embedding-Matching mit greedy
   1:1-Zuordnung, Reziprozitäts-Check und Margin zum zweitbesten Kandidaten.
4. Ausgabe: Mapping-CSV, Liste der URLs ohne Zuordnung, HTML-Datei mit einem
   hreflang-Block je Cluster-Mitglied inklusive Selbstreferenz und x-default.

Details zum Design: `docs/superpowers/specs/2026-09-29-hreflang-matcher-design.md`
````

- [ ] **Step 4: Gesamte Test-Suite laufen lassen**

Run: `.venv/bin/python -m pytest -v`
Expected: alle PASS, keine Warnungen über fehlende Module.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline.py README.md
git commit -m "test: end-to-end pipeline test and README

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
