# Embedding-Quelle im Tool – Design-Spezifikation

Datum: 2026-10-15
Status: freigegeben (Design), Umsetzungsplan folgt
Erweitert: `2026-09-29-hreflang-matcher-design.md` (bleibt gültig)

## 1. Ziel

Das Tool soll Embeddings auf Wunsch selbst berechnen, statt sie nur aus dem
Crawl-Export zu lesen. Zwei neue Wege:

- **Aus Textspalten:** Der Nutzer lädt URL plus Text (z. B. „Main Content
  Extraction“, Title, H1) hoch; das Tool schickt die Texte an eine
  Embedding-API.
- **Aus Seiteninhalten:** Der Nutzer lädt nur URLs hoch; das Tool holt die
  Seiten, extrahiert den Haupttext mit Trafilatura und berechnet daraus die
  Embeddings.

**Unveränderlich:** Der bestehende Weg (Embeddings in der Datei) bleibt der
Standard und verhält sich exakt wie heute. Alles ab dem Matching
(`matching.py`, `output.py`, Ergebnisse, Downloads) bleibt unverändert, weil
die neuen Wege dieselben `LanguageSet`-Objekte liefern.

## 2. Nicht-Ziele

- Kein Modell im Tool laden (kein torch, keine sentence-transformers).
- Keine anbieterspezifischen SDKs. Ein generischer OpenAI-kompatibler
  HTTP-Client deckt OpenAI, gehostete Modelle (DeepInfra, Hugging Face,
  Jina, Together …) und lokale Server (Ollama, LM Studio) ab.
- Kein JavaScript-Rendering beim Seitenabruf.
- Keine Persistenz von API-Keys; sie leben nur in der Session.

## 3. Projektstruktur (Ergänzungen)

```
hreflang_matcher/
  embed.py          # OpenAI-kompatibler Embedding-Client, Batches, Retry, Presets
  textprep.py       # Textspalten kombinieren und kürzen, Token-Schätzung
  fetch.py          # Seiten laden + Trafilatura-Extraktion, parallel, mit Gründen
tests/
  test_embed.py     # gegen lokalen HTTP-Testserver (http.server im Test-Thread)
  test_textprep.py
  test_fetch.py     # Extraktion aus HTML-Fixture, Fehlerpfade mit lokalem Server
requirements.txt    # + requests>=2.31, trafilatura>=1.8
```

Die drei Module importieren kein Streamlit.

## 4. Datenmodell (Ergänzungen)

```python
@dataclass
class EmbeddingConfig:
    base_url: str          # z. B. "https://api.openai.com/v1", "http://localhost:11434/v1"
    model: str             # z. B. "text-embedding-3-small", "bge-m3"
    api_key: str           # darf leer sein (lokale Server)
    max_chars: int = 20000 # Kürzung je Text vor dem Versand
    batch_size: int = 64
    timeout: float = 60.0

@dataclass
class EmbedResult:
    vectors: dict[int, np.ndarray]   # Zeilenindex -> Vektor (float32)
    errors: dict[int, str]           # Zeilenindex -> Grund
    dim: int                         # 0 wenn keine Vektoren

@dataclass
class FetchResult:
    texts: dict[int, str]            # Zeilenindex -> extrahierter Haupttext
    errors: dict[int, str]           # Zeilenindex -> Grund ("HTTP 403", "Timeout", "kein Text extrahiert")
```

## 5. Datenfluss

### 5.1 Eingabe (Schritt 1, Erweiterung)

Unter der bestehenden Wahl der Dateistruktur (eine Datei je Sprache /
Gesamtdatei) kommt ein Radio **„Embedding-Quelle“**:

1. **„Embeddings sind in der Datei enthalten“** – Standard. Verhalten wie
   heute, keine neuen Elemente.
2. **„Embeddings aus Textspalten berechnen“**
3. **„Embeddings aus den Seiteninhalten berechnen (Trafilatura)“**

Beide Dateistrukturen funktionieren mit allen drei Quellen. Im Split-Modus
(Gesamtdatei) wird die Quelle je Gruppe gleich behandelt.

### 5.2 Schritt 2 bei Quelle „Textspalten“

Je Quelle (Datei bzw. Gruppe):

- hreflang-Code, URL-Spalte wie heute.
- Statt „Embedding-Spalte“: **Multiselect „Textspalten“**. Vorschlag: alle
  Spalten, deren Name eines der Muster enthält: `main content`,
  `content extraction`, `body`, `text`, `title`, `h1`, `meta description`.
  Reihenfolge der Auswahl = Reihenfolge im zusammengesetzten Text.
- Caption mit Anzahl Texte, mittlerer Länge, geschätzten Tokens
  (`Zeichen / 4`) nach Kürzung.
- Kennzahlen (Gültige URLs, Dimension, Verworfen) erscheinen erst nach der
  Berechnung.

Textaufbau (`textprep.build_texts(df, cols, max_chars)`): Werte der Spalten
als Strings, leere Werte übersprungen, mit `"\n\n"` verbunden, Whitespace
normalisiert, auf `max_chars` gekürzt. Zeilen mit leerem Gesamttext werden
mit Grund „kein Text in den gewählten Spalten“ verworfen.

### 5.3 Schritt 2 bei Quelle „Seiteninhalte“

Je Quelle nur hreflang-Code und URL-Spalte. Caption mit Anzahl URLs.
**Ab 500 URLs je Lauf** erscheint eine Warnung (kein hartes Limit): Der
Abruf dauert lange, die Session muss offen bleiben, der Crawl-Weg mit
Screaming Frog ist für große Sites besser.

Abruf (`fetch.fetch_main_texts(urls, concurrency=5, timeout=10)`):

- `requests.get` mit User-Agent `ONE-hreflang-Matcher/1.0 (+https://onebeyondsearch.com)`,
  `timeout=10`, Weiterleitungen erlaubt, max. 5 parallele Threads
  (`ThreadPoolExecutor`).
- Nur `text/html` mit Status 200 wird verarbeitet; sonst Grund
  `HTTP <status>` bzw. `kein HTML (<content-type>)`.
- Extraktion mit `trafilatura.extract(html, include_comments=False,
  include_tables=False, favor_precision=True, fast=True)`. `fast=True`
  schaltet die readability/justext-Fallbacks ab: Auf echten Seiten
  (Wikipedia, Google Docs, Unternehmensseiten, geprüft 2026-10-15) ist das
  Ergebnis identisch, bei reinen Navigationsseiten verhindert es, dass
  Menütexte als Inhalt durchgehen. Leeres Ergebnis → Grund
  „kein Text extrahiert“. Netzwerkfehler → „Timeout“ oder
  „Verbindungsfehler: <kurz>“.
- Fortschrittsbalken in der UI über einen Callback `on_progress(done, total)`.

Die extrahierten Texte durchlaufen danach dieselbe Kürzung wie 5.2.

### 5.4 Embedding-Einstellungen (neuer Expander in Schritt 2)

Sichtbar nur bei Quelle 2 oder 3:

- **Preset** (Selectbox): „OpenAI“, „Ollama (lokal)“, „LM Studio (lokal)“,
  „Eigener OpenAI-kompatibler Endpunkt“. Presets setzen Base-URL und
  Modell vor:
  - OpenAI: `https://api.openai.com/v1`, Modellauswahl
    `text-embedding-3-small` (Standard) / `text-embedding-3-large`
  - Ollama: `http://localhost:11434/v1`, Modell `bge-m3`
  - LM Studio: `http://localhost:1234/v1`, Modell leer, Nutzer trägt ein
  - Eigener Endpunkt: beide Felder leer
- **Base-URL**, **Modell** (Textfelder, bei Presets vorbelegt, editierbar).
- **API-Key** (Passwortfeld; bei lokalen Presets optional, Hinweis).
- **Max. Zeichen je Text** (Slider 2.000–60.000, Standard 20.000).
- Hinweis bei lokalen Presets: „Lokale Endpunkte funktionieren nur, wenn
  das Tool auf demselben Rechner läuft. Streamlit Cloud erreicht dein
  Localhost nicht.“
- Button **„Verbindung testen“**: schickt einen Ein-Wort-Text, zeigt
  Dimension oder Fehlertext.

### 5.5 Berechnung (expliziter Button)

Button **„Embeddings berechnen“** (primary) unterhalb der Quellen. Keine
API-Anfrage läuft ohne diesen Klick; Slider, Spaltenwechsel und Reruns
lösen nichts aus. Ablauf je Quelle:

1. Texte bauen (5.2) oder abrufen (5.3).
2. Je Text ein Schlüssel `sha1(model + "|" + base_url + "|" + text)`.
   Vektoren mit bekanntem Schlüssel kommen aus dem Session-Cache
   `st.session_state["emb_cache"]` (dict Schlüssel → Vektor, begrenzt
   auf 50.000 Einträge, FIFO).
3. Fehlende Texte in Batches von 64 an `POST {base_url}/embeddings` mit
   `{"model": model, "input": [texte]}` und Header
   `Authorization: Bearer <key>` (nur wenn Key gesetzt).
4. Antwortformat OpenAI: `data[i].embedding`, Reihenfolge wie Eingabe.
5. Fehler: HTTP 429/5xx und Netzwerkfehler → bis zu 3 Wiederholungen mit
   Wartezeit 2, 4, 8 s. HTTP 4xx außer 429 → Batch gilt als fehlgeschlagen,
   jede Zeile bekommt den Grund `Embedding-Anfrage fehlgeschlagen: HTTP <code> <kurzer Text>`.
   Bei einem Fehler mit 400 wegen Länge wird der Batch einmal mit halbiertem
   `max_chars` wiederholt, danach Grund.
6. Fortschrittsbalken je Quelle.
7. Ergebnis: `LanguageSet` wie heute über `build_language_set`-Logik
   (Normalisierung, Duplikate, Nullvektoren, Dimensionsmehrheit), aber mit
   bereits vorhandenen Vektoren statt Parsing. Dafür bekommt `io_utils`
   eine Funktion `language_set_from_vectors(df, code, label, url_col,
   vectors: dict[int, np.ndarray], errors: dict[int, str], filter_indexable,
   status_col, index_col) -> LanguageSet`, die denselben Filter- und
   Verwerfungspfad nutzt (gemeinsame Hilfsfunktion, kein Duplikat).
8. Die fertigen Sets landen in `st.session_state["computed"]` mit einer
   Signatur (Quellen-Keys, Spalten, Konfig, max_chars). Weicht die aktuelle
   Signatur ab, zeigt die UI „Embeddings veraltet, bitte neu berechnen“ und
   blockiert „Let's Go“ bis zur Neuberechnung.

Danach greift Schritt 3 und 4 unverändert.

### 5.6 Export der berechneten Embeddings

Unter den Kennzahlen je Quelle ein Download **„Embeddings als CSV“**:
Spalten `Address`, `Embedding` (JSON-Array mit 6 Nachkommastellen),
Dateiname `embeddings_<code>.csv`, utf-8-sig. Die Datei ist im heutigen
Modus „in der Datei enthalten“ direkt verwendbar (Spaltenerkennung greift
über den Namen `Embedding`).

### 5.7 Fehlerbehandlung

- Verbindung zum Endpunkt scheitert komplett (erste Anfrage): Fehlermeldung
  mit Base-URL und Ursache, keine Teilsets, Button bleibt aktiv.
- Teilweise Fehler: Set wird gebildet, fehlgeschlagene Zeilen stehen in
  „Verworfen“ mit Grund.
- Dimensionen unterschiedlich (Modell gewechselt zwischen Läufen): der
  bestehende `check_dimensions` meldet es; Hinweis, den Cache zu leeren
  (Button „Embedding-Cache leeren“ im Expander).
- Kein Key bei OpenAI-Preset: Fehler vor dem Versand.

### 5.8 Kosten- und Zeithinweise

Vor dem Button: „N Texte, ca. T Tokens“ (Zeichen/4). Bei OpenAI zusätzlich
die Preisklasse als Caption („text-embedding-3-small: ca. 0,02 $ je 1 Mio.
Tokens“). Keine Live-Preisabfrage. Der Hilfetext weist darauf hin, dass der sinnvolle Threshold modellabhängig ist (BGE-M3 typischerweise 0,65–0,80, OpenAI text-embedding-3 meist > 0,85).

### 5.9 Performance und Grenzen

- Batches von 64, maximal 5 parallele Batches beim Embedding, 5 parallele
  Abrufe bei Trafilatura.
- Texte werden nicht dauerhaft gehalten; nach der Berechnung bleiben nur
  Vektoren und Verwerfungsgründe in der Session.
- Zielgröße wie bisher: 5.000 URLs je Sprache. Trafilatura-Modus realistisch
  bis einige hundert URLs je Lauf.

## 6. Tests

- `test_textprep`: Spaltenkombination, Reihenfolge, leere Werte,
  Whitespace, Kürzung, leerer Gesamttext, Token-Schätzung.
- `test_embed`: lokaler `http.server` im Test liefert OpenAI-Format;
  Batching, Reihenfolge, Header mit/ohne Key, Retry bei 429 und 500,
  Grund bei 401, Halbierung bei 400-Längenfehler, Dimension.
- `test_fetch`: Extraktion aus HTML-Fixture (ohne Netz, `trafilatura.extract`
  direkt); lokaler Server für 200/404/Timeout/nicht-HTML; Parallelität
  liefert vollständige, korrekt zugeordnete Ergebnisse.
- `test_io_utils`: `language_set_from_vectors` mit Filtern, Fehlern,
  Duplikaten, Nullvektoren; CSV-Export-Roundtrip (Export → `read_any_file`
  → `detect_embedding_column` → `build_language_set` ergibt gleiche Vektoren).
- `test_app`: AppTest mit gefaktem Uploader und gemocktem `embed_texts`
  (kein Netz): Quelle „Textspalten“ bis zu den Ergebnissen, Button-Gate
  (keine Berechnung ohne Klick), Veraltet-Hinweis bei Spaltenwechsel.

## 7. Abhängigkeiten

```
requests>=2.31
trafilatura>=1.8
```
Beide Pakete verlangen **Python 3.10 oder neuer** (geprüft am 2026-10-15:
trafilatura 2.3.1, requests 2.34.2). Damit endet die Unterstützung für
Python 3.9: `pyproject.toml` bekommt `requires-python = ">=3.10"`, die
lokale Entwicklungsumgebung läuft mit Python 3.14 (angelegt über `uv venv
--python 3.14 .venv`), Streamlit Cloud nutzt 3.12+. Die Module behalten
`from __future__ import annotations`, nutzen aber keine 3.9-Rücksichten
mehr. trafilatura bringt lxml mit vorgefertigten Wheels mit; auf Python
3.14 ist der Import geprüft.
