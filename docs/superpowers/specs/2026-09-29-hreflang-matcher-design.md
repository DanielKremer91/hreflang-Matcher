# ONE hreflang Matcher – Design-Spezifikation

Datum: 2026-09-29
Status: freigegeben (Design), Umsetzungsplan folgt

## 1. Ziel

Eine Streamlit-App, die für mehrere Sprach- bzw. Regionsvarianten einer Website
die zueinander passenden URL-Paare auf Basis von Embeddings (Cosinus-Ähnlichkeit)
ermittelt und daraus hreflang-Cluster sowie fertigen HTML-Code erzeugt.

Gematcht wird ausschließlich **zwischen** Sprachvarianten (DE-URL zu bester
FR-URL, EN-URL usw.), niemals innerhalb einer Sprache.

Deployment: Streamlit Community Cloud aus einem GitHub-Repository. Daraus folgt:
leichte Abhängigkeiten (pandas, numpy, openpyxl, streamlit), keine Embedding-
Erzeugung im Tool (kein torch, keine sentence-transformers, kein faiss).

UI-Sprache: nur Deutsch. Look and Feel: ONE Beyond Search (Logo, rote
Download-Buttons, Info-Box mit LinkedIn-Verweis, Hilfe-Expander, nummerierte
Schritte).

## 2. Nicht-Ziele (bewusst weggelassen)

- Keine Embedding-Erzeugung im Tool.
- Keine hreflang-XML-Sitemap.
- Kein Abgleich mit bereits vorhandenen hreflang-Tags aus dem Crawl.
- Keine englische UI.
- Kein vollständig paarweises Matching aller Sprachpaare (siehe Pivot-Logik).

## 3. Projektstruktur

```
one-hreflang-matcher/
  app.py                    # Streamlit-UI, Branding, Orchestrierung; keine Fachlogik
  hreflang_matcher/
    __init__.py
    io_utils.py             # Dateien lesen, Spalten erkennen, Vektoren parsen
    lang_detect.py          # hreflang-Code aus URL ableiten, Gesamtdatei splitten
    matching.py             # Slug-Match, Cosinus, Reziprozität, 1:1-Zuordnung, Konfidenz
    output.py               # Mapping-Tabelle, Unmatched-Liste, HTML-Blöcke
  tests/
    test_io_utils.py
    test_lang_detect.py
    test_matching.py
    test_output.py
  requirements.txt
  .streamlit/config.toml    # ONE-Theme (Primärfarbe Rot)
  README.md
  docs/superpowers/specs/   # diese Spec
```

Die Module unter `hreflang_matcher/` importieren kein Streamlit. Sie arbeiten
auf pandas/numpy und einfachen Datenklassen und sind mit pytest testbar.

## 4. Datenmodell

```python
@dataclass
class LanguageSet:
    code: str                 # hreflang-Code, z. B. "de", "de-AT", "fr-CH"
    label: str                # Anzeigename, z. B. Dateiname oder Muster
    urls: list[str]           # Original-URLs in Eingabereihenfolge
    norm_urls: list[str]      # normalisierte URLs (für Matching)
    vectors: np.ndarray       # Form (n, dim), float32, L2-normalisiert
    dropped: pd.DataFrame     # URLs, die verworfen wurden, mit Grund

@dataclass
class Match:
    pivot_url: str
    other_url: str
    other_code: str
    score: float              # 1.0 bei Slug-Match, sonst Cosinus
    method: str               # "slug" | "embedding"
    reciprocal: bool          # bester Treffer in beide Richtungen
    second_url: str | None    # zweitbester Kandidat derselben Sprache
    second_score: float | None
    margin: float | None      # score - second_score
    confidence: str           # "sicher" | "prüfen"
    reason: str               # leer bei "sicher", sonst z. B. "knapp über Threshold",
                              # "nicht reziprok", "mehrdeutig (Margin 0.01)"

@dataclass
class MatchResult:
    pivot_code: str
    matches: list[Match]
    unmatched: dict[str, list[str]]   # code -> URLs ohne Zuordnung (inkl. Pivot)
    stats: dict                       # Zähler je Sprache und Stufe
```

## 5. Datenfluss

### 5.1 Upload (Schritt 1)

Zwei Modi, per Radio wählbar:

**Modus A – eine Datei je Sprache.** `st.file_uploader(accept_multiple_files=True)`,
CSV oder XLSX. Jede Datei wird zu einem `LanguageSet`.

**Modus B – eine Gesamtdatei.** Eine Datei, die alle Sprachvarianten enthält.
Das Tool schlägt Muster vor (siehe 5.3) und splittet die URLs danach in
`LanguageSet`s. URLs, die auf kein Muster passen, landen in einer Restliste
und werden angezeigt, aber nicht gematcht.

Dateien werden mit `read_any_file` gelesen: CSV mit Encoding-Fallback
(utf-8-sig, utf-8, cp1252, latin1) und Separator-Erkennung; XLSX über openpyxl.
Lesen ist mit `st.cache_data` über (Dateiname, Bytes) gecacht.

### 5.2 Spaltenerkennung (io_utils)

**URL-Spalte:** zuerst exakte Namen (address, url, page, adresse, seite,
landing page), dann Teilstring, dann Inhaltsprüfung (≥ 20 % der ersten 50
Werte sehen wie URLs aus). Ergebnis ist im UI als Selectbox korrigierbar.

**Embedding-Spalte:** zuerst Spaltenname mit „embed“ oder „vector“, sonst
Inhaltsprüfung: die ersten 20 nicht-leeren Werte müssen als Zahlenliste
parsebar sein (JSON-Array oder durch Komma/Semikolon/Leerzeichen/Pipe
getrennte Floats) mit mindestens 8 Elementen. Auch korrigierbar.

**Vektor-Parsing:** JSON zuerst, dann Regex-Float-Extraktion. Dimension je
Datei = häufigste Länge. Vektoren mit anderer Länge werden **verworfen**,
nicht gepaddet, und mit Grund („Dimension 1021 statt 1536, vermutlich
abgeschnitten“) in `dropped` gemeldet. Über alle `LanguageSet`s muss die
Dimension gleich sein, sonst harter Fehler mit Hinweis auf gemischte Modelle
oder Excel-Zellenlimit (32.767 Zeichen).

**Optionale Spalten:** Status-Code (Spaltenname `Status Code` oder
`Statuscode`) und Indexierbarkeit (`Indexability` oder `Indexierbarkeit`).
Werden nur genutzt, wenn der Indexierbarkeits-Schalter aktiv ist (siehe 5.5).

**Alle anderen Spalten werden ignoriert.** Screaming-Frog-Exporte mit
Dutzenden Spalten sind ausdrücklich erlaubt.

**URL-Normalisierung** (für Matching, nie für Ausgabe): Schema entfernen,
Host lowercase, `www.` entfernen, Fragment entfernen, Tracking-Parameter
(utm_*, gclid, fbclid) entfernen, übrige Query-Parameter alphabetisch
sortieren, trailing slash außer bei Root entfernen. Duplikate nach
Normalisierung innerhalb einer Sprache: erste Zeile gewinnt, weitere werden
als verworfen gemeldet.

### 5.3 Sprachzuordnung (Schritt 2, lang_detect)

Für jedes `LanguageSet` wird ein hreflang-Code vorgeschlagen, aus den URLs
abgeleitet, in dieser Reihenfolge:

1. Erstes Pfadsegment, wenn es einem Muster `^[a-z]{2}(-[a-z]{2})?$` entspricht
   (z. B. `/de/`, `/de-at/`, `/fr-ch/`), mehrheitlich über die URLs der Datei.
2. Subdomain, wenn sie zwei Buchstaben lang ist (`fr.example.com`).
3. ccTLD (`.de`, `.fr`, `.at`, `.ch`), wobei `.ch` und `.at` ohne Sprache nur
   als Hinweis dienen und ein leerer Vorschlag ausgegeben wird.
4. Sonst leer; der Nutzer muss eintragen.

Der Code ist im UI je Datei ein Textfeld. Validierung: Muster
`^[a-z]{2}(-[A-Za-z]{2})?$` oder `x-default` ist nicht erlaubt (x-default wird
separat gewählt). Doppelte Codes blockieren „Let's Go“.

**Split-Modus (Modus B):** `lang_detect.suggest_patterns(urls)` liefert eine
Liste von (Muster, Beispiel-URL, Anzahl) aus Pfadsegmenten, Subdomains und
TLDs. Der Nutzer sieht eine editierbare Tabelle: Muster (Regex), hreflang-Code.
Jede URL wird dem ersten passenden Muster zugeordnet (top-down).

### 5.4 Einstellungen (Schritt 3)

- **Pivot-Sprache:** Selectbox über die erkannten Codes.
- **Cosinus-Threshold:** Slider 0.0–1.0, Default 0.80, Schritt 0.01.
- **Review-Band:** Slider 0.0–0.2, Default 0.05. Treffer mit
  `threshold <= score < threshold + band` gelten als „prüfen“, weil sie nur
  knapp über der Schwelle liegen.
- **Mindestabstand zum Zweitbesten (Margin):** Slider 0.0–0.2, Default 0.02.
  Liegt der zweitbeste Kandidat derselben Sprache weniger als diesen Wert
  unter dem besten, gilt der Treffer als mehrdeutig und wird „prüfen“. Der
  zweitbeste Kandidat wird mit URL und Score in der Ausgabe genannt.
- **Slug-Match aktivieren:** Checkbox, Default aus. Läuft vor dem
  Embedding-Matching.
- **Nur Status 200 und Indexable:** Checkbox, Default an.
- **hreflang-Codes im Export:** Radio „nur Sprache“ (de, fr) oder
  „Sprache-Region“ (de-AT, fr-CH). Default: „Sprache-Region“, wenn mindestens
  ein Code eine Region enthält, sonst „nur Sprache“. Erkennt das Tool zwei
  Sets mit derselben Sprache (z. B. de und de-AT), wird ein Hinweis
  angezeigt, dass Sprache-Region nötig ist, und „nur Sprache“ ist nicht
  wählbar, weil die Codes sonst kollidieren würden. Codes werden normalisiert
  ausgegeben: Sprache klein, Region groß (de-AT).
- **x-default:** Selectbox über die Codes plus Option „kein x-default“.
- **„prüfen“-Cluster in HTML-Export aufnehmen:** Checkbox, Default aus.

### 5.5 Indexierbarkeits-Filter

Wenn aktiv und die Datei eine Status-Code-Spalte hat: nur Zeilen mit 200.
Wenn aktiv und eine Indexierbarkeits-Spalte vorhanden ist: nur Zeilen, deren
Wert case-insensitiv „indexable“ oder „indexierbar“ ist. Werte wie
„Non-Indexable“ oder „Nicht indexierbar“ werden ausgeschlossen (Prüfung:
Wert beginnt mit „indexable“ oder „indexierbar“, nicht mit „non“ oder „nicht“).
Fehlt eine der Spalten, wird der jeweilige Teilfilter übersprungen und eine
Warnung je Datei angezeigt. Gefilterte URLs landen in `dropped` mit Grund.

### 5.6 Matching (matching.py)

Für jedes Nicht-Pivot-`LanguageSet` `L` gegen das Pivot-Set `P`:

**Stufe 1 – Slug-Match (optional).**
`strip_language(norm_url, code)` entfernt Sprachsegment im Pfad, Sprach-
Subdomain und ersetzt den Host durch einen Platzhalter. Zwei URLs matchen,
wenn der Rest identisch ist. Jeder Slug-Treffer ist ein `Match` mit
`score=1.0, method="slug", reciprocal=True, confidence="sicher"`. Gematchte
URLs werden aus beiden Seiten für Stufe 2 entfernt. Bei mehreren P-URLs mit
gleichem Rest (nach Normalisierung sollte das nicht vorkommen) gewinnt die
erste.

**Stufe 2 – Embedding-Match.**
Cosinus-Matrix `S = P_rest @ L_rest.T` blockweise (Zeilenblöcke von 1024), da
Vektoren L2-normalisiert sind. Je Pivot-Zeile werden bester und zweitbester
Index gemerkt; je L-Spalte der beste Pivot-Index (für Reziprozität).

Kandidaten: alle (i, j) mit `S[i, j] >= threshold`, aber nur die Top-5 je
Pivot-Zeile, um Speicher zu begrenzen. Kandidaten werden nach Score
absteigend sortiert und **greedy 1:1** zugeordnet: ein Paar wird
übernommen, wenn weder i noch j bereits vergeben sind.

Bekannte Einschränkung: Durch die Top-5-Grenze kann eine Pivot-URL ohne
Zuordnung bleiben, obwohl ein sechstbester Kandidat noch frei gewesen wäre.
Das ist bei sinnvollen Thresholds praktisch irrelevant und wird akzeptiert.

`reciprocal = (argmax_j S[i, :] == j) and (argmax_i S[:, j] == i)`.

**Konfidenz:** Ein Treffer ist `sicher`, wenn alle drei Bedingungen gelten:
1. `score >= threshold + band` (nicht nur knapp über der Schwelle),
2. `reciprocal` (bester Treffer in beide Richtungen),
3. `margin >= min_margin` oder es gibt keinen zweitbesten Kandidaten.

Sonst `prüfen`, und `reason` nennt alle verletzten Bedingungen. Der
zweitbeste Kandidat wird immer mitgeführt, damit man im Ergebnis sieht,
welche zwei URLs derselben Sprache eng beieinander lagen. Paare unter
`threshold` werden nicht erzeugt.

**Unmatched:** alle P- und L-URLs ohne Zuordnung, je Sprache.

**Stats:** je Sprache Anzahl Slug-Treffer, Embedding-Treffer, sicher, prüfen,
unmatched Pivot, unmatched L.

### 5.7 Cluster und Ausgabe (output.py)

**Cluster:** Pivot-URL plus alle zugeordneten URLs der anderen Sprachen.
Cluster-Konfidenz = „prüfen“, sobald ein Mitglied „prüfen“ ist, sonst
„sicher“. Nicht-Pivot-URLs ohne Pivot-Treffer bilden keinen Cluster.

**Mapping-Tabelle (wide):** Spalten
`Pivot-URL (<code>)`, dann je andere Sprache
`URL (<code>)`, `Score (<code>)`, `Methode (<code>)`, `Konfidenz (<code>)`,
`Grund (<code>)`, `Zweitbeste URL (<code>)`, `Zweitbester Score (<code>)`,
zuletzt `Cluster-Konfidenz` und `x-default-Fallback` (ja/nein). Eine Zeile je
Pivot-URL, auch wenn kein einziger Treffer vorliegt (dann leere Zellen und
`Cluster-Konfidenz` = „kein Treffer“).

**Unmatched-Tabelle (long):** Spalten `Sprache`, `URL`, `Grund`
(„kein Treffer über Threshold“ oder Verwerfungsgrund aus `dropped`).

**HTML-Blöcke:** je Cluster ein Block, je Mitglied ein eigener Satz Tags.
Für Mitglied M des Clusters C:

```html
<!-- https://example.com/de/seite/ -->
<link rel="alternate" hreflang="de" href="https://example.com/de/seite/" />
<link rel="alternate" hreflang="fr" href="https://example.com/fr/page/" />
<link rel="alternate" hreflang="x-default" href="https://example.com/de/seite/" />
```

Regeln:
- Reihenfolge: Pivot zuerst, dann übrige Codes alphabetisch, x-default zuletzt.
- Selbstreferenz ist immer enthalten (M steht in seiner eigenen Liste).
- Die Tag-Liste ist für alle Mitglieder eines Clusters identisch; der Kommentar
  darüber nennt die Seite, in die der Block gehört.
- x-default: URL der gewählten x-default-Sprache im Cluster. Fehlt sie, wird
  die Pivot-URL genommen und der Cluster in der Mapping-Tabelle mit
  `x-default-Fallback = ja` markiert. Ist „kein x-default“ gewählt, entfällt
  die Zeile.
- Cluster mit nur einem Mitglied (kein Treffer) erzeugen keinen Block.
- Cluster mit Konfidenz „prüfen“ nur, wenn der Schalter aktiv ist.
- `href` ist immer die Original-URL, nie die normalisierte.
- Der hreflang-Wert folgt dem gewählten Modus (nur Sprache oder
  Sprache-Region), normalisiert als `de` bzw. `de-AT`.
- HTML-Escaping der URLs (`&` → `&amp;`).

**Downloads:** `hreflang_mapping.csv`, `hreflang_unmatched.csv`,
`hreflang_tags.html` (reiner Text mit den Blöcken, durch Leerzeilen getrennt).
Alle als utf-8-sig bzw. utf-8.

### 5.8 UI-Ablauf (app.py)

1. Branding-Header wie bei ONE Link Intelligence: Logo, Titel
   „ONE hreflang Matcher“, graue Box „Entwickelt von Daniel Kremer von
   ONE Beyond Search | Folge mir auf LinkedIn …“ mit denselben Links,
   Trennlinie, roter Download-Button-Style, Hilfe-Expander mit Erklärung
   inkl. Hinweis auf multilinguale Embedding-Modelle und Excel-Zellenlimit.
2. Schritt 1 Upload (Modus A/B).
3. Schritt 2 Sprachzuordnung: Tabelle je Datei bzw. Muster mit Code, Anzahl
   URLs, erkannter Dimension, verworfene Zeilen (Expander mit Details).
4. Schritt 3 Einstellungen.
5. „Let's Go“ (primary). Ergebnisse werden in `st.session_state` gehalten,
   damit Downloads keinen Re-Run auslösen.
6. Ergebnisse: Kennzahlen je Sprache (st.metric), Mapping-Tabelle,
   Unmatched-Tabelle, HTML-Vorschau (st.code, erste 3 Cluster), Downloads.

### 5.9 Fehlerbehandlung

- Weniger als zwei `LanguageSet`s mit gültigen Vektoren: Fehler, Stopp.
- Dimensionen ungleich: Fehler mit Liste je Datei, Stopp.
- Pivot-Set leer nach Filter: Fehler, Stopp.
- Keine Embedding-Spalte erkannt: Fehler mit Hinweis auf manuelle Auswahl.
- Ungültige Regex im Split-Modus: Warnung, Muster wird ignoriert.
- Alle Verwerfungen sind je Datei im Expander einsehbar, nie stumm.

### 5.10 Performance

Ziel: 5 000 URLs je Sprache und 5 Sprachen laufen auf Streamlit Cloud
(1 GB RAM) durch; größere Crawls werden aufgeteilt. Blockweises Matmul mit float32; niemals die volle
n×m-Matrix für n, m > 2048 im Speicher halten. Top-5 je Zeile statt aller
Kandidaten. Lesen und Parsen gecacht.

## 6. Tests

pytest, keine Streamlit-Abhängigkeit:

- `test_io_utils`: Spaltenerkennung (Name, Inhalt), Vektor-Parsing (JSON,
  Komma, Semikolon, abgeschnitten), Dimensions-Mehrheit, Normalisierung,
  Duplikate.
- `test_lang_detect`: Code aus Pfad, Subdomain, TLD; Normalisierung
  (de-at → de-AT); Muster-Vorschlag; Split top-down; Restliste.
- `test_matching`: Slug-Match mit Sprachsegment; Cosinus auf synthetischen
  Vektoren; Reziprozität; greedy 1:1 bei Konflikt; Threshold und Band;
  Konfidenz inkl. Margin und Grund; Zweitbester Kandidat; Unmatched;
  Blockgrenzen (n > Blockgröße).
- `test_output`: Wide-Tabelle; Unmatched; HTML-Block mit Selbstreferenz,
  Reihenfolge, x-default-Fallback, Escaping, „prüfen“-Schalter; Code-Modus
  nur Sprache vs. Sprache-Region; Kollisionserkennung (de + de-AT).

## 7. Abhängigkeiten

```
streamlit>=1.36
pandas>=2.0
numpy>=1.26
openpyxl>=3.1
pytest>=8   # nur dev
```
