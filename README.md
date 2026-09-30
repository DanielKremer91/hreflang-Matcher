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

## Getestete Umgebungen

- Python 3.9 mit streamlit 1.50, pandas 2.3, numpy 2.0
- Python 3.14 mit streamlit 1.64, pandas 3.0, numpy 2.5 (Streamlit-Cloud-Standard)

Beide Stacks laufen die komplette Testsuite (124 Tests) ohne Warnungen durch.

## Deployment auf Streamlit Community Cloud

Repository auf GitHub pushen, in Streamlit Cloud „New app" wählen, `app.py` als Main file.
`requirements.txt` wird automatisch installiert. Es werden keine ML-Modelle geladen,
der Speicherbedarf bleibt gering.

## Grenzen

- Streamlit Community Cloud stellt etwa 1 GB RAM bereit.
- Geprüft und komfortabel: bis etwa 5.000 URLs je Sprache bei 1536 Dimensionen und 5 Sprachen.
- Größere Crawls je Verzeichnis aufteilen oder die App lokal starten.
- Upload-Limit: 200 MB je Datei (`server.maxUploadSize`).
- CSV-Exporte ohne unnötige Spalten halten den Speicherbedarf niedrig.

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
- Leere Embeddings (Screaming Frog konnte die Anfrage nicht ausführen, z. B. „input length exceeds the
  context length“) werden verworfen; der Grund aus „Prompt Request Status“ wird mit ausgegeben.

## Ablauf

1. Upload, Erkennung von URL- und Embedding-Spalte, hreflang-Code-Vorschlag je Datei.
2. Pivot-Sprache wählen. Jede andere Sprache wird gegen die Pivot-Sprache gematcht.
3. Optional Exact-Slug-Match als Vorstufe, danach Embedding-Matching mit greedy
   1:1-Zuordnung, Reziprozitäts-Check und Margin zum zweitbesten Kandidaten.
4. Ausgabe: Mapping-CSV, Liste der URLs ohne Zuordnung, HTML-Datei mit einem
   hreflang-Block je Cluster-Mitglied inklusive Selbstreferenz und x-default.

Details zum Design: `docs/superpowers/specs/2026-09-29-hreflang-matcher-design.md`
