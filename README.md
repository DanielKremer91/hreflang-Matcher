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

- Python 3.14 mit streamlit 1.66, pandas 3.0, numpy 2.5, trafilatura 2.3, requests 2.34 (Streamlit-Cloud-Standard)

Python 3.10 oder neuer wird benötigt (trafilatura und requests), außerdem streamlit 1.66 oder neuer
(die Embedding-CSV wird erst beim Klick auf den Download erzeugt). Die Testsuite läuft ohne Warnungen durch.

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
- Speicher beim Berechnen im Tool: ca. 6 KB je URL bei 1536 Dimensionen, ca. 12 KB bei 3072. Bei 5 × 5.000 URLs
  sind das rund 150–300 MB für Vektoren plus Embedding-Cache. Große Läufe nach Sprachen aufteilen oder lokal rechnen.
- Trafilatura-Modus: realistisch bis einige hundert URLs je Lauf; ab 500 URLs warnt das Tool.
  Seiten, die JavaScript zum Rendern brauchen, liefern wenig oder keinen Text.

## Sicherheit

Der Trafilatura-Modus ruft nur öffentliche http(s)-Adressen ab. Interne oder lokale Ziele (localhost, private
Netze, Link-Local wie die Cloud-Metadaten-Adresse 169.254.169.254) werden abgelehnt, auch wenn eine Seite per
Weiterleitung dorthin zeigt. Je Seite gelten höchstens 5 Weiterleitungen, 5 MB und eine Gesamtfrist. Lokale
Embedding-Endpunkte auf localhost (Ollama, LM Studio) sind dagegen bewusst erlaubt; private Netze und
Link-Local bleiben auch dort gesperrt.

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
- Alternativ berechnet das Tool die Embeddings selbst („Embedding-Quelle“): aus Textspalten
  (z. B. Main Content Extraction, Title, H1) oder aus den Seiteninhalten per Trafilatura. Dafür ist
  ein OpenAI-kompatibler Endpunkt nötig: OpenAI (API-Key), ein gehosteter Dienst oder lokal
  Ollama/LM Studio (nur bei lokalem Start des Tools). Berechnete Embeddings lassen sich als CSV
  exportieren und später ohne API-Kosten wieder hochladen.

## Ablauf

1. Upload, Erkennung von URL- und Embedding-Spalte, hreflang-Code-Vorschlag je Datei.
   Bei Quelle „berechnen“: Textspalten wählen oder URLs laden, Endpunkt einstellen, „Embeddings berechnen“ klicken.
2. Pivot-Sprache wählen. Jede andere Sprache wird gegen die Pivot-Sprache gematcht.
3. Optional Exact-Slug-Match als Vorstufe, danach Embedding-Matching mit greedy
   1:1-Zuordnung, Reziprozitäts-Check und Margin zum zweitbesten Kandidaten.
4. Ausgabe: Mapping-CSV, Liste der URLs ohne Zuordnung, HTML-Datei mit einem
   hreflang-Block je Cluster-Mitglied inklusive Selbstreferenz und x-default.

## Threshold kalibrieren

Welche Cosinus-Werte „gut“ sind, hängt vom Modell ab: OpenAI text-embedding-3 liefert für echte Übersetzungen
meist 0,85 und mehr, BGE-M3 oft nur 0,65–0,80. Setze den Threshold beim ersten Lauf niedrig (z. B. 0,60), schau dir
die Score-Verteilung in der Mapping-Tabelle an und ziehe ihn dann so weit hoch, dass die klaren Fehlzuordnungen verschwinden.

Details zum Design: `docs/superpowers/specs/2026-09-29-hreflang-matcher-design.md`
