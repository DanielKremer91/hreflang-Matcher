from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from hreflang_matcher import embed, fetch, io_utils, lang_detect, matching, output, textprep
from hreflang_matcher.models import EmbeddingConfig, LanguageSet
from hreflang_matcher.ui_keys import MODE_FILE, MODE_GROUP, source_key

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

**Embedding-Quelle:** Standard ist „in der Datei enthalten“ (z. B. aus Screaming Frog). Alternativ berechnet das
Tool die Embeddings selbst: aus gewählten Textspalten (Main Content, Title, H1 …) oder aus den Seiteninhalten,
die es mit Trafilatura extrahiert. Dafür brauchst du einen OpenAI-kompatiblen Endpunkt: OpenAI mit API-Key,
oder lokal Ollama/LM Studio (nur wenn das Tool auf demselben Rechner läuft). Texte werden vor dem Versand
gekürzt, die Berechnung startet nur per Button, und die fertigen Embeddings kannst du als CSV sichern und
beim nächsten Mal ohne API-Kosten hochladen.

**Threshold kalibrieren:** Welche Cosinus-Werte „gut“ sind, hängt vom Modell ab. OpenAI text-embedding-3
liefert für echte Übersetzungen meist 0,85 und mehr, BGE-M3 oft nur 0,65–0,80. Setze den Threshold beim ersten
Lauf niedrig (z. B. 0,60), schau dir die Score-Verteilung in der Mapping-Tabelle an und ziehe ihn dann so weit
hoch, dass die klaren Fehlzuordnungen verschwinden.

**Wichtig für gute Ergebnisse:**
- Die Embeddings müssen von einem **multilingualen Modell** stammen (z. B. OpenAI text-embedding-3).
  Rein englische Modelle liefern über Sprachgrenzen hinweg keine brauchbaren Ähnlichkeiten.
- Alle Dateien müssen mit **demselben Modell** erzeugt sein (gleiche Dimension).
- **Excel schneidet Zellen bei 32.767 Zeichen ab.** Bei 1536 Dimensionen ist das knapp, bei 3072 werden
  Vektoren zerstört. Nutze CSV. Abgeschnittene Vektoren werden erkannt und verworfen, nicht repariert.
- **Leere Embeddings** entstehen, wenn Screaming Frog die Anfrage nicht ausführen konnte, etwa weil der
  Seitentext das Kontextfenster des Modells sprengt („input length exceeds the context length“). Der
  Grund aus der Spalte „Prompt Request Status“ steht bei den verworfenen Zeilen. Abhilfe: in Screaming Frog
  den übergebenen Text kürzen (z. B. nur Main Content oder die ersten Zeichen) und diese URLs neu crawlen.

**So funktioniert die Zuordnung:** Für jede Pivot-URL werden die fünf ähnlichsten URLs je Sprache über dem
Threshold gesammelt. Alle Kandidatenpaare werden nach Score sortiert und von oben abgearbeitet. Ein Paar wird
übernommen, wenn beide URLs noch frei sind. So bekommt jede URL höchstens einen Partner je Sprache (1:1).

**Konfidenz:** Ein Treffer ist *sicher*, wenn er deutlich über dem Threshold liegt (außerhalb des
Review-Bands), *reziprok* ist (beide URLs sind gegenseitig der beste Treffer) und der zweitbeste Kandidat
mindestens um die Margin schlechter ist. Sonst *prüfen*, mit Angabe des Grunds und des zweitbesten Kandidaten.
Bei „nicht reziprok“ nennt die Spalte *Bevorzugte Pivot-URL*, welche URL der Pivot-Sprache die zugeordnete
Seite eigentlich am ähnlichsten findet. So siehst du direkt die konkurrierende Ausgangs-URL.

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
FILE_TYPES = ["csv", "xlsx", "xlsm"]
EMB_IN_FILE = "Embeddings sind in der Datei enthalten"
EMB_FROM_TEXT = "Embeddings aus Textspalten berechnen"
EMB_FROM_PAGES = "Embeddings aus den Seiteninhalten berechnen (Trafilatura)"
FETCH_WARN_URLS = 500


# ============================================================
# Gecachte Helfer
# ============================================================
# cache_resource gibt dasselbe DataFrame-Objekt ohne Kopie zurück. Es darf deshalb nie verändert
# werden; Teilmengen immer über .iloc[...] (Kopie) bilden.
@st.cache_resource(max_entries=8, show_spinner=False)
def read_cached(name: str, raw: bytes) -> pd.DataFrame:
    return io_utils.read_any_file(name, raw)


@st.cache_data(max_entries=16, show_spinner=False)
def build_cached(
    name: str,
    raw: bytes,
    code: str,
    label: str,
    url_col: str,
    emb_col: str,
    filter_idx: bool,
    status_col: str | None,
    index_col: str | None,
    row_idx: tuple[int, ...] | None,
    note_col: str | None = None,
) -> LanguageSet:
    df = read_cached(name, raw)   # Cache-Treffer, kein erneutes Parsen
    if row_idx is not None:
        df = df.iloc[list(row_idx)]
    return io_utils.build_language_set(df, code, label, url_col, emb_col, filter_idx, status_col, index_col,
                                       note_col=note_col)


@st.cache_data(max_entries=32, show_spinner=False)
def text_stats_cached(
    name: str, raw: bytes, row_idx: tuple[int, ...] | None, cols: tuple, max_chars: int
) -> tuple[int, int, int]:
    """(Anzahl Texte, Ø Zeichen, geschätzte Tokens) nach Kürzung, ohne Neuberechnung bei jedem Rerun."""
    df = read_cached(name, raw)
    if row_idx is not None:
        df = df.iloc[list(row_idx)]
    texts, _ = textprep.build_texts(df, list(cols), max_chars)
    avg = int(sum(len(t) for t in texts.values()) / max(len(texts), 1))
    return len(texts), avg, textprep.estimate_tokens(texts.values())


def warn_bad_lines(name: str, df: pd.DataFrame) -> None:
    n = df.attrs.get("bad_lines", 0)
    if n > 0:
        st.warning(f"{n} fehlerhafte Zeile(n) in {name} konnten nicht gelesen werden und wurden übersprungen.")


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
            if p["models"] and not p["local"]:
                model = st.selectbox("Modell", p["models"], key=f"emb_model_{preset}")
            else:
                model = st.text_input("Modell", value=p["models"][0] if p["models"] else "", key=f"emb_model_{preset}",
                                      help="Modellname laut Endpunkt, z. B. bge-m3")
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
        keep, _ = io_utils.row_filter_reasons(df, p["url_col"], filter_idx, p["status_col"], p["index_col"])
        errors: dict[int, str] = {}
        if emb_source == EMB_FROM_TEXT:
            texts, errs = textprep.build_texts(df, p["text_cols"], cfg.max_chars)
            texts = {i: t for i, t in texts.items() if i in keep}
            errors.update({i: r for i, r in errs.items() if i in keep})
        else:
            bar.progress(0.0, text="Seiten werden geladen …")
            # Bereits geladene Seiten (auch aus einem abgebrochenen Lauf) kommen aus dem Session-Cache.
            fr = fetch.fetch_main_texts(
                {i: keep[i] for i in keep}, concurrency=5, timeout=10.0,
                on_progress=lambda d, t: bar.progress(0.5 * d / max(t, 1), text=f"Seiten geladen: {d}/{t}"),
                cache=st.session_state.setdefault("fetch_cache", {}),
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


def _clean(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


# ============================================================
# 1. Upload
# ============================================================
st.subheader("1. Crawl-Dateien hochladen")
mode = st.radio("Wie liegen deine Crawl-Daten vor?", [MODE_MULTI, MODE_SINGLE], horizontal=True)
emb_source = st.radio(
    "Embedding-Quelle",
    [EMB_IN_FILE, EMB_FROM_TEXT, EMB_FROM_PAGES],
    index=0,
    help="Standard: Embeddings stehen bereits in der Datei. Alternativ berechnet das Tool sie aus Textspalten "
         "oder lädt die Seiten selbst und extrahiert den Haupttext.",
)

sources: list[dict] = []

if mode == MODE_MULTI:
    files = st.file_uploader(
        "Eine Datei je Sprachvariante (CSV oder Excel), Mehrfachauswahl möglich",
        type=FILE_TYPES,
        accept_multiple_files=True,
        help="Mindestens eine URL-Spalte; bei Quelle „in der Datei enthalten“ zusätzlich eine Embedding-Spalte. Weitere Spalten werden ignoriert.",
    )
    for f in files or []:
        raw = f.getvalue()
        try:
            df = read_cached(f.name, raw)
        except ValueError as e:
            st.error(f"{f.name}: {e}")
            continue
        warn_bad_lines(f.name, df)
        sources.append({
            "name": f.name, "raw": raw, "df": df, "row_idx": None, "label": f.name, "code_default": None,
            "key": source_key(MODE_FILE, getattr(f, "file_id", None), raw, None),
        })
        sources[-1]["file_key"] = sources[-1]["key"]
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
            warn_bad_lines(f.name, df_all)
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
                default_rows, num_rows="dynamic", width="stretch",
                disabled=["Beispiel", "URLs"], key="pattern_editor",
            )
            st.caption(
                "Tipp: Eine letzte Zeile mit Muster `.` und dem Code der Standardsprache fängt alle übrigen URLs auf "
                "(z. B. Root-URLs ohne Sprachpräfix)."
            )
            patterns = [(_clean(r["Muster"]), _clean(r["hreflang-Code"])) for _, r in edited.iterrows()]
            groups, rest, invalid = lang_detect.split_by_patterns(urls_all, patterns)
            for p in invalid:
                st.warning(f"Ungültiges Regex-Muster wird ignoriert: `{p}`")
            if rest:
                st.warning(f"{len(rest)} URLs passen auf kein Muster und werden nicht gematcht.")
                with st.expander("URLs ohne Muster anzeigen"):
                    st.dataframe(pd.DataFrame({"URL": [urls_all[i] for i in rest]}), width="stretch", hide_index=True)
            file_key_all = getattr(f, "file_id", None) or hashlib.md5(raw).hexdigest()
            for code, idxs in groups.items():
                sources.append({
                    "name": f.name, "raw": raw, "df": df_all.iloc[idxs], "row_idx": tuple(idxs),
                    "label": f"{f.name} · {code}", "code_default": code,
                    "key": source_key(MODE_GROUP, None, raw, code),
                    # Der Gruppen-Key enthält keine Datei-Identität; sonst bliebe ein Dateitausch unbemerkt.
                    "file_key": file_key_all,
                })

if not sources:
    st.info("Bitte lade deine Crawl-Dateien hoch, um fortzufahren.")
    st.stop()

filter_idx = st.checkbox(
    "Nur URLs mit Status 200 und Indexable / Indexierbar ins Matching aufnehmen",
    value=True,
    help="Nutzt die Spalten 'Status Code'/'Statuscode' und 'Indexability'/'Indexierbarkeit', falls vorhanden.",
)
cfg: EmbeddingConfig | None = None
if emb_source != EMB_IN_FILE:
    cfg = render_embedding_settings()

# ============================================================
# 2. Sprachzuordnung
# ============================================================
st.subheader("2. Sprachzuordnung und Spaltenerkennung")
sets: list[LanguageSet] = []
set_meta: list[dict] = []   # je Sprachvariante: Werte für die Ergebnis-Signatur
invalid_codes = 0
pending: list[dict] = []

for src in sources:
    df = src["df"]
    key = src["key"]
    with st.container(border=True):
        st.markdown(f"**{src['label']}** – {len(df)} Zeilen")
        cols = list(df.columns)
        url_guess = io_utils.detect_url_column(df)
        emb_guess = io_utils.detect_embedding_column(df)
        status_col = io_utils.detect_status_column(df)
        index_col = io_utils.detect_indexability_column(df)

        c1, c2, c3 = st.columns([1, 2, 2])
        with c1:
            if src["code_default"] is not None:
                # Modus Gesamtdatei: einzige Quelle für den Code ist die Mustertabelle.
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
        n_tokens = 0
        if emb_source == EMB_FROM_TEXT:
            if not text_cols:
                st.warning("Bitte mindestens eine Textspalte wählen.")
            else:
                n_texts, avg, n_tokens = text_stats_cached(src["name"], src["raw"], src["row_idx"], tuple(text_cols),
                                                           cfg.max_chars)
                st.caption(f"{n_texts} Texte, Ø {avg} Zeichen, ca. {n_tokens} Tokens")
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
                            "status_col": status_col, "index_col": index_col, "key": key, "file_key": src["file_key"],
                            "row_idx": src["row_idx"], "tokens": n_tokens})

if emb_source != EMB_IN_FILE and invalid_codes == 0:
    if emb_source == EMB_FROM_TEXT and any(not p["text_cols"] for p in pending):
        st.stop()
    if len(pending) < 2:
        st.error("Es werden mindestens zwei Sprachvarianten benötigt.")
        st.stop()
    pending_codes = [p["code"] for p in pending]
    pending_dupes = sorted({c for c in pending_codes if pending_codes.count(c) > 1})
    if pending_dupes:
        st.error(f"Doppelte hreflang-Codes: {', '.join(pending_dupes)}. Jede Sprachvariante braucht einen eigenen Code.")
        st.stop()
    emb_sig = (emb_source, cfg.base_url, cfg.model, cfg.max_chars, filter_idx,
               tuple((p["key"], p["file_key"], p["code"], tuple(p["text_cols"]), p["url_col"], p["row_idx"])
                     for p in pending))
    n_rows = sum(len(p["df"]) for p in pending)
    if emb_source == EMB_FROM_TEXT:
        total_tokens = sum(p["tokens"] for p in pending)
        summary = f"{n_rows} Zeilen in {len(pending)} Sprachvariante(n), ca. {total_tokens} Tokens nach Kürzung."
    else:
        summary = f"{n_rows} Zeilen in {len(pending)} Sprachvariante(n)."
    prices = {"text-embedding-3-small": "ca. 0,02 $ je 1 Mio. Tokens", "text-embedding-3-large": "ca. 0,13 $ je 1 Mio. Tokens"}
    is_openai = st.session_state.get("emb_preset", "OpenAI") == "OpenAI"
    hint = f" {cfg.model} kostet {prices[cfg.model]}." if is_openai and cfg.model in prices else ""
    st.caption(f"{summary} Die Berechnung startet nur über den Button.{hint}")
    if st.button("Embeddings berechnen", type="primary", key="emb_compute"):
        if embed.PRESETS[st.session_state.get("emb_preset", "OpenAI")]["needs_key"] and not cfg.api_key:
            st.error("Für dieses Preset wird ein API-Key benötigt.")
        elif not cfg.base_url or not cfg.model:
            st.error("Base-URL und Modell müssen gesetzt sein.")
        else:
            try:
                with st.spinner("Embeddings werden berechnet …"):
                    new_sets = compute_language_sets(pending, emb_source, cfg, filter_idx)
                    # CSV-Payloads einmalig bei der Berechnung erzeugen, nicht bei jedem Rerun.
                    st.session_state["computed"] = {"sets": new_sets, "sig": emb_sig,
                                                    "csv": [io_utils.embeddings_csv(ls) for ls in new_sets]}
                st.session_state.pop("result", None)
            except embed.EmbeddingError as e:
                st.session_state.pop("computed", None)
                st.error(f"Verbindung zum Endpunkt fehlgeschlagen ({cfg.base_url}): {e}")
    comp = st.session_state.get("computed")
    if comp is not None and any(k not in comp for k in ("sets", "sig", "csv")):   # ältere Codeversion
        st.session_state.pop("computed", None)
        comp = None
    if not comp:
        st.info("Bitte auf „Embeddings berechnen“ klicken, um fortzufahren.")
        st.stop()
    if comp["sig"] != emb_sig:
        st.warning("Die Embeddings sind veraltet (Spalten, Dateien oder Einstellungen haben sich geändert). "
                   "Bitte erneut auf „Embeddings berechnen“ klicken.")
        st.stop()
    for ls, p, csv_bytes in zip(comp["sets"], pending, comp["csv"]):
        with st.container(border=True):
            st.markdown(f"**{p['label']}** – berechnet mit {cfg.model}")
            render_set_metrics(ls)
            st.download_button(f"📥 Embeddings als CSV ({ls.code})", csv_bytes,
                               f"embeddings_{ls.code}.csv", "text/csv", key=f"dl_emb_{p['key']}", on_click="ignore",
                               help="Im Modus „Embeddings sind in der Datei enthalten“ direkt wieder verwendbar.")
        sets.append(ls)
        set_meta.append({"key": p["key"], "file_key": p["file_key"], "code": ls.code, "len": len(ls), "dim": ls.dim,
                         "url_col": p["url_col"], "emb_col": f"berechnet:{cfg.model}"})

codes = [s.code for s in sets]
problems: list[str] = []
if invalid_codes > 0:
    problems.append(f"{invalid_codes} Sprachvariante(n) haben einen ungültigen hreflang-Code.")
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
signature = (
    tuple((m["key"], m["file_key"], m["code"], m["len"], m["dim"], m["url_col"], m["emb_col"]) for m in set_meta),
    filter_idx, pivot_code, threshold, band, min_margin, use_slug, code_mode, x_default_code, include_review,
)

if st.button("Let's Go", type="primary"):
    try:
        with st.spinner("Matching läuft …"):
            result = matching.run_matching(sets, pivot_code, threshold, band, min_margin, use_slug)
            clusters = output.build_clusters(result, sets)
            codes_sorted = [pivot_code] + sorted(c for c in codes if c != pivot_code)
            mapping = output.mapping_table(clusters, codes_sorted, pivot_code, x_default_code)
            unmatched = output.unmatched_table(result, sets)
            html_out = output.html_blocks(clusters, pivot_code, x_default_code, code_mode, include_review)
            st.session_state["result"] = {
                "mapping": mapping,
                "unmatched": unmatched,
                "html": html_out,
                # Download-Payloads einmalig beim Lauf erzeugen, nicht bei jedem Rerun.
                "mapping_csv": mapping.to_csv(index=False).encode("utf-8-sig"),
                "lean_csv": output.lean_mapping_table(clusters, codes_sorted, pivot_code)
                .to_csv(index=False).encode("utf-8-sig"),
                "unmatched_csv": unmatched.to_csv(index=False).encode("utf-8-sig"),
                "html_bytes": html_out.encode("utf-8"),
                "signature": signature,
                "stats": result.stats,
                "pivot": pivot_code,
                "codes": codes_sorted,
                "n_clusters": sum(1 for c in clusters if len(c.members) > 1),
                "n_review": sum(1 for c in clusters if len(c.members) > 1 and c.confidence == "prüfen"),
            }
    except ValueError as e:
        st.session_state.pop("result", None)
        st.error(str(e))

res = st.session_state.get("result")
# Ergebnisse einer älteren Codeversion (z. B. nach Hot Reload) können Schlüssel vermissen: verwerfen.
RESULT_KEYS = ("mapping", "unmatched", "html", "mapping_csv", "lean_csv", "unmatched_csv", "html_bytes",
               "signature", "stats", "pivot", "codes", "n_clusters", "n_review")
if res is not None and any(k not in res for k in RESULT_KEYS):
    st.session_state.pop("result", None)
    res = None
if res:
    st.subheader("4. Ergebnisse")
    if res.get("signature") != signature:
        st.warning("Die Einstellungen oder Dateien haben sich seit dem letzten Lauf geändert. Bitte erneut auf „Let's Go“ klicken.")
    mcols = st.columns(len(res["codes"]))
    for col, code in zip(mcols, res["codes"]):
        s = res["stats"].get(code, {})
        if code == res["pivot"]:
            col.metric(f"{code} (Pivot)", f"{s.get('total', 0) - s.get('unmatched', 0)} / {s.get('total', 0)} zugeordnet")
        else:
            col.metric(code, f"{s.get('slug', 0) + s.get('embedding', 0)} Treffer")
            col.caption(f"{s.get('prüfen', 0)} prüfen")
    st.caption(f"{res['n_clusters']} Cluster, davon {res['n_review']} mit Konfidenz „prüfen“.")

    st.markdown("#### Zuordnung (Mapping)")
    st.dataframe(res["mapping"], width="stretch", hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("📥 Mapping als CSV herunterladen", res["mapping_csv"],
                       "hreflang_mapping.csv", "text/csv", key="dl_map", on_click="ignore",
                       help="Alle Spalten: Methode, Konfidenz, Grund, zweitbester Kandidat, bevorzugte Pivot-URL.")
    d2.download_button("📥 Kompaktes Mapping als CSV (nur URLs + Score)", res["lean_csv"],
                       "hreflang_mapping_kompakt.csv", "text/csv", key="dl_lean", on_click="ignore",
                       help="Nur Pivot-URL sowie je Sprache die zugeordnete URL und der Cosinus-Score.")

    st.markdown("#### URLs ohne Zuordnung")
    if res["unmatched"].empty:
        st.success("Alle URLs wurden zugeordnet.")
    else:
        st.dataframe(res["unmatched"], width="stretch", hide_index=True)
        st.download_button("📥 URLs ohne Zuordnung als CSV herunterladen", res["unmatched_csv"],
                           "hreflang_unmatched.csv", "text/csv", key="dl_unmatched", on_click="ignore")

    st.markdown("#### hreflang-HTML")
    if not res["html"]:
        st.info("Keine Cluster für den HTML-Export. Prüfe Threshold oder aktiviere „prüfen“-Cluster im Export.")
    else:
        blocks = res["html"].strip().split("\n\n")
        st.code("\n\n".join(blocks[:3]), language="html")
        st.caption(f"Vorschau der ersten {min(3, len(blocks))} von {len(blocks)} Blöcken. Die Datei enthält alle.")
        st.download_button("📥 hreflang-Tags als HTML-Datei herunterladen", res["html_bytes"],
                           "hreflang_tags.html", "text/html", key="dl_html", on_click="ignore")
