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
FILE_TYPES = ["csv", "xlsx", "xlsm"]


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
    try:
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
    except ValueError as e:
        st.session_state.pop("result", None)
        st.error(str(e))

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
