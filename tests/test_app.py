import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

# AppTest kann keine Dateien hochladen. Der Wrapper ersetzt st.file_uploader durch gefälschte Upload-Objekte:
# Mehrfach-Upload aus st.session_state["_fake_files"] (Liste), Einzel-Upload aus ["_fake_single"] (Tupel).
_WRAPPER = f'''
import streamlit as st

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
with open({APP!r}, encoding="utf-8") as _fh:
    _src = _fh.read()
try:
    exec(compile(_src, {APP!r}, "exec"))
finally:
    st.file_uploader = _orig_uploader
'''


def _csv(lang: str) -> bytes:
    rows = ["Address;Embedding"]
    for i in range(3):
        vec = [0.0] * 8
        vec[i] = 1.0
        rows.append(f'https://a.com/{lang}/p{i}/;"{vec}"')
    return "\n".join(rows).encode("utf-8")


def test_app_renders_without_exception():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("ONE hreflang Matcher" in t.value for t in at.title)
    # Ohne Upload endet die App mit dem Hinweis zum Hochladen
    assert any("crawl-dateien" in i.value.lower() for i in at.info)


def test_codes_stay_with_their_file_after_removal():
    files = [(f"{lang}.csv", _csv(lang), f"id-{lang}") for lang in ("my", "de", "fr")]
    at = AppTest.from_string(_WRAPPER, default_timeout=60)
    at.session_state["_fake_files"] = files
    at.run()
    assert not at.exception, at.exception
    assert [t.value for t in at.text_input if t.label == "hreflang-Code"] == ["my", "de", "fr"]

    at.session_state["_fake_files"] = files[1:]   # erste Datei entfernt
    at.run()
    assert not at.exception, at.exception
    assert [t.value for t in at.text_input if t.label == "hreflang-Code"] == ["de", "fr"]


def test_bad_csv_lines_are_reported():
    bad = _csv("de") + b'\nhttps://a.com/de/kaputt;"[1.0]";zu viel'
    at = AppTest.from_string(_WRAPPER, default_timeout=60)
    at.session_state["_fake_files"] = [("de.csv", bad, "id-de"), ("fr.csv", _csv("fr"), "id-fr")]
    at.run()
    assert not at.exception, at.exception
    assert any("1 fehlerhafte Zeile(n) in de.csv" in w.value for w in at.warning)


def test_full_run_shows_results_and_downloads():
    at = AppTest.from_string(_WRAPPER, default_timeout=60)
    at.session_state["_fake_files"] = [("de.csv", _csv("de"), "id-de"), ("fr.csv", _csv("fr"), "id-fr")]
    at.run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Let's Go").click().run()
    assert not at.exception, at.exception
    assert any(h.value == "4. Ergebnisse" for h in at.subheader)
    res = at.session_state["result"]
    assert res["mapping_csv"].startswith("﻿".encode("utf-8"))
    assert b"https://a.com/fr/p0/" in res["mapping_csv"]
    assert res["html_bytes"].count(b'hreflang="x-default"') == 6   # 3 Cluster x 2 Mitglieder
    lean = res["lean_csv"].decode("utf-8-sig").splitlines()
    assert lean[0] == "Pivot-URL (de),URL (fr),Score (fr)"
    assert b"https://a.com/fr/p0/" in res["lean_csv"]
    assert not any("geändert" in w.value for w in at.warning)


def test_changed_filter_marks_result_as_stale():
    at = AppTest.from_string(_WRAPPER, default_timeout=60)
    at.session_state["_fake_files"] = [("de.csv", _csv("de"), "id-de"), ("fr.csv", _csv("fr"), "id-fr")]
    at.run()
    next(b for b in at.button if b.label == "Let's Go").click().run()
    assert not any("geändert" in w.value for w in at.warning)
    # Ohne Status-/Indexierbarkeitsspalten ändert der Filter die URL-Zahl nicht, der Lauf ist trotzdem veraltet.
    next(c for c in at.checkbox if c.label.startswith("Nur URLs mit Status 200")).uncheck().run()
    assert not at.exception, at.exception
    assert any("geändert" in w.value for w in at.warning)


def _single_csv(shift: int) -> bytes:
    """Gesamtdatei mit /de/- und /fr/-URLs; `shift` ändert nur die Embedding-Werte, nicht URLs oder Anzahl."""
    rows = ["Address;Embedding"]
    for lang in ("de", "fr"):
        for i in range(3):
            vec = [0.0] * 8
            vec[(i + shift) % 8] = 1.0
            rows.append(f'https://a.com/{lang}/p{i}/;"{vec}"')
    return "\n".join(rows).encode("utf-8")


def test_mode_b_file_swap_marks_result_as_stale():
    at = AppTest.from_string(_WRAPPER, default_timeout=60)
    at.session_state["_fake_single"] = ("all.csv", _single_csv(0), "id-1")
    at.run()
    at.radio[0].set_value("Eine Gesamtdatei mit allen Sprachvarianten").run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Let's Go").click().run()
    assert not at.exception, at.exception
    assert "result" in at.session_state
    assert not any("seit dem letzten Lauf" in w.value for w in at.warning)

    # Gleiche URLs, gleiche Anzahl, andere Embedding-Werte und andere Datei-ID.
    at.session_state["_fake_single"] = ("all.csv", _single_csv(1), "id-2")
    at.run()
    assert not at.exception, at.exception
    assert any("seit dem letzten Lauf" in w.value for w in at.warning)


def test_stale_result_dict_is_discarded():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["result"] = {"mapping": None}
    at.run()
    assert not at.exception, at.exception
    # Ohne Upload stoppt die App vor dem Ergebnisbereich; das Ergebnis wird dort nie gelesen.
    assert any("crawl-dateien" in i.value.lower() for i in at.info)


def test_stale_result_dict_is_discarded_with_files():
    at = AppTest.from_string(_WRAPPER, default_timeout=60)
    at.session_state["_fake_files"] = [("de.csv", _csv("de"), "id-de"), ("fr.csv", _csv("fr"), "id-fr")]
    at.session_state["result"] = {"mapping": None}
    at.run()
    assert not at.exception, at.exception
    assert "result" not in at.session_state
    assert not any(h.value == "4. Ergebnisse" for h in at.subheader)
