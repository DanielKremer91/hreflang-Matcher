import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

# AppTest kann keine Dateien hochladen. Der Wrapper ersetzt st.file_uploader (Mehrfach-Upload)
# durch eine Liste gefälschter Upload-Objekte aus st.session_state["_fake_files"].
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
    return None
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
