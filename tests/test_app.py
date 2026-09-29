from streamlit.testing.v1 import AppTest


def test_app_renders_without_exception():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("ONE hreflang Matcher" in t.value for t in at.title)
    # Ohne Upload endet die App mit dem Hinweis zum Hochladen
    assert any("crawl-dateien" in i.value.lower() for i in at.info)
