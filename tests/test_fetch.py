import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import pytest

from hreflang_matcher import fetch

ARTICLE_HTML = """<!doctype html><html lang="de"><head><title>Küche renovieren</title></head><body>
<nav><a href="/">Start</a> <a href="/kontakt">Kontakt</a></nav>
<main><article>
<h1>Küche renovieren: Wie gelingt die perfekte Modernisierung?</h1>
<p>Alte Küche, frische Optik: Ist Ihre Küche in die Jahre gekommen, muss sie nicht zwingend eine neue her.
Mit kreativen Ideen und gezielten Maßnahmen lässt sich Ihre Küche kostengünstig modernisieren.</p>
<p>Erfahren Sie, wie Sie durch die Renovierung von Fronten, Arbeitsplatten und Wänden einen völlig neuen Look
schaffen können. Wir zeigen Möglichkeiten der Küchenrenovierung, Tipps und Inspiration sowie die Kosten.</p>
<p>Küchengeräte tauschen, Wände und Böden erneuern oder nur die Fronten lackieren: Jede Maßnahme hat ihren Preis
und ihren Effekt. Planen Sie Schritt für Schritt und holen Sie sich Angebote ein.</p>
</article></main>
<footer><p>Impressum · Datenschutz</p></footer></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ok":
            body = ARTICLE_HTML.encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        elif self.path == "/latin":
            body = ARTICLE_HTML.encode("utf-8")   # UTF-8-Bytes ohne charset im Header
            self.send_response(200); self.send_header("Content-Type", "text/html")
        elif self.path == "/redirect":
            self.send_response(302); self.send_header("Location", "/ok"); body = b""
        elif self.path == "/redirect-meta":
            self.send_response(302); self.send_header("Location", "http://169.254.169.254/latest/meta-data"); body = b""
        elif self.path == "/loop":
            self.send_response(302); self.send_header("Location", "/loop"); body = b""
        elif self.path == "/big-nolength":
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>" + b"x" * 5000 + b"</body></html>")
            return
        elif self.path == "/drip":
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", "100000")
            self.end_headers()
            try:
                for _ in range(30):
                    self.wfile.write(b"<p>x</p>"); self.wfile.flush(); time.sleep(0.1)
            except OSError:
                pass
            return
        elif self.path == "/pdf":
            body = b"%PDF-1.4"
            self.send_response(200); self.send_header("Content-Type", "application/pdf")
        elif self.path == "/empty":
            body = b"<html><body><nav>Start Kontakt</nav></body></html>"
            self.send_response(200); self.send_header("Content-Type", "text/html")
        elif self.path == "/slow":
            time.sleep(1.5)
            body = b"<html></html>"
            self.send_response(200); self.send_header("Content-Type", "text/html")
        else:
            body = b"nope"
            self.send_response(404); self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def base():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()
    t.join()


def test_extract_main_text_drops_nav_and_footer():
    text = fetch.extract_main_text(ARTICLE_HTML, "https://example.com/de/kueche")
    assert "Küche renovieren" in text and "kostengünstig modernisieren" in text
    assert "Impressum" not in text and "Kontakt" not in text


def test_extract_main_text_empty_for_boilerplate_only():
    assert fetch.extract_main_text("<html><body><nav>Start</nav></body></html>") == ""


def test_fetch_one_ok(base):
    text, err = fetch.fetch_one(base + "/ok", allow_private=True)
    assert err is None and "Modernisierung" in text


def test_fetch_one_utf8_without_charset_header_keeps_umlauts(base):
    text, err = fetch.fetch_one(base + "/latin", allow_private=True)
    assert err is None
    assert "Küche" in text and "Maßnahmen" in text and "kostengünstig" in text and "für" in text
    assert "Ã" not in text


def test_extract_main_text_accepts_bytes():
    text = fetch.extract_main_text(ARTICLE_HTML.encode("utf-8"), "https://example.com/de/kueche")
    assert "Küche renovieren" in text and "Maßnahmen" in text


@pytest.mark.parametrize("path,reason", [
    ("/missing", "HTTP 404"), ("/pdf", "kein HTML (application/pdf)"), ("/empty", "kein Text extrahiert"),
])
def test_fetch_one_reasons(base, path, reason):
    text, err = fetch.fetch_one(base + path, allow_private=True)
    assert text == "" and err == reason


def test_fetch_one_timeout(base):
    text, err = fetch.fetch_one(base + "/slow", timeout=0.3, allow_private=True)
    assert text == "" and err == "Timeout"


def test_fetch_one_connection_error():
    text, err = fetch.fetch_one("http://127.0.0.1:9/x", timeout=0.5, allow_private=True)
    assert text == "" and err.startswith("Verbindungsfehler:")


def test_fetch_main_texts_parallel_keeps_indices(base):
    urls = {0: base + "/ok", 1: base + "/missing", 2: base + "/ok", 3: base + "/pdf"}
    progress = []
    res = fetch.fetch_main_texts(urls, concurrency=3, timeout=5, on_progress=lambda d, t: progress.append((d, t)),
                                 allow_private=True)
    assert set(res.texts) == {0, 2} and res.errors == {1: "HTTP 404", 3: "kein HTML (application/pdf)"}
    assert progress[-1] == (4, 4)


def test_fetch_one_rejects_loopback_by_default(base):
    assert fetch.fetch_one(base + "/ok") == ("", "interne oder lokale Adresse nicht erlaubt")


def test_fetch_one_rejects_non_http_scheme():
    assert fetch.fetch_one("ftp://example.com/x") == ("", "nur http/https erlaubt")


def test_fetch_one_follows_safe_redirect(base):
    text, err = fetch.fetch_one(base + "/redirect", allow_private=True)
    assert err is None and "Modernisierung" in text


def test_fetch_one_blocks_redirect_to_internal_address(base):
    assert fetch.fetch_one(base + "/redirect-meta", allow_private=True) == ("", "Weiterleitung zu unzulässiger Adresse")


def test_fetch_one_stops_redirect_loop(base):
    assert fetch.fetch_one(base + "/loop", allow_private=True) == ("", "zu viele Weiterleitungen")


def test_fetch_one_caps_body_without_content_length(base):
    text, err = fetch.fetch_one(base + "/big-nolength", allow_private=True, max_bytes=1000)
    assert text == "" and err.startswith("Seite zu groß")


def test_fetch_one_caps_body_by_content_length(base):
    text, err = fetch.fetch_one(base + "/ok", allow_private=True, max_bytes=100)
    assert text == "" and err.startswith("Seite zu groß")


def test_size_reason_default_wording():
    assert fetch._too_big(5_000_000) == "Seite zu groß (> 5 MB)"


def test_fetch_one_overall_deadline(base):
    # Jedes Stück kommt schneller als der Lese-Timeout, die Seite insgesamt aber zu langsam.
    start = time.monotonic()
    text, err = fetch.fetch_one(base + "/drip", timeout=0.3, allow_private=True)
    assert (text, err) == ("", "Timeout") and time.monotonic() - start < 2.0


def test_fetch_main_texts_rejects_loopback_by_default(base):
    res = fetch.fetch_main_texts({0: base + "/ok"}, timeout=5)
    assert res.errors == {0: "interne oder lokale Adresse nicht erlaubt"}


def test_user_agent_constant():
    assert fetch.USER_AGENT == "ONE-hreflang-Matcher/1.0 (+https://onebeyondsearch.com)"


def test_fetch_main_texts_survives_unexpected_exception(base, monkeypatch):
    real = fetch.fetch_one

    def flaky(url, **kw):
        if url.endswith("/boom"):
            raise RuntimeError("kaputt")
        return real(url, **kw)

    monkeypatch.setattr(fetch, "fetch_one", flaky)
    res = fetch.fetch_main_texts({0: base + "/ok", 1: base + "/boom", 2: base + "/ok"}, concurrency=2, timeout=5,
                                 allow_private=True)
    assert set(res.texts) == {0, 2} and res.errors == {1: "Fehler: RuntimeError"}


class _Interrupt(Exception):
    """Simuliert den Abbruch durch einen Streamlit-Rerun."""


class _SlowHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(1.0)
        body = ARTICLE_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def slow_base():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()
    t.join()


def test_fetch_interruption_returns_promptly_and_keeps_completed_texts(slow_base):
    urls = {i: f"{slow_base}/p{i}" for i in range(6)}
    cache: dict = {}
    raised_at: list[float] = []

    def on_progress(d, t):
        raised_at.append(time.monotonic())
        raise _Interrupt()

    start = time.monotonic()
    with pytest.raises(_Interrupt):
        fetch.fetch_main_texts(urls, concurrency=2, timeout=5, on_progress=on_progress, cache=cache,
                               allow_private=True)
    end = time.monotonic()
    assert end - start < 2.0, end - start
    assert end - raised_at[0] < 0.5, end - raised_at[0]
    assert len(cache) == 1
    url, text = next(iter(cache.items()))
    assert url in urls.values() and "Modernisierung" in text


def test_fetch_main_texts_uses_cache_without_request(base):
    cache = {base + "/missing": "Text aus dem Cache"}
    res = fetch.fetch_main_texts({0: base + "/missing", 1: base + "/ok"}, concurrency=2, timeout=5, cache=cache,
                                 allow_private=True)
    assert res.texts[0] == "Text aus dem Cache" and "Modernisierung" in res.texts[1] and res.errors == {}
    assert base + "/ok" in cache


def test_fetch_cache_fifo_limit(base, monkeypatch):
    monkeypatch.setattr(fetch, "FETCH_CACHE_LIMIT", 2)
    cache = {"a": "1", "b": "2"}
    fetch.fetch_main_texts({0: base + "/ok"}, timeout=5, cache=cache, allow_private=True)
    assert list(cache) == ["b", base + "/ok"]
