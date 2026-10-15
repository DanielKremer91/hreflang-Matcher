import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from hreflang_matcher import embed
from hreflang_matcher.models import EmbeddingConfig


def _vec_for(text: str, dim: int = 4) -> list[float]:
    h = sum(ord(c) for c in text)
    return [float((h + i) % 7) for i in range(dim)]


class _Handler(BaseHTTPRequestHandler):
    # Pro Serverinstanz: Liste geplanter Antworten [(status, body_dict | None)]; None = normale Antwort.
    script: list = []
    requests_seen: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests_seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        status, payload = (200, None)
        if type(self).script:
            status, payload = type(self).script.pop(0)
        if payload is None and status == 200:
            payload = {"data": [{"index": i, "embedding": _vec_for(t)} for i, t in enumerate(body["input"])]}
        if payload is None:
            payload = {"error": {"message": "boom"}}
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Handler.script = []
    _Handler.requests_seen = []
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    yield srv, _Handler
    srv.shutdown()
    srv.server_close()
    t.join()


def _cfg(srv, key="", **kw):
    return EmbeddingConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="m", api_key=key, **kw)


def test_presets_exact():
    assert list(embed.PRESETS) == ["OpenAI", "Ollama (lokal)", "LM Studio (lokal)", "Eigener OpenAI-kompatibler Endpunkt"]
    assert embed.PRESETS["OpenAI"]["base_url"] == "https://api.openai.com/v1"
    assert embed.PRESETS["OpenAI"]["models"][0] == "text-embedding-3-small"
    assert embed.PRESETS["Ollama (lokal)"]["local"] is True and embed.PRESETS["Ollama (lokal)"]["models"] == ["bge-m3"]


def test_cache_key_depends_on_model_url_text():
    c1 = EmbeddingConfig("http://x/v1", "a")
    c2 = EmbeddingConfig("http://x/v1", "b")
    assert embed.cache_key(c1, "t") != embed.cache_key(c2, "t")
    assert embed.cache_key(c1, "t") == embed.cache_key(EmbeddingConfig("http://x/v1", "a", api_key="k"), "t")


def test_embed_batch_posts_to_embeddings_with_bearer(server):
    srv, h = server
    out = embed.embed_batch(_cfg(srv, key="sk-1"), ["hallo", "welt"])
    assert [v.tolist() for v in out] == [_vec_for("hallo"), _vec_for("welt")]
    assert out[0].dtype == np.float32
    req = h.requests_seen[0]
    assert req["path"] == "/v1/embeddings" and req["auth"] == "Bearer sk-1"
    assert req["body"] == {"model": "m", "input": ["hallo", "welt"]}


def test_embed_batch_without_key_sends_no_auth_header(server):
    srv, h = server
    embed.embed_batch(_cfg(srv), ["x"])
    assert h.requests_seen[0]["auth"] is None


def test_retry_on_429_then_500_then_ok(server):
    srv, h = server
    h.script[:] = [(429, None), (500, None)]
    sleeps = []
    out = embed.embed_batch(_cfg(srv), ["x"], sleep=sleeps.append)
    assert len(out) == 1 and sleeps == [2, 4] and len(h.requests_seen) == 3


def test_gives_up_after_three_retries(server):
    srv, h = server
    h.script[:] = [(503, None)] * 4
    with pytest.raises(embed.EmbeddingError) as e:
        embed.embed_batch(_cfg(srv), ["x"], sleep=lambda s: None)
    assert "HTTP 503" in str(e.value) and len(h.requests_seen) == 4


def test_401_is_not_retried(server):
    srv, h = server
    h.script[:] = [(401, {"error": {"message": "Incorrect API key"}})]
    with pytest.raises(embed.EmbeddingError) as e:
        embed.embed_batch(_cfg(srv), ["x"], sleep=lambda s: None)
    assert str(e.value).startswith("HTTP 401") and "Incorrect API key" in str(e.value)
    assert len(h.requests_seen) == 1


def test_400_length_error_retries_once_with_half_max_chars(server):
    srv, h = server
    h.script[:] = [(400, {"error": {"message": "This model's maximum context length is 8192 tokens"}})]
    long = "a" * 1000
    out = embed.embed_batch(_cfg(srv, max_chars=1000), [long], sleep=lambda s: None)
    assert len(out) == 1
    assert len(h.requests_seen) == 2 and len(h.requests_seen[1]["body"]["input"][0]) == 500


def test_embed_texts_batches_cache_and_errors(server):
    srv, h = server
    cfg = _cfg(srv, batch_size=2)
    cache: dict = {}
    texts = {0: "a", 1: "b", 2: "c"}
    progress = []
    res = embed.embed_texts(texts, cfg, cache=cache, on_progress=lambda d, t: progress.append((d, t)), max_workers=1)
    assert set(res.vectors) == {0, 1, 2} and res.errors == {} and res.dim == 4
    assert len(h.requests_seen) == 2 and progress[-1] == (3, 3)
    assert len(cache) == 3
    # zweiter Lauf: alles aus dem Cache, keine Anfrage
    res2 = embed.embed_texts(texts, cfg, cache=cache, max_workers=1)
    assert len(h.requests_seen) == 2 and res2.vectors[1].tolist() == _vec_for("b")


def test_embed_texts_reports_batch_failure_per_row(server):
    srv, h = server
    h.script[:] = [(401, {"error": {"message": "bad key"}})]
    res = embed.embed_texts({0: "a", 1: "b"}, _cfg(srv, batch_size=2), max_workers=1, sleep=lambda s: None)
    assert res.vectors == {} and res.dim == 0
    assert res.errors == {0: "Embedding-Anfrage fehlgeschlagen: HTTP 401 bad key",
                          1: "Embedding-Anfrage fehlgeschlagen: HTTP 401 bad key"}


def test_cache_fifo_limit(monkeypatch, server):
    srv, _ = server
    monkeypatch.setattr(embed, "CACHE_LIMIT", 3)
    cache: dict = {}
    embed.embed_texts({i: f"t{i}" for i in range(5)}, _cfg(srv, batch_size=5), cache=cache, max_workers=1)
    assert len(cache) == 3 and embed.cache_key(_cfg(srv), "t4") in cache and embed.cache_key(_cfg(srv), "t0") not in cache


def test_connection_returns_dim(server):
    srv, _ = server
    assert embed.test_connection(_cfg(srv)) == 4


def test_connection_error_is_embedding_error():
    cfg = EmbeddingConfig(base_url="http://127.0.0.1:9/v1", model="m", timeout=1)
    with pytest.raises(embed.EmbeddingError) as e:
        embed.embed_batch(cfg, ["x"], sleep=lambda s: None)
    assert "Verbindungsfehler" in str(e.value)
