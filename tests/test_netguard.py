import socket

import pytest

from hreflang_matcher import netguard

INTERNAL = "interne oder lokale Adresse nicht erlaubt"


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data",   # Cloud-Metadaten
    "http://10.0.0.1/",
    "http://192.168.1.1/",
    "http://172.16.0.1/",
    "http://[fe80::1]/",
    "http://0.0.0.0/",
    "http://224.0.0.1/",
    "http://[::ffff:10.0.0.1]/",
    "http://100.64.0.1/",
])
def test_internal_addresses_rejected(url):
    assert netguard.check_url(url) == INTERNAL
    assert netguard.check_url(url, allow_loopback=True) == INTERNAL


def test_non_http_scheme_rejected():
    assert netguard.check_url("ftp://x") == "nur http/https erlaubt"
    assert netguard.check_url("file:///etc/passwd") == "nur http/https erlaubt"
    assert netguard.check_url("") == "nur http/https erlaubt"


def test_missing_host_rejected():
    assert netguard.check_url("http:///pfad") == "Host fehlt"


def test_loopback_only_with_allow_loopback():
    assert netguard.check_url("http://127.0.0.1:1/", allow_loopback=True) is None
    assert netguard.check_url("http://127.0.0.1:1/") == INTERNAL
    assert netguard.check_url("http://[::1]:1/", allow_loopback=True) is None
    assert netguard.check_url("http://[::1]:1/") == INTERNAL


def test_unresolvable_host(monkeypatch):
    def fail(*a, **k):
        raise socket.gaierror("nope")
    monkeypatch.setattr(netguard.socket, "getaddrinfo", fail)
    assert netguard.check_url("https://gibt-es-nicht.invalid/") == "Host nicht auflösbar"


def test_host_resolving_to_internal_address_rejected(monkeypatch):
    monkeypatch.setattr(netguard.socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80)),
                                         (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 80))])
    assert netguard.check_url("https://rebind.example/") == INTERNAL


def test_public_host_allowed():
    try:
        socket.getaddrinfo("example.com", 443)
    except socket.gaierror:
        pytest.skip("DNS nicht verfügbar")
    assert netguard.check_url("https://example.com/") is None
