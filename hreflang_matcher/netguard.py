"""Schutz vor SSRF: nur http(s) zu öffentlich routbaren Adressen (Loopback optional)."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

BAD_SCHEME = "nur http/https erlaubt"
NO_HOST = "Host fehlt"
INTERNAL = "interne oder lokale Adresse nicht erlaubt"
UNRESOLVABLE = "Host nicht auflösbar"
BAD_URL = "ungültige URL"


_THIS_NETWORK = ipaddress.ip_network("0.0.0.0/8")


def _is_forbidden(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, allow_loopback: bool,
                  allow_private: bool) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    if ip.is_loopback:
        return not allow_loopback
    # Link-Local (169.254/16 mit Cloud-Metadaten, fe80::/10), Multicast, unspezifiziert, reserviert und
    # 0.0.0.0/8 sind immer gesperrt, auch mit allow_private.
    if (ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved
            or (ip.version == 4 and ip in _THIS_NETWORK)):
        return True
    if allow_private and ip.is_private:   # RFC 1918, ULA fc00::/7
        return False
    # is_global deckt zusätzlich private Netze und Shared Address Space (100.64/10) ab.
    return not ip.is_global or ip.is_private


def check_url(url: str, allow_loopback: bool = False, allow_private: bool = False) -> str | None:
    """Grund (deutsch), warum die URL nicht abgerufen werden darf, sonst None.

    Alle Adressen, auf die der Host auflöst, müssen öffentlich sein. Loopback nur mit `allow_loopback`,
    private Netze (RFC 1918, ULA) nur mit `allow_private`; Link-Local/Metadaten bleiben immer gesperrt."""
    try:
        parts = urlsplit(str(url or "").strip())
        port = parts.port
    except ValueError:
        return BAD_URL
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return BAD_SCHEME
    host = parts.hostname
    if not host:
        return NO_HOST
    try:
        infos = socket.getaddrinfo(host, port or (443 if scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError, ValueError):
        return UNRESOLVABLE
    if not infos:
        return UNRESOLVABLE
    for info in infos:
        addr = str(info[4][0]).split("%", 1)[0]   # IPv6-Scope-ID entfernen
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return INTERNAL
        if _is_forbidden(ip, allow_loopback, allow_private):
            return INTERNAL
    return None
