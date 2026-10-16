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


def _is_forbidden(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, allow_loopback: bool) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    if ip.is_loopback:
        return not allow_loopback
    # is_global deckt private Netze, Link-Local (169.254/16, fe80::/10), reservierte, unspezifizierte
    # und Shared-Address-Bereiche ab; Multicast ist teils "global" und wird extra ausgeschlossen.
    return (not ip.is_global or ip.is_private or ip.is_link_local or ip.is_reserved
            or ip.is_multicast or ip.is_unspecified)


def check_url(url: str, allow_loopback: bool = False) -> str | None:
    """Grund (deutsch), warum die URL nicht abgerufen werden darf, sonst None.

    Alle Adressen, auf die der Host auflöst, müssen öffentlich sein (Loopback nur mit `allow_loopback`)."""
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
        if _is_forbidden(ip, allow_loopback):
            return INTERNAL
    return None
