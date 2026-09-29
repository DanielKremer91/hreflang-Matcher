from __future__ import annotations

import re
from collections import Counter
from urllib.parse import urlparse

from .models import PatternSuggestion

CODE_RE = re.compile(r"^([a-z]{2})(?:-([a-z]{2}))?$", re.I)

# ccTLD -> Sprache. Leere Werte sind Regionen ohne eindeutige Sprache.
CCTLD_LANG = {
    "de": "de", "fr": "fr", "it": "it", "es": "es", "nl": "nl", "pl": "pl", "pt": "pt",
    "cz": "cs", "dk": "da", "se": "sv", "no": "no", "fi": "fi", "ru": "ru", "jp": "ja",
    "cn": "zh", "uk": "en", "us": "en", "hu": "hu", "ro": "ro", "gr": "el", "tr": "tr",
    "at": "", "ch": "", "be": "", "ca": "", "com": "", "net": "", "org": "", "eu": "",
}


def normalize_code(code: str) -> str:
    s = (code or "").strip()
    m = CODE_RE.match(s)
    if not m:
        return s
    lang, reg = m.group(1).lower(), m.group(2)
    return f"{lang}-{reg.upper()}" if reg else lang


def is_valid_code(code: str) -> bool:
    s = (code or "").strip()
    return bool(CODE_RE.match(s)) and s.lower() != "x-default"


def language_of(code: str) -> str:
    return (code or "").strip().split("-")[0].lower()


def find_language_collisions(codes: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in codes:
        groups.setdefault(language_of(c), []).append(c)
    return {lang: cs for lang, cs in groups.items() if len(cs) > 1}


def _host_and_path(url: str) -> tuple[str, str]:
    s = str(url or "").strip()
    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s
    p = urlparse(s)
    return (p.hostname or "").lower(), (p.path or "/")


def _code_candidate(url: str) -> str:
    host, path = _host_and_path(url)
    segs = [x for x in path.split("/") if x]
    if segs and CODE_RE.match(segs[0]):
        return normalize_code(segs[0])
    parts = host.split(".")
    if len(parts) >= 3 and len(parts[0]) == 2 and parts[0] != "www":
        return parts[0].lower()
    tld = parts[-1] if parts else ""
    return CCTLD_LANG.get(tld, "")


def suggest_code(urls: list[str]) -> str:
    counts = Counter(c for c in (_code_candidate(u) for u in urls) if c)
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def strip_language(norm_url: str, code: str) -> str:
    """Entfernt Host und Sprachsegment aus einer normalisierten URL ('host/path'). Ergebnis '*/rest'."""
    s = norm_url or ""
    slash = s.find("/")
    path = s[slash:] if slash >= 0 else "/"
    lang = language_of(code)
    segs = path.split("/")  # segs[0] == "" wegen führendem "/"
    if len(segs) > 1 and segs[1]:
        first = segs[1].lower()
        if first == code.lower() or first == lang or (CODE_RE.match(first) and language_of(first) == lang):
            rest = "/" + "/".join(segs[2:])
            return "*" + rest
    return "*" + path


def suggest_patterns(urls: list[str]) -> list[PatternSuggestion]:
    """Schlägt Regex-Muster aus Pfadsegment, Subdomain und ccTLD vor, nach Häufigkeit sortiert."""
    buckets: dict[str, dict] = {}
    for u in urls:
        host, path = _host_and_path(u)
        if not host:
            continue
        segs = [x for x in path.split("/") if x]
        parts = host.split(".")
        key = None
        code = ""
        if segs and CODE_RE.match(segs[0]):
            seg = segs[0].lower()
            key = rf"^https?://[^/]+/{re.escape(seg)}(/|$)"
            code = normalize_code(seg)
        elif len(parts) >= 3 and len(parts[0]) == 2 and parts[0] != "www":
            sub = parts[0]
            key = rf"^https?://{re.escape(sub)}\."
            code = sub
        else:
            tld = parts[-1]
            if tld in CCTLD_LANG:
                key = rf"^https?://[^/]+\.{re.escape(tld)}(/|$)"
                code = CCTLD_LANG[tld]
        if key is None:
            continue
        b = buckets.setdefault(key, {"code": code, "example": u, "count": 0})
        b["count"] += 1
    out = [PatternSuggestion(pattern=k, code=v["code"], example=v["example"], count=v["count"]) for k, v in buckets.items()]
    out.sort(key=lambda s: (-s.count, s.pattern))
    return out


def split_by_patterns(
    urls: list[str], patterns: list[tuple[str, str]]
) -> tuple[dict[str, list[int]], list[int], list[str]]:
    """Ordnet jede URL dem ersten passenden Muster zu (top-down). Leere Muster/Codes werden übersprungen."""
    compiled: list[tuple[re.Pattern, str]] = []
    invalid: list[str] = []
    for pat, code in patterns:
        pat = (pat or "").strip()
        code = (code or "").strip()
        if not pat or not code:
            continue
        try:
            compiled.append((re.compile(pat, re.I), code))
        except re.error:
            invalid.append(pat)
    groups: dict[str, list[int]] = {}
    rest: list[int] = []
    for i, u in enumerate(urls):
        s = str(u or "")
        for rx, code in compiled:
            if rx.search(s):
                groups.setdefault(code, []).append(i)
                break
        else:
            rest.append(i)
    return groups, rest, invalid
