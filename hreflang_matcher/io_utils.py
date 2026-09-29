from __future__ import annotations

import json
import re
from urllib.parse import parse_qsl, urlencode, urlparse

import numpy as np
import pandas as pd

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "pk_campaign", "pk_kwd",
}

_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def normalize_url(url) -> str:
    """Normalisiert eine URL für das Matching. Ergebnis ohne Schema, z. B. 'example.com/de/seite'."""
    s = str(url or "").strip()
    if not s or s.lower() == "nan":
        return ""
    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s
    p = urlparse(s)
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    qs = sorted(
        (k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS
    )
    query = urlencode(qs)
    return host + path + ("?" + query if query else "")


def parse_vector(value) -> np.ndarray | None:
    """Parst einen Embedding-Wert (JSON-Array, Zahlenliste, list) zu float32. None wenn nicht lesbar."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32)
        return arr if arr.size > 0 else None
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = np.asarray(json.loads(s), dtype=np.float32)
            if arr.ndim == 1 and arr.size > 0:
                return arr
        except (ValueError, TypeError):
            pass
    nums = _FLOAT_RE.findall(s)
    if not nums:
        return None
    return np.asarray([float(x) for x in nums], dtype=np.float32)
