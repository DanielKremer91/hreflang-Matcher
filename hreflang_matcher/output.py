from __future__ import annotations

import html

import pandas as pd

from .lang_detect import language_of, normalize_code
from .models import Cluster, LanguageSet, Match, MatchResult

NO_MATCH_REASON = "kein Treffer über Threshold"


def build_clusters(result: MatchResult, sets: list[LanguageSet]) -> list[Cluster]:
    pivot = next(s for s in sets if s.code == result.pivot_code)
    by_pivot: dict[str, dict[str, Match]] = {}
    for m in result.matches:
        by_pivot.setdefault(m.pivot_url, {})[m.other_code] = m
    clusters: list[Cluster] = []
    for u in pivot.urls:
        ms = by_pivot.get(u, {})
        members = {result.pivot_code: u}
        for code, m in ms.items():
            members[code] = m.other_url
        conf = "prüfen" if any(m.confidence == "prüfen" for m in ms.values()) else "sicher"
        clusters.append(Cluster(pivot_url=u, members=members, matches=ms, confidence=conf))
    return clusters


def x_default_fallback(cluster: Cluster, x_default_code: str | None) -> bool:
    """True, wenn x-default gewünscht ist, der Cluster mehr als ein Mitglied hat und die x-default-Sprache fehlt."""
    if not x_default_code or len(cluster.members) < 2:
        return False
    return x_default_code not in cluster.members


def mapping_table(
    clusters: list[Cluster], codes: list[str], pivot_code: str, x_default_code: str | None
) -> pd.DataFrame:
    others = [c for c in codes if c != pivot_code]
    cols = [f"Pivot-URL ({pivot_code})"]
    for c in others:
        cols += [f"URL ({c})", f"Score ({c})", f"Methode ({c})", f"Konfidenz ({c})", f"Grund ({c})",
                 f"Zweitbeste URL ({c})", f"Zweitbester Score ({c})"]
    cols += ["Cluster-Konfidenz", "x-default-Fallback"]

    rows = []
    for cl in clusters:
        row = {cols[0]: cl.pivot_url}
        for c in others:
            m = cl.matches.get(c)
            row[f"URL ({c})"] = m.other_url if m else ""
            row[f"Score ({c})"] = m.score if m else None
            row[f"Methode ({c})"] = m.method if m else ""
            row[f"Konfidenz ({c})"] = m.confidence if m else ""
            row[f"Grund ({c})"] = m.reason if m else ""
            row[f"Zweitbeste URL ({c})"] = (m.second_url or "") if m else ""
            row[f"Zweitbester Score ({c})"] = m.second_score if m else None
        row["Cluster-Konfidenz"] = cl.confidence
        row["x-default-Fallback"] = "ja" if x_default_fallback(cl, x_default_code) else "nein"
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def unmatched_table(result: MatchResult, sets: list[LanguageSet]) -> pd.DataFrame:
    rows = []
    for s in sets:
        for _, r in s.dropped.iterrows():
            rows.append((s.code, r["URL"], r["Grund"]))
        for u in result.unmatched.get(s.code, []):
            rows.append((s.code, u, NO_MATCH_REASON))
    return pd.DataFrame(rows, columns=["Sprache", "URL", "Grund"])
