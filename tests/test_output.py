import numpy as np
import pandas as pd
import pytest

from hreflang_matcher import output
from hreflang_matcher.models import Cluster, LanguageSet, Match, MatchResult


def _ls(code, urls, dropped=None):
    dropped = dropped or []
    return LanguageSet(code, code, list(urls), list(urls), np.eye(len(urls), 4, dtype=np.float32),
                       pd.DataFrame(dropped, columns=["URL", "Grund"]))


def _m(p, o, code, score=0.95, conf="sicher", reason="", method="embedding", second=None, second_score=None, margin=None, reciprocal=True):
    return Match(p, o, code, score, method, reciprocal, second, second_score, margin, conf, reason)


@pytest.fixture
def scenario():
    de = _ls("de", ["https://a.com/de/1", "https://a.com/de/2", "https://a.com/de/3"], dropped=[("https://a.com/de/bad", "Embedding nicht lesbar")])
    fr = _ls("fr", ["https://a.com/fr/1", "https://a.com/fr/2", "https://a.com/fr/x"])
    en = _ls("en", ["https://a.com/en/1"])
    matches = [
        _m("https://a.com/de/1", "https://a.com/fr/1", "fr", method="slug", score=1.0),
        _m("https://a.com/de/1", "https://a.com/en/1", "en"),
        _m("https://a.com/de/2", "https://a.com/fr/2", "fr", score=0.82, conf="prüfen", reason="nicht reziprok",
           second="https://a.com/fr/x", second_score=0.81, margin=0.01, reciprocal=False),
    ]
    result = MatchResult("de", matches, {"fr": ["https://a.com/fr/x"], "en": [], "de": ["https://a.com/de/3"]}, {})
    return [de, fr, en], result


class TestBuildClusters:
    def test_clusters(self, scenario):
        sets, result = scenario
        cl = output.build_clusters(result, sets)
        assert [c.pivot_url for c in cl] == ["https://a.com/de/1", "https://a.com/de/2", "https://a.com/de/3"]
        assert cl[0].members == {"de": "https://a.com/de/1", "fr": "https://a.com/fr/1", "en": "https://a.com/en/1"}
        assert cl[0].confidence == "sicher"
        assert cl[1].members == {"de": "https://a.com/de/2", "fr": "https://a.com/fr/2"}
        assert cl[1].confidence == "prüfen"
        assert cl[2].members == {"de": "https://a.com/de/3"} and cl[2].matches == {}


class TestXDefaultFallback:
    def test_rules(self):
        c = Cluster("p", {"de": "p", "fr": "f"}, {}, "sicher")
        assert output.x_default_fallback(c, "de") is False
        assert output.x_default_fallback(c, "en") is True
        assert output.x_default_fallback(c, None) is False
        single = Cluster("p", {"de": "p"}, {}, "sicher")
        assert output.x_default_fallback(single, "en") is False


class TestMappingTable:
    def test_columns_and_values(self, scenario):
        sets, result = scenario
        cl = output.build_clusters(result, sets)
        df = output.mapping_table(cl, ["de", "fr", "en"], "de", "en")
        assert list(df.columns) == [
            "Pivot-URL (de)",
            "URL (fr)", "Score (fr)", "Methode (fr)", "Konfidenz (fr)", "Grund (fr)", "Zweitbeste URL (fr)", "Zweitbester Score (fr)",
            "URL (en)", "Score (en)", "Methode (en)", "Konfidenz (en)", "Grund (en)", "Zweitbeste URL (en)", "Zweitbester Score (en)",
            "Cluster-Konfidenz", "x-default-Fallback",
        ]
        assert len(df) == 3
        r0 = df.iloc[0]
        assert r0["URL (fr)"] == "https://a.com/fr/1" and r0["Methode (fr)"] == "slug" and r0["x-default-Fallback"] == "nein"
        r1 = df.iloc[1]
        assert r1["Konfidenz (fr)"] == "prüfen" and r1["Grund (fr)"] == "nicht reziprok"
        assert r1["Zweitbeste URL (fr)"] == "https://a.com/fr/x" and r1["Zweitbester Score (fr)"] == 0.81
        assert r1["URL (en)"] == "" and r1["x-default-Fallback"] == "ja"
        r2 = df.iloc[2]
        assert r2["URL (fr)"] == "" and r2["Cluster-Konfidenz"] == "sicher" and r2["x-default-Fallback"] == "nein"


class TestUnmatchedTable:
    def test_rows(self, scenario):
        sets, result = scenario
        df = output.unmatched_table(result, sets)
        assert list(df.columns) == ["Sprache", "URL", "Grund"]
        rows = set(map(tuple, df.values.tolist()))
        assert ("de", "https://a.com/de/bad", "Embedding nicht lesbar") in rows
        assert ("de", "https://a.com/de/3", "kein Treffer über Threshold") in rows
        assert ("fr", "https://a.com/fr/x", "kein Treffer über Threshold") in rows
        assert len(df) == 3
