import numpy as np
import pandas as pd
import pytest

from hreflang_matcher import output
from hreflang_matcher.models import Cluster, LanguageSet, Match, MatchResult


def _ls(code, urls, dropped=None):
    dropped = dropped or []
    return LanguageSet(code, code, list(urls), list(urls), np.eye(len(urls), 4, dtype=np.float32),
                       pd.DataFrame(dropped, columns=["URL", "Grund"]))


def _m(p, o, code, score=0.95, conf="sicher", reason="", method="embedding", second=None, second_score=None, margin=None, reciprocal=True,
       preferred=None, preferred_score=None):
    return Match(p, o, code, score, method, reciprocal, second, second_score, margin, conf, reason,
                 preferred_pivot_url=preferred, preferred_pivot_score=preferred_score)


@pytest.fixture
def scenario():
    de = _ls("de", ["https://a.com/de/1", "https://a.com/de/2", "https://a.com/de/3"], dropped=[("https://a.com/de/bad", "Embedding nicht lesbar")])
    fr = _ls("fr", ["https://a.com/fr/1", "https://a.com/fr/2", "https://a.com/fr/x"])
    en = _ls("en", ["https://a.com/en/1"])
    matches = [
        _m("https://a.com/de/1", "https://a.com/fr/1", "fr", method="slug", score=1.0),
        _m("https://a.com/de/1", "https://a.com/en/1", "en"),
        _m("https://a.com/de/2", "https://a.com/fr/2", "fr", score=0.82, conf="prüfen", reason="nicht reziprok",
           second="https://a.com/fr/x", second_score=0.81, margin=0.01, reciprocal=False,
           preferred="https://a.com/de/3", preferred_score=0.9),
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

    def test_unknown_pivot_raises(self, scenario):
        sets, result = scenario
        result.pivot_code = "it"
        with pytest.raises(ValueError, match="Pivot-Sprache 'it' nicht in den Sprachsets"):
            output.build_clusters(result, sets)


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
            "Bevorzugte Pivot-URL (fr)", "Score bevorzugte Pivot-URL (fr)",
            "URL (en)", "Score (en)", "Methode (en)", "Konfidenz (en)", "Grund (en)", "Zweitbeste URL (en)", "Zweitbester Score (en)",
            "Bevorzugte Pivot-URL (en)", "Score bevorzugte Pivot-URL (en)",
            "Cluster-Konfidenz", "x-default-Fallback",
        ]
        assert len(df) == 3
        r0 = df.iloc[0]
        assert r0["URL (fr)"] == "https://a.com/fr/1" and r0["Methode (fr)"] == "slug" and r0["x-default-Fallback"] == "nein"
        r1 = df.iloc[1]
        assert r1["Konfidenz (fr)"] == "prüfen" and r1["Grund (fr)"] == "nicht reziprok"
        assert r1["Zweitbeste URL (fr)"] == "https://a.com/fr/x" and r1["Zweitbester Score (fr)"] == 0.81
        assert r1["Bevorzugte Pivot-URL (fr)"] == "https://a.com/de/3" and r1["Score bevorzugte Pivot-URL (fr)"] == 0.9
        assert r0["Bevorzugte Pivot-URL (fr)"] == "" and pd.isna(r0["Score bevorzugte Pivot-URL (fr)"])
        assert r1["URL (en)"] == "" and r1["x-default-Fallback"] == "ja"
        r2 = df.iloc[2]
        assert r2["URL (fr)"] == "" and r2["Cluster-Konfidenz"] == "kein Treffer" and r2["x-default-Fallback"] == "nein"


class TestLeanMappingTable:
    def test_only_urls_and_scores(self, scenario):
        sets, result = scenario
        cl = output.build_clusters(result, sets)
        df = output.lean_mapping_table(cl, ["de", "fr", "en"], "de")
        assert list(df.columns) == ["Pivot-URL (de)", "URL (fr)", "Score (fr)", "URL (en)", "Score (en)"]
        assert len(df) == 3
        r0 = df.iloc[0]
        assert r0["Pivot-URL (de)"] == "https://a.com/de/1"
        assert r0["URL (fr)"] == "https://a.com/fr/1" and r0["Score (fr)"] == 1.0
        assert r0["URL (en)"] == "https://a.com/en/1" and r0["Score (en)"] == 0.95
        r2 = df.iloc[2]
        assert r2["URL (fr)"] == "" and pd.isna(r2["Score (fr)"])

    def test_empty(self):
        df = output.lean_mapping_table([], ["de", "fr"], "de")
        assert list(df.columns) == ["Pivot-URL (de)", "URL (fr)", "Score (fr)"] and df.empty


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


class TestFormatCode:
    def test_modes(self):
        assert output.format_hreflang_code("de-at", "language-region") == "de-AT"
        assert output.format_hreflang_code("de-at", "language") == "de"
        assert output.format_hreflang_code("fr", "language-region") == "fr"


class TestHtmlBlocks:
    def _clusters(self):
        c1 = Cluster("https://a.com/de/1?x=1&y=2", {"de": "https://a.com/de/1?x=1&y=2", "fr": "https://a.com/fr/1", "en": "https://a.com/en/1"}, {}, "sicher")
        c2 = Cluster("https://a.com/de/2", {"de": "https://a.com/de/2", "fr": "https://a.com/fr/2"}, {}, "prüfen")
        c3 = Cluster("https://a.com/de/3", {"de": "https://a.com/de/3"}, {}, "sicher")
        return [c1, c2, c3]

    def test_block_structure_order_and_escaping(self):
        out = output.html_blocks(self._clusters(), "de", "en", "language-region", include_review=False)
        blocks = out.strip().split("\n\n")
        assert len(blocks) == 3   # nur Cluster 1, ein Block je Mitglied
        b = blocks[0].split("\n")
        assert b[0] == "<!-- https://a.com/de/1?x=1&amp;y=2 -->"
        assert b[1] == '<link rel="alternate" hreflang="de" href="https://a.com/de/1?x=1&amp;y=2" />'
        assert b[2] == '<link rel="alternate" hreflang="en" href="https://a.com/en/1" />'
        assert b[3] == '<link rel="alternate" hreflang="fr" href="https://a.com/fr/1" />'
        assert b[4] == '<link rel="alternate" hreflang="x-default" href="https://a.com/en/1" />'
        assert blocks[1].startswith("<!-- https://a.com/en/1 -->") or blocks[1].startswith("<!-- https://a.com/fr/1 -->")
        assert blocks[1].split("\n")[1:] == b[1:]   # identische Tag-Liste
        assert out.endswith("\n")

    def test_include_review_and_xdefault_fallback(self):
        out = output.html_blocks(self._clusters(), "de", "en", "language-region", include_review=True)
        blocks = out.strip().split("\n\n")
        assert len(blocks) == 5
        c2_block = next(b for b in blocks if b.startswith("<!-- https://a.com/de/2 -->"))
        assert 'hreflang="x-default" href="https://a.com/de/2"' in c2_block

    def test_no_xdefault(self):
        out = output.html_blocks(self._clusters(), "de", None, "language-region", include_review=False)
        assert "x-default" not in out

    def test_language_mode(self):
        c = Cluster("p", {"de-AT": "https://a.at/p", "fr-CH": "https://a.ch/fr/p"}, {}, "sicher")
        out = output.html_blocks([c], "de-AT", None, "language", include_review=False)
        assert 'hreflang="de"' in out and 'hreflang="fr"' in out and "de-AT" not in out

    def test_empty(self):
        assert output.html_blocks([], "de", None, "language", include_review=True) == ""

    def test_language_mode_collision_raises(self):
        c = Cluster("https://a.at/p", {"de-AT": "https://a.at/p", "de-CH": "https://a.ch/p"}, {}, "sicher")
        with pytest.raises(ValueError, match="Mehrere Varianten derselben Sprache"):
            output.html_blocks([c], "de-AT", None, "language", include_review=False)
        # language-region mode should still work
        out = output.html_blocks([c], "de-AT", None, "language-region", include_review=False)
        assert 'hreflang="de-AT"' in out and 'hreflang="de-CH"' in out
