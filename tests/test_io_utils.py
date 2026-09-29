import numpy as np
import pandas as pd
import pytest

from hreflang_matcher import io_utils


class TestNormalizeUrl:
    def test_strips_scheme_www_trailing_slash_and_tracking(self):
        assert io_utils.normalize_url("https://www.Example.com/de/seite/?utm_source=x") == "example.com/de/seite"

    def test_keeps_root_slash(self):
        assert io_utils.normalize_url("https://example.com/") == "example.com/"
        assert io_utils.normalize_url("https://example.com") == "example.com/"

    def test_sorts_query_and_drops_fragment(self):
        assert io_utils.normalize_url("http://example.com/p?b=2&a=1#top") == "example.com/p?a=1&b=2"

    def test_adds_scheme_when_missing(self):
        assert io_utils.normalize_url("example.com/x/") == "example.com/x"

    def test_empty(self):
        assert io_utils.normalize_url("") == ""
        assert io_utils.normalize_url(None) == ""

    def test_malformed_url(self):
        assert io_utils.normalize_url("http://[") == "http://["


class TestParseVector:
    def test_json_array(self):
        v = io_utils.parse_vector("[0.1, -0.2, 3e-3]")
        assert v is not None
        np.testing.assert_allclose(v, [0.1, -0.2, 0.003], rtol=1e-5)
        assert v.dtype == np.float32

    def test_comma_separated_without_brackets(self):
        v = io_utils.parse_vector("0.1,0.2,0.3")
        np.testing.assert_allclose(v, [0.1, 0.2, 0.3], rtol=1e-5)

    def test_semicolon_and_whitespace(self):
        v = io_utils.parse_vector("0.1; 0.2 0.3|0.4")
        assert len(v) == 4

    def test_list_input(self):
        v = io_utils.parse_vector([1, 2, 3])
        assert v.shape == (3,)

    def test_invalid_returns_none(self):
        assert io_utils.parse_vector("") is None
        assert io_utils.parse_vector(None) is None
        assert io_utils.parse_vector(float("nan")) is None
        assert io_utils.parse_vector("abc") is None

    def test_non_numeric_list(self):
        assert io_utils.parse_vector(["a", "b"]) is None

    def test_multidimensional_array(self):
        assert io_utils.parse_vector([[1, 2], [3, 4]]) is None


def _vec(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return "[" + ", ".join(f"{x:.4f}" for x in rng.normal(size=n)) + "]"


class TestReadAnyFile:
    def test_csv_utf8_comma(self):
        raw = "Address,Embeddings\nhttps://a.de/x,\"[0.1, 0.2]\"\n".encode("utf-8-sig")
        df = io_utils.read_any_file("crawl.csv", raw)
        assert list(df.columns) == ["Address", "Embeddings"]
        assert len(df) == 1

    def test_csv_semicolon_latin1(self):
        raw = "Adresse;Embedding\nhttps://a.de/ü;\"[0.1, 0.2]\"\n".encode("latin1")
        df = io_utils.read_any_file("crawl.csv", raw)
        assert list(df.columns) == ["Adresse", "Embedding"]

    def test_csv_bad_line_is_counted_not_silent(self):
        raw = (
            "Address;Embedding\n"
            "https://a.de/1;\"[0.1, 0.2]\"\n"
            "https://a.de/2;\"[0.3, 0.4]\"\n"
            "https://a.de/kaputt;\"[0.5, 0.6]\";zu viel\n"
            "https://a.de/3;\"[0.7, 0.8]\"\n"
        ).encode("utf-8")
        df = io_utils.read_any_file("crawl.csv", raw)
        assert df["Address"].tolist() == ["https://a.de/1", "https://a.de/2", "https://a.de/3"]
        assert df.attrs["bad_lines"] == 1

    def test_csv_without_bad_lines_reports_zero(self):
        raw = "Address,Embeddings\nhttps://a.de/x,\"[0.1, 0.2]\"\n".encode("utf-8")
        assert io_utils.read_any_file("crawl.csv", raw).attrs["bad_lines"] == 0

    def test_xlsx(self, tmp_path):
        p = tmp_path / "c.xlsx"
        pd.DataFrame({"Address": ["https://a.de/x"], "Embeddings": ["[0.1, 0.2]"]}).to_excel(p, index=False)
        df = io_utils.read_any_file("c.xlsx", p.read_bytes())
        assert "Address" in df.columns

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            io_utils.read_any_file("x.xlsx", b"not an excel file")


class TestDetectColumns:
    def test_url_by_name(self):
        df = pd.DataFrame({"Foo": ["a"], "Address": ["https://a.de/x"]})
        assert io_utils.detect_url_column(df) == "Address"

    def test_url_by_content(self):
        df = pd.DataFrame({"Spalte1": ["https://a.de/x", "https://a.de/y", "https://a.de/z"], "Spalte2": [1, 2, 3]})
        assert io_utils.detect_url_column(df) == "Spalte1"

    def test_url_none(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert io_utils.detect_url_column(df) is None

    def test_embedding_by_name(self):
        df = pd.DataFrame({"Address": ["u"], "OpenAI Embedding 1": [_vec()]})
        assert io_utils.detect_embedding_column(df) == "OpenAI Embedding 1"

    def test_embedding_by_content(self):
        df = pd.DataFrame({"Address": ["u"] * 3, "Extraktion 1": [_vec(), _vec(), _vec()]})
        assert io_utils.detect_embedding_column(df) == "Extraktion 1"

    def test_embedding_ignores_short_numeric(self):
        df = pd.DataFrame({"Address": ["u"] * 3, "Status Code": ["200", "200", "404"]})
        assert io_utils.detect_embedding_column(df) is None

    def test_embedding_detected_despite_truncated_row(self):
        urls = [f"https://a.de/{i}" for i in range(11)]
        vecs = [_vec(8, i) for i in range(10)] + ["[0.1, 0.2, 0.3]"]
        df = pd.DataFrame({"Address": urls, "OpenAI Embedding 1": vecs})
        assert io_utils.detect_embedding_column(df) == "OpenAI Embedding 1"

    def test_embedding_named_column_mostly_broken_still_detected(self):
        df = pd.DataFrame({"Embeddings": [_vec(), _vec(8, 1)] + ["kaputt"] * 8})
        assert io_utils.detect_embedding_column(df) == "Embeddings"

    def test_embedding_unnamed_column_mostly_broken_not_detected(self):
        df = pd.DataFrame({"Spalte X": [_vec(), _vec(8, 1)] + ["kaputt"] * 8})
        assert io_utils.detect_embedding_column(df) is None

    def test_status_and_indexability_german_and_english(self):
        df = pd.DataFrame({"Statuscode": [200], "Indexierbarkeit": ["Indexierbar"]})
        assert io_utils.detect_status_column(df) == "Statuscode"
        assert io_utils.detect_indexability_column(df) == "Indexierbarkeit"
        df2 = pd.DataFrame({"Status Code": [200], "Indexability": ["Indexable"]})
        assert io_utils.detect_status_column(df2) == "Status Code"
        assert io_utils.detect_indexability_column(df2) == "Indexability"

    def test_text_status_column_is_not_a_status_code(self):
        df = pd.DataFrame({"Status": ["OK", "Not Found"], "Address": ["https://a.de/x", "https://a.de/y"]})
        assert io_utils.detect_status_column(df) is None


class TestIsIndexable:
    @pytest.mark.parametrize("v,exp", [
        ("Indexable", True), ("indexable", True), ("Indexierbar", True),
        ("Non-Indexable", False), ("Nicht indexierbar", False), ("", False), (None, False),
    ])
    def test_values(self, v, exp):
        assert io_utils.is_indexable(v) is exp


class TestBuildLanguageSet:
    def _df(self):
        return pd.DataFrame({
            "Address": [
                "https://a.de/1/", "https://a.de/2", "https://a.de/1",  # 3 ist Duplikat von 1
                "https://a.de/3", "https://a.de/4", "https://a.de/5", "",
            ],
            "Embeddings": [_vec(8, 1), _vec(8, 2), _vec(8, 3), _vec(5, 4), "kaputt", _vec(8, 6), _vec(8, 7)],
            "Status Code": [200, 200, 200, 200, 200, 404, 200],
            "Indexability": ["Indexable"] * 7,
        })

    def test_builds_and_drops_with_reasons(self):
        ls = io_utils.build_language_set(
            self._df(), "de", "de.csv", "Address", "Embeddings",
            filter_indexable=True, status_col="Status Code", index_col="Indexability",
        )
        assert ls.urls == ["https://a.de/1/", "https://a.de/2"]
        assert ls.norm_urls == ["a.de/1", "a.de/2"]
        assert ls.vectors.shape == (2, 8)
        np.testing.assert_allclose(np.linalg.norm(ls.vectors, axis=1), [1.0, 1.0], rtol=1e-5)
        reasons = dict(zip(ls.dropped["URL"], ls.dropped["Grund"]))
        assert "Duplikat" in reasons["https://a.de/1"]
        assert "Dimension 5 statt 8" in reasons["https://a.de/3"]
        assert "nicht lesbar" in reasons["https://a.de/4"]
        assert "Status Code 404" in reasons["https://a.de/5"]
        assert "leere URL" in reasons[""]

    def test_no_filter_keeps_404(self):
        ls = io_utils.build_language_set(
            self._df(), "de", "de.csv", "Address", "Embeddings",
            filter_indexable=False, status_col="Status Code", index_col="Indexability",
        )
        assert "https://a.de/5" in ls.urls

    def test_non_indexable_dropped(self):
        df = self._df()
        df.loc[1, "Indexability"] = "Non-Indexable"
        ls = io_utils.build_language_set(
            df, "de", "de.csv", "Address", "Embeddings",
            filter_indexable=True, status_col="Status Code", index_col="Indexability",
        )
        assert "https://a.de/2" not in ls.urls

    def test_empty_result_has_2d_vectors(self):
        df = pd.DataFrame({"Address": ["https://a.de/x"], "Embeddings": ["kaputt"]})
        ls = io_utils.build_language_set(df, "de", "x", "Address", "Embeddings", filter_indexable=False)
        assert ls.vectors.shape == (0, 0)
        assert len(ls) == 0


class TestCheckDimensions:
    def _ls(self, code, dim):
        from hreflang_matcher.models import LanguageSet
        return LanguageSet(code, code, ["u"], ["u"], np.ones((1, dim), dtype=np.float32), pd.DataFrame(columns=["URL", "Grund"]))

    def test_equal_ok(self):
        io_utils.check_dimensions([self._ls("de", 8), self._ls("fr", 8)])

    def test_unequal_raises_with_details(self):
        with pytest.raises(ValueError) as e:
            io_utils.check_dimensions([self._ls("de", 8), self._ls("fr", 16)])
        assert "de: 8" in str(e.value) and "fr: 16" in str(e.value)
