import numpy as np

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
