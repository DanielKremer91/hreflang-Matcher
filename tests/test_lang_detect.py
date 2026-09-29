import pytest

from hreflang_matcher import lang_detect as ld


class TestCodes:
    @pytest.mark.parametrize("raw,exp", [("de-at", "de-AT"), ("DE", "de"), ("fr-CH", "fr-CH"), (" en ", "en")])
    def test_normalize(self, raw, exp):
        assert ld.normalize_code(raw) == exp

    @pytest.mark.parametrize("raw,exp", [("de", True), ("de-AT", True), ("de-at", True), ("x-default", False), ("deu", False), ("", False), ("de_AT", False)])
    def test_valid(self, raw, exp):
        assert ld.is_valid_code(raw) is exp

    def test_language_of(self):
        assert ld.language_of("de-AT") == "de"
        assert ld.language_of("FR") == "fr"

    def test_collisions(self):
        assert ld.find_language_collisions(["de", "de-AT", "fr"]) == {"de": ["de", "de-AT"]}
        assert ld.find_language_collisions(["de", "fr"]) == {}


class TestSuggestCode:
    def test_path_segment(self):
        urls = ["https://a.com/de/x", "https://a.com/de/y", "https://a.com/impressum"]
        assert ld.suggest_code(urls) == "de"

    def test_path_segment_with_region(self):
        assert ld.suggest_code(["https://a.com/de-at/x", "https://a.com/de-at/y"]) == "de-AT"

    def test_subdomain(self):
        assert ld.suggest_code(["https://fr.a.com/x", "https://fr.a.com/y"]) == "fr"

    def test_www_is_not_a_language(self):
        assert ld.suggest_code(["https://www.a.com/x"]) == ""

    def test_cctld(self):
        assert ld.suggest_code(["https://a.fr/x", "https://a.fr/y"]) == "fr"
        assert ld.suggest_code(["https://a.at/x"]) == ""   # Region ohne Sprache -> leer

    def test_nothing(self):
        assert ld.suggest_code(["https://a.com/x"]) == ""
        assert ld.suggest_code([]) == ""


class TestStripLanguage:
    def test_path_segment(self):
        assert ld.strip_language("a.com/de/seite", "de") == "*/seite"
        assert ld.strip_language("a.com/de-at/seite", "de-AT") == "*/seite"
        assert ld.strip_language("a.com/de/", "de") == "*/"

    def test_subdomain(self):
        assert ld.strip_language("fr.a.com/page", "fr") == "*/page"

    def test_cctld_only_host_replaced(self):
        assert ld.strip_language("a.fr/page", "fr") == "*/page"

    def test_other_segment_kept(self):
        assert ld.strip_language("a.com/produkte/x", "de") == "*/produkte/x"

    def test_root(self):
        assert ld.strip_language("a.com/", "de") == "*/"
