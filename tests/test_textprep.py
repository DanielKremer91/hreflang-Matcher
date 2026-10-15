import pandas as pd

from hreflang_matcher import textprep


class TestSuggestTextColumns:
    def test_hint_order_main_content_first(self):
        df = pd.DataFrame(columns=["Address", "Title 1", "H1-1", "Main Content Extraction 1", "Status Code"])
        assert textprep.suggest_text_columns(df) == ["Main Content Extraction 1", "Title 1", "H1-1"]

    def test_german_and_meta(self):
        df = pd.DataFrame(columns=["Adresse", "Meta Description 1", "Body Text", "Inlinks"])
        assert textprep.suggest_text_columns(df) == ["Body Text", "Meta Description 1"]

    def test_none(self):
        assert textprep.suggest_text_columns(pd.DataFrame(columns=["Address", "Inlinks"])) == []


class TestTruncate:
    def test_cuts_and_normalizes(self):
        assert textprep.truncate("  a   b\n\n\tc  ", 100) == "a b c"
        assert textprep.truncate("abcdef", 3) == "abc"
        assert textprep.truncate(None, 3) == ""


class TestBuildTexts:
    def test_joins_in_column_order_and_skips_empty(self):
        df = pd.DataFrame({
            "Address": ["u1", "u2", "u3"],
            "Title": ["Titel 1", None, ""],
            "Main": ["Inhalt   eins", "Inhalt zwei", float("nan")],
        })
        texts, errors = textprep.build_texts(df, ["Title", "Main"], max_chars=100)
        assert texts == {0: "Titel 1\n\nInhalt eins", 1: "Inhalt zwei"}
        assert errors == {2: textprep.NO_TEXT_REASON}

    def test_truncates_total(self):
        df = pd.DataFrame({"T": ["x" * 50], "M": ["y" * 50]})
        texts, _ = textprep.build_texts(df, ["T", "M"], max_chars=60)
        assert len(texts[0]) == 60 and texts[0].startswith("x" * 50)

    def test_positional_index_not_label(self):
        df = pd.DataFrame({"T": ["a", "b"]}, index=[10, 20])
        texts, _ = textprep.build_texts(df, ["T"], max_chars=10)
        assert texts == {0: "a", 1: "b"}


def test_estimate_tokens():
    assert textprep.estimate_tokens(["abcd" * 10, "ab"]) == 10
