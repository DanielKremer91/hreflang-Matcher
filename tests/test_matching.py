import numpy as np
import pandas as pd
import pytest

from hreflang_matcher import matching
from hreflang_matcher.models import LanguageSet


def _norm(M):
    M = np.asarray(M, dtype=np.float32)
    return M / np.linalg.norm(M, axis=1, keepdims=True)


class TestTopkSimilarity:
    def test_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        P = _norm(rng.normal(size=(50, 16)))
        L = _norm(rng.normal(size=(70, 16)))
        with np.errstate(all="ignore"):
            S = (P.astype(np.float64) @ L.astype(np.float64).T).astype(np.float32)
        top_idx, top_val, col_best = matching.topk_similarity(P, L, k=5, row_block=7, col_block=11)
        exp_idx = np.argsort(-S, axis=1)[:, :5]
        np.testing.assert_array_equal(top_idx, exp_idx)
        np.testing.assert_allclose(top_val, np.take_along_axis(S, exp_idx, axis=1), rtol=1e-5)
        np.testing.assert_array_equal(col_best, S.argmax(axis=0))

    def test_k_larger_than_m_is_capped(self):
        P = _norm(np.eye(3, 4))
        L = _norm(np.eye(2, 4))
        top_idx, top_val, col_best = matching.topk_similarity(P, L, k=5)
        assert top_idx.shape == (3, 2)
        assert top_idx[0, 0] == 0 and top_idx[1, 0] == 1
        assert top_val[2].tolist() == [0.0, 0.0]
        assert col_best.tolist() == [0, 1]

    def test_empty_inputs(self):
        P = np.zeros((0, 4), dtype=np.float32)
        L = _norm(np.eye(2, 4))
        top_idx, top_val, col_best = matching.topk_similarity(P, L, k=3)
        assert top_idx.shape == (0, 2) and col_best.shape == (2,) and (col_best == -1).all()
        top_idx, top_val, col_best = matching.topk_similarity(L, P, k=3)
        assert top_idx.shape == (2, 0) and col_best.shape == (0,)


def _ls(code, urls, vectors=None):
    from hreflang_matcher.io_utils import normalize_url
    if vectors is None:
        vectors = np.eye(len(urls), max(len(urls), 4), dtype=np.float32)
    return LanguageSet(code, code, list(urls), [normalize_url(u) for u in urls], _norm(vectors),
                       pd.DataFrame(columns=["URL", "Grund"]))


class TestSlugMatch:
    def test_matches_identical_slug_across_language_segment(self):
        de = _ls("de", ["https://a.com/de/x/", "https://a.com/de/y", "https://a.com/de/only-de"])
        fr = _ls("fr", ["https://a.com/fr/y", "https://a.com/fr/x", "https://a.com/fr/only-fr"])
        matches, rem_p, rem_o = matching.slug_match(de, fr)
        pairs = {(m.pivot_url, m.other_url) for m in matches}
        assert pairs == {("https://a.com/de/x/", "https://a.com/fr/x"), ("https://a.com/de/y", "https://a.com/fr/y")}
        assert all(m.method == "slug" and m.score == 1.0 and m.confidence == "sicher" and m.reciprocal for m in matches)
        assert rem_p == [2] and rem_o == [2]

    def test_subdomain_and_tld(self):
        de = _ls("de", ["https://a.de/p"])
        fr = _ls("fr", ["https://fr.a.com/p"])
        matches, rem_p, rem_o = matching.slug_match(de, fr)
        assert len(matches) == 1 and rem_p == [] and rem_o == []

    def test_no_matches(self):
        de = _ls("de", ["https://a.com/de/x"])
        fr = _ls("fr", ["https://a.com/fr/y"])
        matches, rem_p, rem_o = matching.slug_match(de, fr)
        assert matches == [] and rem_p == [0] and rem_o == [0]


class TestEmbeddingMatch:
    def _sets(self):
        # de0 ~ fr1 (0.99), de1 ~ fr0 (0.95), de2 ~ nichts, fr2 ~ nichts
        base = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        de = _ls("de", ["https://a.com/de/0", "https://a.com/de/1", "https://a.com/de/2"],
                 base[:3])
        fr_vecs = np.array([[0.05, 1, 0, 0], [1, 0.1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        fr = _ls("fr", ["https://a.com/fr/0", "https://a.com/fr/1", "https://a.com/fr/2"], fr_vecs)
        return de, fr

    def test_basic_assignment_and_unmatched(self):
        de, fr = self._sets()
        ms = matching.embedding_match(de, fr, [0, 1, 2], [0, 1, 2], threshold=0.8, band=0.0, min_margin=0.0)
        pairs = {(m.pivot_url, m.other_url): m for m in ms}
        assert set(pairs) == {("https://a.com/de/0", "https://a.com/fr/1"), ("https://a.com/de/1", "https://a.com/fr/0")}
        m = pairs[("https://a.com/de/0", "https://a.com/fr/1")]
        assert m.method == "embedding" and m.reciprocal and m.confidence == "sicher" and m.reason == ""
        assert m.second_url == "https://a.com/fr/0" and m.second_score is not None
        assert m.margin == pytest.approx(m.score - m.second_score, abs=1e-6)

    def test_threshold_excludes(self):
        de, fr = self._sets()
        ms = matching.embedding_match(de, fr, [0, 1, 2], [0, 1, 2], threshold=0.999, band=0.0, min_margin=0.0)
        assert ms == []

    def test_band_marks_review(self):
        de, fr = self._sets()
        ms = matching.embedding_match(de, fr, [0, 1, 2], [0, 1, 2], threshold=0.9, band=0.2, min_margin=0.0)
        assert all(m.confidence == "prüfen" and "knapp über Threshold" in m.reason for m in ms)

    def test_margin_marks_ambiguous_and_names_second(self):
        # fr0 und fr1 fast gleich nah an de0
        de = _ls("de", ["https://a.com/de/0"], np.array([[1, 0, 0, 0]], dtype=np.float32))
        fr = _ls("fr", ["https://a.com/fr/0", "https://a.com/fr/1"],
                 np.array([[1, 0.10, 0, 0], [1, 0.11, 0, 0]], dtype=np.float32))
        ms = matching.embedding_match(de, fr, [0], [0, 1], threshold=0.5, band=0.0, min_margin=0.05)
        assert len(ms) == 1
        m = ms[0]
        assert m.other_url == "https://a.com/fr/0" and m.second_url == "https://a.com/fr/1"
        assert m.confidence == "prüfen" and "mehrdeutig" in m.reason
        assert m.margin < 0.05

    def test_greedy_one_to_one_and_non_reciprocal(self):
        # de0 und de1 wollen beide fr0; de0 ist näher. de1 bekommt fr1 (nicht reziprok, weil fr1 de0 lieber mag).
        de = _ls("de", ["https://a.com/de/0", "https://a.com/de/1"],
                 np.array([[1, 0, 0, 0], [0.9, 0.3, 0, 0]], dtype=np.float32))
        fr = _ls("fr", ["https://a.com/fr/0", "https://a.com/fr/1"],
                 np.array([[1, 0, 0, 0], [0.6, 0.8, 0, 0]], dtype=np.float32))
        # de0·fr0 = 1.0, de1·fr0 = 0.949, de1·fr1 = 0.822, de0·fr1 = 0.6
        ms = matching.embedding_match(de, fr, [0, 1], [0, 1], threshold=0.5, band=0.0, min_margin=0.0)
        by_pivot = {m.pivot_url: m for m in ms}
        assert by_pivot["https://a.com/de/0"].other_url == "https://a.com/fr/0"
        assert by_pivot["https://a.com/de/1"].other_url == "https://a.com/fr/1"
        assert by_pivot["https://a.com/de/1"].reciprocal is False
        assert "nicht reziprok" in by_pivot["https://a.com/de/1"].reason
        urls_o = [m.other_url for m in ms]
        assert len(urls_o) == len(set(urls_o))

    def test_respects_remaining_indices(self):
        de, fr = self._sets()
        ms = matching.embedding_match(de, fr, [1], [0, 1, 2], threshold=0.8, band=0.0, min_margin=0.0)
        assert len(ms) == 1 and ms[0].pivot_url == "https://a.com/de/1"

    def test_empty_remaining(self):
        de, fr = self._sets()
        assert matching.embedding_match(de, fr, [], [0], threshold=0.5, band=0.0, min_margin=0.0) == []
        assert matching.embedding_match(de, fr, [0], [], threshold=0.5, band=0.0, min_margin=0.0) == []
