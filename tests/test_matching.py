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
        S = P @ L.T
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
