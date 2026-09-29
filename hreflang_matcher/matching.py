from __future__ import annotations

import numpy as np

from .lang_detect import strip_language
from .models import LanguageSet, Match, MatchResult


def topk_similarity(
    P: np.ndarray, L: np.ndarray, k: int = 5, row_block: int = 1024, col_block: int = 4096
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k Cosinus je Pivot-Zeile und bester Pivot je L-Spalte, blockweise ohne volle Matrix."""
    n, m = int(P.shape[0]), int(L.shape[0])
    k = max(0, min(k, m))
    top_idx = np.full((n, k), -1, dtype=np.int64)
    top_val = np.full((n, k), -np.inf, dtype=np.float32)
    col_best_idx = np.full(m, -1, dtype=np.int64)
    col_best_val = np.full(m, -np.inf, dtype=np.float32)
    if n == 0 or m == 0:
        return top_idx, top_val, col_best_idx

    P = np.ascontiguousarray(P, dtype=np.float32)
    L = np.ascontiguousarray(L, dtype=np.float32)

    for r0 in range(0, n, row_block):
        r1 = min(r0 + row_block, n)
        rows_val = top_val[r0:r1].copy()
        rows_idx = top_idx[r0:r1].copy()
        for c0 in range(0, m, col_block):
            c1 = min(c0 + col_block, m)
            with np.errstate(all="ignore"):
                S = P[r0:r1] @ L[c0:c1].T                      # (b, c)
            # bester Pivot je Spalte
            blk_arg = S.argmax(axis=0)
            blk_max = S[blk_arg, np.arange(c1 - c0)]
            better = blk_max > col_best_val[c0:c1]
            col_best_val[c0:c1] = np.where(better, blk_max, col_best_val[c0:c1])
            col_best_idx[c0:c1] = np.where(better, blk_arg + r0, col_best_idx[c0:c1])
            # Top-k je Zeile mit bisherigem Stand zusammenführen
            kk = min(k, c1 - c0)
            if kk < S.shape[1]:
                part = np.argpartition(S, -kk, axis=1)[:, -kk:]
            else:
                part = np.tile(np.arange(S.shape[1]), (S.shape[0], 1))
            cand_val = np.take_along_axis(S, part, axis=1).astype(np.float32)
            cand_idx = (part + c0).astype(np.int64)
            all_val = np.concatenate([rows_val, cand_val], axis=1)
            all_idx = np.concatenate([rows_idx, cand_idx], axis=1)
            order = np.argsort(-all_val, axis=1, kind="stable")[:, :k]
            rows_val = np.take_along_axis(all_val, order, axis=1)
            rows_idx = np.take_along_axis(all_idx, order, axis=1)
        top_val[r0:r1] = rows_val
        top_idx[r0:r1] = rows_idx
    return top_idx, top_val, col_best_idx


def slug_match(pivot: LanguageSet, other: LanguageSet) -> tuple[list[Match], list[int], list[int]]:
    """Exakter Pfad-Match nach Entfernen des Sprachsegments. Score 1.0, Methode 'slug'."""
    other_by_key: dict[str, int] = {}
    for j, u in enumerate(other.norm_urls):
        other_by_key.setdefault(strip_language(u, other.code), j)
    matches: list[Match] = []
    used_other: set[int] = set()
    rem_p: list[int] = []
    for i, u in enumerate(pivot.norm_urls):
        j = other_by_key.get(strip_language(u, pivot.code))
        if j is not None and j not in used_other:
            used_other.add(j)
            matches.append(Match(
                pivot_url=pivot.urls[i], other_url=other.urls[j], other_code=other.code,
                score=1.0, method="slug", reciprocal=True,
                second_url=None, second_score=None, margin=None,
                confidence="sicher", reason="",
            ))
        else:
            rem_p.append(i)
    rem_o = [j for j in range(len(other.urls)) if j not in used_other]
    return matches, rem_p, rem_o
