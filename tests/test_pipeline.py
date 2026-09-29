import numpy as np

from hreflang_matcher import io_utils, lang_detect, matching, output


def _csv(rows: list[tuple[str, np.ndarray, int, str]]) -> bytes:
    lines = ["Address;OpenAI Embedding 1;Status Code;Indexability"]
    for url, vec, status, idx in rows:
        lines.append(f'{url};"[{", ".join(f"{x:.5f}" for x in vec)}]";{status};{idx}')
    return "\n".join(lines).encode("utf-8-sig")


def test_full_pipeline_two_languages():
    rng = np.random.default_rng(42)
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    base = q[:4].astype(np.float32)          # vier orthonormale Vektoren
    noise = lambda: rng.normal(scale=0.05, size=16).astype(np.float32)

    de_rows = [
        ("https://a.com/de/produkt-1/", base[0], 200, "Indexable"),
        ("https://a.com/de/produkt-2/", base[1], 200, "Indexable"),
        ("https://a.com/de/nur-deutsch/", base[2], 200, "Indexable"),
        ("https://a.com/de/geloescht/", base[3], 404, "Non-Indexable"),
    ]
    fr_rows = [
        ("https://a.com/fr/produit-2/", base[1] + noise(), 200, "Indexable"),
        ("https://a.com/fr/produit-1/", base[0] + noise(), 200, "Indexable"),
        ("https://a.com/fr/seulement-fr/", base[3], 200, "Indexable"),
    ]

    sets = []
    for name, rows in (("de.csv", de_rows), ("fr.csv", fr_rows)):
        df = io_utils.read_any_file(name, _csv(rows))
        url_col = io_utils.detect_url_column(df)
        emb_col = io_utils.detect_embedding_column(df)
        assert url_col == "Address" and emb_col == "OpenAI Embedding 1"
        code = lang_detect.suggest_code(df[url_col].astype(str).tolist())
        sets.append(io_utils.build_language_set(
            df, code, name, url_col, emb_col, filter_indexable=True,
            status_col=io_utils.detect_status_column(df), index_col=io_utils.detect_indexability_column(df),
        ))
    assert [s.code for s in sets] == ["de", "fr"]
    assert len(sets[0]) == 3   # 404 gefiltert
    io_utils.check_dimensions(sets)

    result = matching.run_matching(sets, "de", threshold=0.8, band=0.05, min_margin=0.02, use_slug=False)
    pairs = {(m.pivot_url, m.other_url) for m in result.matches}
    assert pairs == {
        ("https://a.com/de/produkt-1/", "https://a.com/fr/produit-1/"),
        ("https://a.com/de/produkt-2/", "https://a.com/fr/produit-2/"),
    }
    assert all(m.confidence == "sicher" for m in result.matches)
    assert result.unmatched["de"] == ["https://a.com/de/nur-deutsch/"]
    assert result.unmatched["fr"] == ["https://a.com/fr/seulement-fr/"]

    clusters = output.build_clusters(result, sets)
    mapping = output.mapping_table(clusters, ["de", "fr"], "de", "de")
    assert len(mapping) == 3
    conf = dict(zip(mapping["Pivot-URL (de)"], mapping["Cluster-Konfidenz"]))
    assert conf["https://a.com/de/nur-deutsch/"] == "kein Treffer"
    assert conf["https://a.com/de/produkt-1/"] == "sicher"
    unmatched = output.unmatched_table(result, sets)
    assert ("de", "https://a.com/de/geloescht/", "Status Code 404") in set(map(tuple, unmatched.values.tolist()))
    html = output.html_blocks(clusters, "de", "de", "language", include_review=False)
    assert html.count('hreflang="x-default"') == 4   # 2 Cluster × 2 Mitglieder
    assert 'href="https://a.com/fr/produit-1/"' in html
