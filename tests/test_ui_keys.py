import hashlib

import pytest

from hreflang_matcher.ui_keys import MODE_FILE, MODE_GROUP, source_key


class TestSourceKey:
    def test_file_mode_stable_for_same_inputs(self):
        assert source_key(MODE_FILE, "abc", b"x", None) == source_key(MODE_FILE, "abc", b"x", None) == "file_abc"

    def test_file_mode_different_ids_give_different_keys(self):
        assert source_key(MODE_FILE, "id-1", b"same", None) != source_key(MODE_FILE, "id-2", b"same", None)

    def test_file_mode_independent_of_list_position(self):
        # Entfernt man die erste Datei, behalten die übrigen ihre Keys.
        ids = ["my", "de", "fr"]
        before = {i: source_key(MODE_FILE, i, b"", None) for i in ids}
        after = {i: source_key(MODE_FILE, i, b"", None) for i in ids[1:]}
        assert all(after[i] == before[i] for i in after)

    def test_file_mode_falls_back_to_content_hash(self):
        raw = b"Address;Embedding\n"
        assert source_key(MODE_FILE, None, raw, None) == f"file_{hashlib.md5(raw).hexdigest()[:12]}"
        assert source_key(MODE_FILE, "", raw, None) == source_key(MODE_FILE, None, raw, None)
        assert source_key(MODE_FILE, None, b"other", None) != source_key(MODE_FILE, None, raw, None)

    def test_group_mode_keyed_by_code(self):
        assert source_key(MODE_GROUP, "fid", b"raw", "de-AT") == "grp_de-AT"
        assert source_key(MODE_GROUP, "fid", b"raw", "de") != source_key(MODE_GROUP, "fid", b"raw", "fr")
        # unabhängig von Datei-ID und Inhalt
        assert source_key(MODE_GROUP, "other", b"x", "de") == "grp_de"

    def test_group_mode_requires_code(self):
        with pytest.raises(ValueError):
            source_key(MODE_GROUP, "fid", b"raw", None)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            source_key("xyz", "fid", b"raw", "de")
