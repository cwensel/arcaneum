"""Tests for the transformers DynamicCache compatibility shim (RDR/dep-notes:
transformers 4.54 removed DynamicCache.get_usable_length, which Stella's pinned
remote code still calls)."""

from arcaneum.embeddings.client import _ensure_dynamic_cache_compat


class FakeCacheWithoutUsableLength:
    def get_seq_length(self, layer_idx=0):
        return 7 + layer_idx


class FakeCacheWithUsableLength:
    def get_seq_length(self, layer_idx=0):
        return 7

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return 42


class TestEnsureDynamicCacheCompat:
    def test_adds_get_usable_length_when_missing(self):
        _ensure_dynamic_cache_compat(cache_cls=FakeCacheWithoutUsableLength)

        cache = FakeCacheWithoutUsableLength()
        assert cache.get_usable_length(100) == 7
        assert cache.get_usable_length(100, layer_idx=3) == 10

    def test_preserves_existing_get_usable_length(self):
        _ensure_dynamic_cache_compat(cache_cls=FakeCacheWithUsableLength)

        cache = FakeCacheWithUsableLength()
        assert cache.get_usable_length(100) == 42

    def test_defaults_to_transformers_dynamic_cache(self):
        # With the real transformers installed, the call must not raise and the
        # class must end up with a working get_usable_length either way.
        _ensure_dynamic_cache_compat()

        from transformers.cache_utils import DynamicCache

        assert hasattr(DynamicCache, "get_usable_length")
