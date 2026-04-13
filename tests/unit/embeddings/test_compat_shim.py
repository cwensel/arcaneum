"""Unit tests for DynamicCache compatibility shim (RDR-023)."""

import pytest


class TestDynamicCacheShim:
    """Verify the compatibility shim restores Cache.get_usable_length()."""

    def test_shim_restores_get_usable_length(self):
        """Cache.get_usable_length should exist after shim import."""
        import arcaneum.embeddings._compat  # noqa: F401
        from transformers.cache_utils import Cache

        assert hasattr(Cache, "get_usable_length"), (
            "Cache.get_usable_length not found — shim did not apply"
        )

    def test_shim_is_callable(self):
        """The restored method should be callable."""
        import arcaneum.embeddings._compat  # noqa: F401
        from transformers.cache_utils import Cache

        assert callable(getattr(Cache, "get_usable_length", None))

    def test_shim_idempotent(self):
        """Importing the shim multiple times should not break anything."""
        import arcaneum.embeddings._compat  # noqa: F401
        import importlib
        importlib.reload(arcaneum.embeddings._compat)

        from transformers.cache_utils import Cache
        assert hasattr(Cache, "get_usable_length")
