"""Compatibility shim for transformers DynamicCache breaking change (RDR-023).

transformers 4.54.0 removed Cache.get_usable_length(), which Stella's custom
modeling_qwen.py calls in 4 places. This shim restores the method using
get_seq_length() and get_max_cache_shape(), which remain available through
transformers 5.5.3.

Spike result: bit-identical embeddings (np.allclose(atol=1e-7) = True).

Deprecation path: Remove this shim when either:
  (a) Stella upstream fixes modeling_qwen.py
  (b) arcaneum switches to the community fork (it-just-works/stella_en_1.5B_v5_bf16)
  (c) arcaneum switches embedding models

See: docs/rdr/RDR-023-advanced-pdf-integration.md
See: https://huggingface.co/dunzhang/stella_en_1.5B_v5/discussions/47
"""

from transformers.cache_utils import Cache

if not hasattr(Cache, "get_usable_length"):

    def _get_usable_length(self, new_seq_length, layer_idx=0):
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    Cache.get_usable_length = _get_usable_length
