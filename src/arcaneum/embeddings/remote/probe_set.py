"""Fixed texts deploy embeds remotely and locally to check vector parity (RDR-025).

The markdown-emphasis chunk guards against tokenizer differences between
dependency versions: it produced the one outlier in the Ollama spike.
"""

PROBE_TEXTS = (
    "Remote embedding workers run arcaneum's own CPU embedding path in Docker.",
    "Vectors from a worker must match the workstation's within float tolerance.",
    "def merge(left, right):\n    return sorted(left + right)\n",
    "The handshake carries the protocol version, arcaneum version and code fingerprint.",
    "Schema evolution changes a table's structure while preserving its data.",
    "A chunk ending in markdown emphasis that once tokenized differently: **bold claim**",
    "Unicode: café, naïve, Ελληνικά, 日本語, emoji 🚀 and math ∑ x² ≤ ∞.",
    "Short.",
)
