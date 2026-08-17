"""fast-universal-sentence-encoder — USE multilingual v3 sentence embeddings (pure numpy)."""

from .embedder import USE

try:
    from importlib.metadata import version as _version

    __version__ = _version("fast-universal-sentence-encoder")
except Exception:  # running from source without an installed distribution
    __version__ = "0.0.0.dev0"

__all__ = ["USE"]
