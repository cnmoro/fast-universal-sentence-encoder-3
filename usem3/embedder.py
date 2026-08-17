"""USE multilingual v3 — fast, dependency-light sentence embeddings.

Pure numpy forward pass (float32 inference, no onnxruntime). Reproduces the
original MiniVectorDB ONNX model at cosine >= 0.99 and returns L2-normalized
512-dim vectors.
"""

import os

import numpy as np

from .tokenizer import UseTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, "resources")


class USE:
    def __init__(self, threads=4, denoise=False, denoise_fn=None):
        """USE multilingual v3 sentence embedder (pure numpy).

        threads: BLAS threads used for the numpy forward pass (via
        ``threadpoolctl``; larger batches benefit from more threads).
        denoise: if True, normalize number tokens to a placeholder before
        embedding (see preprocess.denoise_numbers). OFF by default because it
        slightly changes embeddings; enable it when your texts may contain the
        same content with different quantities (e.g. search over documents).
        denoise_fn: optional custom callable text->text to use instead.
        """
        self.threads = threads
        self.denoise = bool(denoise)
        self._denoise_fn = denoise_fn
        self._backend = None
        self._sp = None

    # ---------- lazy resources ----------
    @property
    def _tokenizer(self):
        if self._sp is None:
            self._sp = UseTokenizer(os.path.join(_RES, "sp.model"))
        return self._sp

    @property
    def _model(self):
        if self._backend is None:
            from .backends import NumpyBackend

            self._backend = NumpyBackend(
                os.path.join(_RES, "weights.npz"), self._tokenizer, threads=self.threads)
        return self._backend

    def _preprocess(self, texts):
        if not self.denoise:
            return texts
        if self._denoise_fn is not None:
            fn = self._denoise_fn
        else:
            from .preprocess import denoise_numbers

            fn = denoise_numbers
        return [fn(t) for t in texts]

    # ---------- public API ----------
    def encode(self, texts):
        """Embed one string -> (512,) or a list of strings -> (N, 512).

        If ``denoise=True`` was set at construction, numbers are normalized
        first (see preprocess.denoise_numbers).
        """
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if self.denoise:
            batch = self._preprocess(batch)
        out = self._model.encode(batch)
        return out[0] if single else out

    def __call__(self, texts):
        return self.encode(texts)

    def similarity(self, a, b):
        """Cosine similarity between two texts or two lists of texts."""
        ea, eb = self.encode(a), self.encode(b)
        ea = ea[None, :] if ea.ndim == 1 else ea
        eb = eb[None, :] if eb.ndim == 1 else eb
        return np.clip(ea @ eb.T / (np.linalg.norm(ea, axis=1, keepdims=True)
                                    * np.linalg.norm(eb, axis=1, keepdims=True)), -1.0, 1.0)


__all__ = ["USE"]