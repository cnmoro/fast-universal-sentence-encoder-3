"""Pure-numpy backend for the USE multilingual v3 sentence encoder.

Implements the exact DAN + CNN n-gram architecture extracted from the
quantized ONNX model that ships inside MiniVectorDB (Google USE multilingual v3).
Weights are dequantized to float32, so this backend matches onnxruntime output
at cosine >= 0.99 while depending only on numpy + the pure-python tokenizer.
"""

import numpy as np

_VOCAB_ROWS = 7530
_NUM_CHUNKS = 17


class NumpyBackend:
    name = "numpy"

    def __init__(self, weights_path, sp_processor, threads=4):
        self.sp = sp_processor
        self.threads = threads
        try:
            from threadpoolctl import threadpool_limits

            self._threadpool = threadpool_limits
        except Exception:
            self._threadpool = None

        d = dict(np.load(weights_path, allow_pickle=False))

        # embedding table: 17 per-chunk-quantized uint8 blocks (see scripts/build_weights.py)
        table = np.empty((_NUM_CHUNKS * _VOCAB_ROWS, 512), dtype=np.float32)
        lo = d["lo"]
        scale = d["scale"]
        for i in range(_NUM_CHUNKS):
            table[i::_NUM_CHUNKS] = d[f"q{i + 1}"].astype(np.float32) * scale[i] + lo[i]
        self.table = table

        self.proj_W = np.concatenate([d[f"proj{n}"] for n in (2, 3, 5)], axis=1)
        self.proj_b = np.concatenate([d[f"proj_b{n}"] for n in (2, 3, 5)])
        self.ln_g_all = np.concatenate([d[f"ln_g{n}"] for n in (2, 3, 5)])
        self.ln_b_all = np.concatenate([d[f"ln_b{n}"] for n in (2, 3, 5)])
        self.w1 = [d[f"w1_{n}"] for n in (2, 3, 5)]
        self.b1 = [d[f"b1_{n}"] for n in (2, 3, 5)]
        self.w2 = [d[f"w2_{n}"] for n in (2, 3, 5)]
        self.b2 = [d[f"b2_{n}"] for n in (2, 3, 5)]
        self.ln_g, self.ln_b = d["ln_g"], d["ln_b"]
        self.cnn_W, self.cnn_b = d["cnn_W"], d["cnn_b"]
        self.d0, self.d1, self.d2, self.d3 = d["d0"], d["d1"], d["d2"], d["d3"]
        self.p1, self.p3 = d["p1"], d["p3"]
        self.eps = float(d["eps"])

    # ---------- primitives ----------
    @staticmethod
    def _layernorm(x, gamma, beta, eps):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mu) * np.reciprocal(np.sqrt(var + eps)) * gamma + beta

    @staticmethod
    def _windows(x, n, left):
        T, D = x.shape
        right = n - 1 - left
        xp = np.pad(x, ((left, right), (0, 0)))
        idx = np.arange(T)[:, None] + np.arange(n)[None, :]
        return xp[idx].reshape(T, n * D)

    # ---------- encoder ----------
    def _cnn_order(self, x, order_idx):
        n, left = ((2, 0), (3, 1), (5, 2))[order_idx]
        c1 = np.maximum(self._windows(x, n, left) @ self.w1[order_idx] + self.b1[order_idx], 0)
        h = x + c1
        c2 = np.maximum(self._windows(h, n, left) @ self.w2[order_idx] + self.b2[order_idx], 0)
        return h + c2

    def encode(self, texts):
        """Embed each text independently (no cross-sentence n-gram leakage)."""
        if self._threadpool is None:
            return np.stack([self._encode_one(t) for t in texts])
        with self._threadpool(limits=self.threads, user_api="blas"):
            return np.stack([self._encode_one(t) for t in texts])

    def _encode_one(self, text):
        ids = self.sp.encode(text, out_type=int, add_bos=True, add_eos=True)
        emb = self.table[np.array(ids, dtype=np.int64)]

        x = self._layernorm(emb @ self.proj_W + self.proj_b, self.ln_g_all, self.ln_b_all, self.eps)
        feats = [emb]
        for i, off in enumerate((0, 256, 512)):
            feats.append(self._cnn_order(x[:, off:off + 256], i))
        cnn = np.concatenate(feats, axis=1)
        cnn = self._layernorm(cnn, self.ln_g, self.ln_b, self.eps)
        mean = (cnn @ self.cnn_W + self.cnn_b).sum(axis=0) / np.sqrt(len(ids))

        y0 = np.maximum(mean @ self.d0, 0)
        y1 = np.maximum(y0 @ self.d1 + mean @ self.p1, 0)
        y2 = np.maximum(y1 @ self.d2, 0)
        z = np.tanh(y2 @ self.d3 + y1 @ self.p3)
        norm = np.sqrt(np.clip((z ** 2).sum(), self.eps, None))
        return z / norm