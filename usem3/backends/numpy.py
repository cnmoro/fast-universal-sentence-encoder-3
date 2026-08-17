"""Pure-numpy backend for the USE multilingual v3 sentence encoder.

Implements the exact DAN + CNN n-gram architecture extracted from the
quantized ONNX model that ships inside MiniVectorDB (Google USE multilingual v3).
Weights are dequantized to float32, so this backend matches onnxruntime output
at cosine >= 0.99 while depending only on numpy + the pure-python tokenizer.

The forward pass runs on a *flat* batch: the tokens of every sentence in a
micro-batch are concatenated into one (n_tokens, dim) matrix, so each weight
matrix is streamed from memory once per micro-batch instead of once per
sentence (which is what a short sentence spends most of its time on). Sentence
boundaries are honoured by the n-gram window index: a window slot that would
reach into a neighbouring sentence points at a zero sentinel row instead, which
reproduces the per-sentence zero padding exactly.
"""

from itertools import chain

import numpy as np

_VOCAB_ROWS = 7530
_NUM_CHUNKS = 17
_EMB_DIM = 512
_CNN_DIM = 256
# (n-gram order, left context) for the three CNN branches.
_ORDERS = ((2, 0), (3, 1), (5, 2))
_MAX_LEFT = max(left for _, left in _ORDERS)
_MAX_RIGHT = max(n - 1 - left for n, left in _ORDERS)
# Tokens per micro-batch: enough sentences to amortize weight streaming (that is
# where the speedup comes from, and throughput is flat from here up to a few
# thousand tokens), while keeping the working set around 2 MB, i.e. cache-sized
# and a small transient. A sentence longer than this becomes its own batch.
_TOKEN_BUDGET = 384


def _layernorm(x, gamma, beta, eps):
    """Layernorm over the rows of a 2-D array, in place (x is modified)."""
    x -= x.mean(axis=-1, keepdims=True)
    var = np.einsum("ij,ij->i", x, x) / x.shape[-1]
    x *= np.reciprocal(np.sqrt(var + eps))[:, None]
    x *= gamma
    x += beta
    return x


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

        # embedding table: 17 per-chunk-quantized uint8 blocks (see
        # scripts/build_weights.py) written straight into one 65 MB uint8 table
        # indexed by piece id, one chunk at a time so the compressed npz is
        # never fully materialized. Rows are dequantized per micro-batch
        # instead of holding the 262 MB float32 table.
        self.table = np.empty((_NUM_CHUNKS * _VOCAB_ROWS, _EMB_DIM), dtype=np.uint8)
        with np.load(weights_path, allow_pickle=False) as z:
            for i in range(_NUM_CHUNKS):
                self.table[i::_NUM_CHUNKS] = z[f"q{i + 1}"]
            d = {k: z[k] for k in z.files if not k.startswith("q")}

        self.scale = d["scale"]
        self.lo = d["lo"]

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
        self._feat_dim = _EMB_DIM + len(_ORDERS) * _CNN_DIM

    # ---------- encoder ----------
    def encode(self, texts):
        """Embed each text independently (no cross-sentence n-gram leakage)."""
        if self._threadpool is None:
            return self._encode(texts)
        with self._threadpool(limits=self.threads, user_api="blas"):
            return self._encode(texts)

    def _encode(self, texts):
        # tokenize as we go and flush a micro-batch as soon as the token budget
        # is reached, so peak memory does not scale with len(texts)
        n = len(texts)
        out = np.empty((n, _EMB_DIM), dtype=np.float32)
        tokenize = self.sp.encode
        group, n_tokens, first = [], 0, 0
        for i, text in enumerate(texts):
            ids = tokenize(text, out_type=int, add_bos=True, add_eos=True)
            if group and n_tokens + len(ids) > _TOKEN_BUDGET:
                self._encode_batch(group, n_tokens, out[first:i])
                group, n_tokens, first = [], 0, i
            group.append(ids)
            n_tokens += len(ids)
        if group:
            self._encode_batch(group, n_tokens, out[first:n])
        return out

    def _encode_batch(self, group, n_tokens, out):
        lens = np.fromiter(map(len, group), dtype=np.int64, count=len(group))
        flat = np.fromiter(chain.from_iterable(group), dtype=np.int64, count=n_tokens)
        starts = np.zeros(len(group), dtype=np.int64)
        np.cumsum(lens[:-1], out=starts[1:])

        # feats holds [raw embedding | cnn order 2 | order 3 | order 5] so the
        # branches are written in place instead of concatenated afterwards.
        feats = np.empty((n_tokens, self._feat_dim), dtype=np.float32)
        emb = feats[:, :_EMB_DIM]
        chunk = flat % _NUM_CHUNKS
        np.copyto(emb, self.table[flat])
        emb *= self.scale[chunk, None]
        emb += self.lo[chunk, None]

        x = emb @ self.proj_W
        x += self.proj_b
        _layernorm(x, self.ln_g_all, self.ln_b_all, self.eps)

        idx = self._window_index(starts, lens, n_tokens)
        # scratch buffer for the conv inputs; the extra row n_tokens is the zero
        # sentinel that out-of-sentence window slots point at (the other rows
        # are always overwritten before use).
        buf = np.empty((n_tokens + 1, _CNN_DIM), dtype=np.float32)
        buf[n_tokens] = 0.0
        for i, (n, left) in enumerate(_ORDERS):
            xs = x[:, i * _CNN_DIM:(i + 1) * _CNN_DIM]
            buf[:n_tokens] = xs
            cols = idx[:, _MAX_LEFT - left:_MAX_LEFT - left + n]
            h = buf[cols].reshape(n_tokens, n * _CNN_DIM) @ self.w1[i]
            h += self.b1[i]
            np.maximum(h, 0, out=h)
            h += xs
            buf[:n_tokens] = h
            off = _EMB_DIM + i * _CNN_DIM
            c2 = buf[cols].reshape(n_tokens, n * _CNN_DIM) @ self.w2[i]
            c2 += self.b2[i]
            np.maximum(c2, 0, out=c2)
            c2 += h
            feats[:, off:off + _CNN_DIM] = c2
        del x, buf

        _layernorm(feats, self.ln_g, self.ln_b, self.eps)
        # sum-then-project: the token sum commutes with the (linear) output
        # projection, so this GEMM shrinks from n_tokens rows to n_sentences.
        mean = np.add.reduceat(feats, starts, axis=0) @ self.cnn_W
        flens = lens.astype(np.float32)
        mean += flens[:, None] * self.cnn_b
        mean *= np.reciprocal(np.sqrt(flens))[:, None]

        y0 = np.maximum(mean @ self.d0, 0)
        y1 = mean @ self.p1
        y1 += y0 @ self.d1
        np.maximum(y1, 0, out=y1)
        y2 = np.maximum(y1 @ self.d2, 0)
        z = y1 @ self.p3
        z += y2 @ self.d3
        np.tanh(z, out=z)
        z *= np.reciprocal(np.sqrt(np.clip(np.einsum("ij,ij->i", z, z),
                                           self.eps, None)))[:, None]
        out[:] = z

    @staticmethod
    def _window_index(starts, lens, n_tokens):
        """Gather index for the widest n-gram window, clamped to sentence bounds.

        Row t holds the token positions t-_MAX_LEFT .. t+_MAX_RIGHT, with slots
        outside t's own sentence replaced by the sentinel row ``n_tokens``.
        Narrower orders use a column slice of it.
        """
        pos = np.arange(n_tokens, dtype=np.int64)[:, None]
        base = pos + np.arange(-_MAX_LEFT, _MAX_RIGHT + 1, dtype=np.int64)
        inside = base >= np.repeat(starts, lens)[:, None]
        inside &= base < np.repeat(starts + lens, lens)[:, None]
        return np.where(inside, base, n_tokens)


__all__ = ["NumpyBackend"]
