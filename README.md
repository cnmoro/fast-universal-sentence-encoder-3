# fast-universal-sentence-encoder

Fast, dependency-light implementation of **Google's universal-sentence-encoder-multilingual v3**.
Same SentencePiece (128k) vocabulary, same DAN + CNN n-gram encoder, same 512-dim L2-normalized output —
reproducing the original quantized ONNX at **cosine ≥ 0.99** — with **no C++ dependencies**:
a pure-Python port of the normalizer + unigram Viterbi and a pure numpy forward pass.

## Install

```bash
pip install fast-universal-sentence-encoder
```

Only needs `numpy`. (Optionally `threadpoolctl` via `pip install "fast-universal-sentence-encoder[fast]"`
for faster small-batch matmuls.)

## Quickstart

```python
from usem3 import USE

use = USE()
vec = use.encode("o gato preto correu pelo jardim")   # (512,) L2-normalized
vecs = use.encode(["a menina lê um livro", "investir em renda fixa é seguro"])  # (2, 512)

use.similarity("gato preto correu", "o gato preto correu pelo jardim")
# 0.53
```

> The package is distributed as `fast-universal-sentence-encoder`; the import
> name is `usem3`.

## Releasing

Publishing is automatic via GitHub Actions (`.github/workflows/publish.yml`):
tag a commit with `v*` (e.g. `git tag v0.1.0 && git push --tags`) and the
workflow runs the test matrix on linux amd64/arm64, Windows x64, macOS Apple
Silicon and macOS Intel, then builds the sdist + universal wheel and uploads it
to PyPI. The version is taken from the tag.

Prerequisites once:

1. Create a PyPI API token
   (https://pypi.org/manage/account/token/), scoped to the project.
2. In the GitHub repo: **Settings → Environments → New environment** named
   `pypi`, and add the token as a secret named `PYPI_TOKEN`.

### Number denoising

USE embeddings are sensitive to number tokens: two sentences that differ only in
their quantities get cosine ~0.6-0.8 instead of ~1.0, which can hurt semantic
search when the same content appears with different numbers. Normalizing numbers
to a placeholder fixes this and is a no-op on clean text:

```python
use = USE(denoise=True)
use.similarity("O projeto custou 1 milhão", "O projeto custou 5 milhões")
# 0.97 (vs 0.80 without denoise)
```

`denoise=True` replaces number tokens (digits, separators, `%`) with a fixed
placeholder before embedding. It is OFF by default because it slightly changes
the embeddings; enable it when your corpus mixes quantities. See
`examples/bench_numbers.py` for the evaluation.

## Model

- **Architecture**: deep averaging network (DAN) with CNN n-gram features (orders 2/3/5)
  followed by a residual DNN, trained multi-task across 16 languages.
- **Output**: 512-dim, L2-normalized sentence embeddings.
- **Languages**: ar, de, en, es, fr, it, ja, ko, nl, pl, pt, ru, th, tr, zh, zh-TW.
- **Source weights**: `WiseIntelligence/universal-sentence-encoder-multilingual-3-onnx-quantized`
  (quantized from the original Google TF model), dequantized and repacked here.
- **Size**: the package is ~33 MB (the embedding table is stored 6-bit-quantized
  per chunk, which halves it with no measurable quality loss; ASSIN2 spearman
  0.6906 vs 0.6899 for the full float32 table).

## Benchmarks (pt-br, ASSIN2 STS test, 2448 pairs)

| implementation | sent/s (per-sentence) | ASSIN2 spearman |
|---|---|---|
| original int8 onnx (MiniVectorDB) | ~750 | 0.698 |
| usem3 numpy | ~430–500 | 0.69 |
| static-nomic-384-pten-v2 (reference) | 36 000 | 0.614 |

## Files

```
usem3/
├── resources/
│   ├── sp.model        # sentencepiece model (128k vocab)
│   └── weights.npz     # compact 6-bit-quantized weights
├── backends/numpy.py   # pure numpy forward pass
├── pure_tokenizer.py   # pure-Python sentencepiece port (normalizer + unigram)
├── preprocess.py       # optional number denoising
├── embedder.py         # USE front-end
└── tokenizer.py        # tokenizer front-end (wraps pure_tokenizer)
scripts/build_weights.py  # regenerate weights.npz from float32 weights
```

## License

Apache-2.0. The underlying model is Google's
[universal-sentence-encoder-multilingual-3](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3)
(Apache-2.0).
