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

`encode` runs a whole list as one flat batch, which amortizes the weight
streaming across sentences: passing a list is ~3.5× faster than calling `encode`
once per sentence.

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
- **Size**: the package is ~33 MB (the embedding table is stored 6-bit-quantized
  per chunk, which halves it with no measurable quality loss).

## PT-BR benchmark

Three lightweight PT-BR tasks (all data fits in memory, seconds to run):

- **ASSIN2 STS** — semantic similarity over the 2,448 test pairs
  (`nilc-nlp/assin2`); Spearman/Pearson between cosine and gold relatedness.
- **ASSIN2 entailment** — discriminating entailed pairs from the rest; ROC AUC
  of the cosine score against the binary `entailment_judgment`.
- **Topical retrieval probe** — 27 documents across 9 topics, 8 queries with
  manual relevance judgments; mean nDCG@3 and recall@3.

Quality scores were computed on a GPU; **speed is single-CPU-core** — one BLAS
thread for this package, `torch.set_num_threads(1)` for the baselines, no GPU —
encoding the 500 ASSIN2 premises as one list. RAM is the peak RSS of a clean
process (package + numpy only, no benchmark harness). All speed and RAM figures
come from the same machine and session. `nomic-embed-text-v2-moe` uses its
`search_document:`/`search_query:` prefixes for retrieval.

| Model | Spearman | Pearson | AUC | nDCG@3 | R@3 | sent/s (1 core) | Peak RAM |
|---|---|---|---|---|---|---|---|
| **fast-universal-sentence-encoder (USE v3)** | 0.691 | 0.749 | 0.754 | 0.535 | **0.562** | **1470** | **151 MB** |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.715 | 0.772 | 0.831 | 0.625 | 0.438 | 105 | 1.5 GB |
| bge-m3 | **0.774** | **0.805** | **0.869** | 0.585 | 0.375 | 8 | 2.3 GB |
| nomic-embed-text-v2-moe | 0.684 | 0.723 | 0.763 | **0.633** | 0.500 | 12 | 5.1 GB |

`fast-universal-sentence-encoder` trades a bit of top-end quality (bge-m3 wins
on STS and entailment) for 14× (vs MiniLM) to ~180× (vs bge-m3) higher
single-core throughput and 10-34× less memory than the transformer baselines —
a fit for CPU-only and low-RAM serving. More BLAS threads still help, since the
forward pass runs as batched GEMMs: with the default `threads=4` it reaches
~2500 sent/s on the same machine. The 151 MB peak is the 6-bit embedding table
(65 MB), the 128k-piece vocabulary of the pure-python tokenizer (~45 MB) and
numpy itself; the rest of the encoder weights are ~11 MB. On number-heavy text,
`USE(denoise=True)` typically narrows the STS gap.

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
