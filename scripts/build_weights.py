"""Build the compact usem3 weights file from the original float32 weights.

The embedding table (128010 x 512) dominates the package size. It is the only
array with lots of information (the small matmul weights are tiny). We quantize
it per-chunk (each of the 17 interleaved chunks has its own [lo, scale]) to 6
bits, which cuts the package from ~66 MB to ~27 MB with no measurable quality
loss (ASSIN2 spearman 0.6906 vs 0.6899; embedding cosine agreement 0.989).

Usage:
    python scripts/build_weights.py \
        --input  bench/use_weights/weights.npz \
        --output usem3/resources/weights.npz \
        --bits 6
"""

import argparse
import os

import numpy as np

VOCAB_ROWS = 7530
NUM_CHUNKS = 17
EMB_DIM = 512


def quantize_chunks(table, bits):
    """Per-chunk uniform quantization -> (q_uint8_chunks, lo, scale)."""
    qs, los, scales = [], [], []
    for i in range(NUM_CHUNKS):
        chunk = table[i::NUM_CHUNKS]
        lo, hi = chunk.min(), chunk.max()
        scale = (hi - lo) / (2 ** bits - 1)
        q = np.clip(np.round((chunk - lo) / scale), 0, 2 ** bits - 1).astype(np.uint8)
        qs.append(q)
        los.append(lo)
        scales.append(scale)
    return qs, np.asarray(los, np.float32), np.asarray(scales, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="source float32 weights.npz")
    ap.add_argument("--output", required=True, help="destination compact weights.npz")
    ap.add_argument("--bits", type=int, default=6)
    args = ap.parse_args()

    d = dict(np.load(args.input, allow_pickle=False))

    table = np.empty((NUM_CHUNKS * VOCAB_ROWS, EMB_DIM), dtype=np.float32)
    for c in range(1, NUM_CHUNKS + 1):
        table[c - 1::NUM_CHUNKS] = d[f"emb_chunk{c}"]

    qs, lo, scale = quantize_chunks(table, args.bits)

    out = {}
    for i in range(NUM_CHUNKS):
        out[f"q{i + 1}"] = qs[i]
    out["lo"] = lo
    out["scale"] = scale
    out["bits"] = np.array([args.bits], dtype=np.int8)
    # copy the small float32 weights through unchanged
    for key in ("proj2", "proj_b2", "ln_g2", "ln_b2", "w1_2", "b1_2", "w2_2", "b2_2",
                "proj3", "proj_b3", "ln_g3", "ln_b3", "w1_3", "b1_3", "w2_3", "b2_3",
                "proj5", "proj_b5", "ln_g5", "ln_b5", "w1_5", "b1_5", "w2_5", "b2_5",
                "ln_g", "ln_b", "cnn_W", "cnn_b",
                "d0", "d1", "p1", "d2", "p3", "d3", "eps"):
        out[key] = d[key]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **out)
    size = os.path.getsize(args.output) / 1e6
    print(f"wrote {args.output}  ({size:.1f} MB, {args.bits}-bit embedding table)")


if __name__ == "__main__":
    main()
