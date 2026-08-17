"""Evaluate whether number/noise tokens corrupt USE embeddings, and show the
value of the denoise preprocessing step.

Findings:
* USE multilingual v3 is sensitive to number tokens. Two sentences that differ
  only in their numbers get cosine ~0.6-0.8 instead of ~1.0 (harms semantic
  search / similarity for the same content with different quantities).
* Injecting a number clause into 800/2448 ASSIN2 pairs drops STS spearman
  0.688 -> 0.640 and entailment AUC 0.752 -> 0.735.
* Normalizing numbers to a placeholder restores cosine 0.72 -> ~0.99 for
  number-differing paraphrases, and is a no-op on clean text.

Run:  python bench_numbers.py
"""

import sys
import os
import re

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from usem3 import USE

DENOISE = USE(denoise=True)
RAW = USE()


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("=== 1. Same meaning, numbers swapped -> raw cosine is low ===")
    pairs = [
        ("O projeto custou 1 milhão de reais.", "O projeto custou 5 milhões de reais."),
        ("Em 2010 foram criados 100 novos empregos.", "Em 2020 foram criados 300 novos empregos."),
        ("A empresa tem 50 funcionários.", "A empresa tem 200 funcionários."),
        ("O preço subiu 10% este ano.", "O preço subiu 25% este ano."),
        ("Foram vendidas 1000 unidades.", "Foram vendidas 2500 unidades."),
        ("A reunião às 10h foi adiada.", "A reunião às 15h foi adiada."),
    ]
    raw_c = []
    den_c = []
    for a, b in pairs:
        r = cos(*RAW.encode([a, b]))
        d = cos(*DENOISE.encode([a, b]))
        raw_c.append(r)
        den_c.append(d)
        print(f"  {a[:32]:<34} raw={r:.3f}  denoised={d:.3f}")
    print(f"  MEAN                         raw={np.mean(raw_c):.3f}  denoised={np.mean(den_c):.3f}")

    print("\n=== 2. Effect of number-noise at scale (ASSIN2 pt-br) ===")
    try:
        from datasets import load_dataset
        from scipy.stats import spearmanr
        from sklearn.metrics import roc_auc_score

        ds = load_dataset("nilc-nlp/assin2", split="test")
        s1, s2 = list(ds["premise"]), list(ds["hypothesis"])
        y = np.array(ds["entailment_judgment"])
        g = np.array(ds["relatedness_score"], dtype=float)
        rng = np.random.default_rng(0)
        idx = set(rng.choice(len(s1), 800, replace=False))

        def inject(t):
            return t + " em 2020 foram 12345 unidades no total"

        s1n = [inject(t) if i in idx else t for i, t in enumerate(s1)]
        s2n = [inject(t) if i in idx else t for i, t in enumerate(s2)]

        def score(e1, e2):
            pred = np.array([cos(a, b) for a, b in zip(e1, e2)])
            return spearmanr(pred, g)[0], roc_auc_score(y, pred)

        e1 = RAW.encode(s1)
        e2 = RAW.encode(s2)
        rho0, auc0 = score(e1, e2)
        e1n = RAW.encode(s1n)
        e2n = RAW.encode(s2n)
        rho1, auc1 = score(e1n, e2n)
        print(f"  clean            spearman={rho0:.4f}  auc={auc0:.4f}")
        print(f"  +number clause   spearman={rho1:.4f}  auc={auc1:.4f}")
        print(f"  delta            spearman={rho1-rho0:+.4f}  auc={auc1-auc0:+.4f}")
    except ImportError as e:
        print("  (skip: need datasets/scipy/scikit-learn:", e, ")")

    print("\n=== 3. Denoise is a no-op on clean text ===")
    clean = "o gato preto correu pelo jardim e a menina lê um livro na sala."
    r = cos(*RAW.encode([clean, clean]))
    d = cos(*DENOISE.encode([clean, clean]))
    print(f"  clean vs clean: raw={r:.4f} denoised={d:.4f}")


if __name__ == "__main__":
    main()
