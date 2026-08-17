"""Quickstart for usem3."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from usem3 import USE

use = USE()

texts = [
    "o gato preto correu pelo jardim",
    "a menina lê um livro",
    "investir em renda fixa é seguro",
    "O desmatamento da Amazônia contribui para o aumento das emissões de carbono.",
]

vecs = use.encode(texts)
print("shape:", vecs.shape)  # (4, 512)

sim = use.similarity(texts, texts)
print("similarity matrix (4x4):")
for row in sim:
    print("  " + " ".join(f"{v:.2f}" for v in row))

# single string -> 1-d vector
print("single:", use.encode("só uma frase").shape)