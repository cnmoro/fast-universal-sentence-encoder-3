"""USE multilingual v3 tokenizer (pure python, no C++ sentencepiece).

This wraps the pure-python sentencepiece port (normalizer + unigram Viterbi)
in the same interface the backends expect. The tokenizer output is byte-identical
to the reference ``sentencepiece`` library for the bundled model.
"""

from .pure_tokenizer import UsePureTokenizer

UseTokenizer = UsePureTokenizer

__all__ = ["UseTokenizer"]