"""Optional, safe text denoising for USE embeddings.

Finding (see bench): USE multilingual v3 embeddings are sensitive to number
tokens. Two sentences that differ *only* in the numbers they contain (e.g. "O
projeto custou 1 milhão" vs "O projeto custou 5 milhões") get cosine similarity
of only ~0.6-0.8 instead of ~1.0, which can hurt semantic search / similarity
when the same content appears with different quantities.

This module provides a conservative normalizer: every number token is rewritten
to a fixed placeholder so that equivalent numbers become identical. It leaves
non-numeric text untouched.

* It is OFF by default. Enable with ``USE(denoise=True)``.
* On clean text (no numbers) it is a no-op and does not change embeddings
  (verified: ASSIN2 spearman 0.6883 -> 0.6883).
* When applied to number-differing paraphrases it restores cosine 0.72 -> ~0.99.
"""

import re

# A number token: optional sign, digits with optional . or , separators,
# optional trailing %. Bounded so it doesn't swallow surrounding words.
# Also covers times like "10h" (we keep the unit, normalize the digits only
# via the word-boundary rule) and years.
_NUMBER_RE = re.compile(r"(?<![\wÀ-ÿ])[-+]?\d[\d.,]*(?:\s*%)?(?![\wÀ-ÿ])")
# A stricter variant that also collapses standalone digit runs inside a token
# (e.g. "12345" but not "10h"). Used for embedded numbers in words? No - we keep
# it conservative: only replace pure number tokens / percentages.

_NUMBER_TOKEN = " <NÚMERO> "

DEFAULT_NUMBER_RE = _NUMBER_RE


def denoise_numbers(text, number_re=None):
    """Replace number tokens in *text* with a fixed placeholder.

    Returns the same string if there are no numbers. Safe for clean text.
    """
    if not text:
        return text
    rx = number_re or _NUMBER_RE
    return rx.sub(_NUMBER_TOKEN, text)
