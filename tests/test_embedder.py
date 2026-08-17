import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from usem3 import USE


def test_basic_encode():
    use = USE()
    texts = ["o gato preto correu pelo jardim", "investir em renda fixa é seguro"]
    va = use.encode(texts)
    assert va.shape == (2, 512)
    assert np.allclose(np.linalg.norm(va, axis=1), 1.0, atol=1e-4)


def test_denoise_on_backend():
    # same table loaded twice (fresh) must be deterministic and consistent
    a = USE()
    b = USE()
    texts = ["o gato preto correu pelo jardim"]
    assert np.allclose(a.encode(texts), b.encode(texts), atol=1e-6)


def test_single_and_batch():
    use = USE()
    single = use.encode("uma frase")
    batch = use.encode(["uma frase"])
    assert single.shape == (512,)
    assert batch.shape == (1, 512)
    assert np.allclose(single, batch[0], atol=1e-6)


def test_normalized():
    use = USE()
    v = use.encode("o gato preto correu pelo jardim")
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-4)


def test_similarity_shape():
    use = USE()
    sim = use.similarity("a", "b")
    assert sim.shape == (1, 1)
    assert -1.0 <= sim[0, 0] <= 1.0


def test_callable():
    use = USE()
    assert use("uma frase").shape == (512,)


def test_denoise_normalizes_numbers():
    from usem3 import USE
    import numpy as np

    raw = USE()
    den = USE(denoise=True)
    a, b = "O projeto custou 1 milhão de reais.", "O projeto custou 5 milhões de reais."
    va = raw.encode([a, b])
    vb = den.encode([a, b])

    def c(v):
        return float(np.dot(v[0], v[1]) / (np.linalg.norm(v[0]) * np.linalg.norm(v[1])))

    assert c(vb) > c(va), f"denoise should raise similarity: {c(va):.3f} -> {c(vb):.3f}"
    # clean text unaffected
    clean = "o gato preto correu pelo jardim"
    v0 = raw.encode(clean)
    v1 = den.encode(clean)
    assert float(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))) > 0.999


def test_batching_does_not_change_embeddings():
    """The flat micro-batch must give each sentence its own embedding."""
    from usem3.backends.numpy import _TOKEN_BUDGET

    use = USE()
    texts = [
        "o gato preto correu pelo jardim",
        "x",
        "investir em renda fixa é seguro mas o retorno é baixo",
        "the quick brown fox jumps over the lazy dog",
        "東京の天気は明日は雨になるでしょう",
        "",
        "  espaços    irregulares\te tabs\n",
        # more tokens than one micro-batch holds, so it forms a batch of its own
        "palavra " * (_TOKEN_BUDGET * 2),
    ]
    one = np.stack([use.encode(t) for t in texts])
    batched = use.encode(texts)
    assert np.allclose(one, batched, atol=1e-5)

    # order must not matter either
    order = [3, 0, 7, 5, 2, 6, 1, 4]
    shuffled = use.encode([texts[i] for i in order])
    assert np.allclose(shuffled, batched[order], atol=1e-5)


def test_many_short_texts_cross_micro_batches():
    use = USE()
    texts = ["frase número %d" % i for i in range(200)]
    vecs = use.encode(texts)
    assert vecs.shape == (200, 512)
    assert vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert np.allclose(vecs[7], use.encode(texts[7]), atol=1e-5)


def test_tokenizer_prefix_walk():
    """The flat prefix table must expose exactly the vocabulary pieces."""
    use = USE()
    tok = use._tokenizer
    s = tok._normalizer.normalize("o gato preto correu")
    for i in range(len(s)):
        for length, pid in tok._trie_prefixes(s, i):
            assert tok._pieces[pid] == s[i:i + length]
    ids = tok.encode("o gato preto correu", out_type=int, add_bos=True, add_eos=True)
    pieces = tok.encode("o gato preto correu", out_type=str, add_bos=False, add_eos=False)
    assert ids[0] == 1 and ids[-1] == 2
    assert [tok._pieces[i] for i in ids[1:-1]] == pieces
    assert "".join(pieces) == s


if __name__ == "__main__":
    test_basic_encode()
    test_denoise_on_backend()
    test_single_and_batch()
    test_normalized()
    test_similarity_shape()
    test_callable()
    test_denoise_normalizes_numbers()
    test_batching_does_not_change_embeddings()
    test_many_short_texts_cross_micro_batches()
    test_tokenizer_prefix_walk()
    print("all tests passed")
