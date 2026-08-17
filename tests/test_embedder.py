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


if __name__ == "__main__":
    test_basic_encode()
    test_denoise_on_backend()
    test_single_and_batch()
    test_normalized()
    test_similarity_shape()
    test_callable()
    test_denoise_normalizes_numbers()
    print("all tests passed")
