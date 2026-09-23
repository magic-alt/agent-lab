import numpy as np

from agent_lab.experiments.phase01_transformer import causal_mask, scaled_dot_product_attention


def test_causal_attention_blocks_future_tokens():
    rng = np.random.default_rng(7)
    q = rng.normal(size=(6, 4))
    k = rng.normal(size=(6, 4))
    v = rng.normal(size=(6, 3))
    out, weights, _ = scaled_dot_product_attention(q, k, v, causal=True)

    assert out.shape == (6, 3)
    assert weights.shape == (6, 6)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights[causal_mask(6)] == 0.0)
