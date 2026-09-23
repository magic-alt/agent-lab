from __future__ import annotations

import math

import numpy as np


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def causal_mask(seq_len: int) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, *, causal: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("Phase-01 demo expects rank-2 [seq, dim] tensors")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0] or q.shape[0] != k.shape[0]:
        raise ValueError("incompatible Q/K/V shapes")
    scores = q @ k.T / math.sqrt(q.shape[1])
    if causal:
        scores = np.where(causal_mask(q.shape[0]), -np.inf, scores)
    weights = stable_softmax(scores, axis=-1)
    output = weights @ v
    return output, weights, scores


def run() -> None:
    rng = np.random.default_rng(42)
    seq_len, d_model = 5, 8
    q = rng.normal(size=(seq_len, d_model))
    k = rng.normal(size=(seq_len, d_model))
    v = rng.normal(size=(seq_len, d_model))
    output, weights, scores = scaled_dot_product_attention(q, k, v)

    future = causal_mask(seq_len)
    forbidden_mass = float(weights[future].sum())
    print(f"Q/K/V: {q.shape} / {k.shape} / {v.shape}")
    print(f"scores: {scores.shape}, weights: {weights.shape}, output: {output.shape}")
    print(f"row sums: {np.round(weights.sum(axis=-1), 6)}")
    print(f"future attention mass: {forbidden_mass:.6g}")
