from __future__ import annotations

import math

import numpy as np


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("Install the optional tokenizer dependency: pip install -e '.[tokens]'") from exc
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for probabilistic sampling")
    x = logits.astype(float) / temperature
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / exp.sum()


def entropy_bits(probs: np.ndarray) -> float:
    safe = probs[probs > 0]
    return float(-(safe * np.log2(safe)).sum())


def sample_index(logits: np.ndarray, temperature: float, seed: int = 42) -> int:
    probs = probabilities(logits, temperature)
    rng = np.random.default_rng(seed)
    return int(rng.choice(len(logits), p=probs))


def run(text: str) -> None:
    print(f"text chars: {len(text)}")
    try:
        print(f"tokens(o200k_base): {count_tokens(text)}")
    except RuntimeError as exc:
        print(f"tokenizer: {exc}")

    logits = np.array([2.4, 1.8, 0.7, -0.5])
    for temp in (0.2, 0.7, 1.2):
        probs = probabilities(logits, temp)
        print(
            f"temperature={temp}: probs={np.round(probs, 4)} "
            f"entropy={entropy_bits(probs):.3f} bits sample={sample_index(logits, temp)}"
        )
