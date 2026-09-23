import numpy as np

from agent_lab.experiments.phase02_tokens_sampling import entropy_bits, probabilities


def test_temperature_increases_entropy_for_fixed_logits():
    logits = np.array([3.0, 1.0, 0.0, -1.0])
    cold = probabilities(logits, 0.2)
    hot = probabilities(logits, 1.2)
    assert np.isclose(cold.sum(), 1.0)
    assert np.isclose(hot.sum(), 1.0)
    assert entropy_bits(cold) < entropy_bits(hot)
