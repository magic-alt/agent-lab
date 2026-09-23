from agent_lab.evals import contains_all


def test_contains_all_is_case_insensitive():
    assert contains_all("KV Cache reduces repeated computation", ["kv cache", "COMPUTATION"])
