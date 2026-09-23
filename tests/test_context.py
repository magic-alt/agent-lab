from agent_lab.context import ContextItem, ContextPolicy


def test_context_packer_prefers_priority_and_respects_budget():
    items = [
        ContextItem("low", "L" * 50, priority=1),
        ContextItem("high", "H" * 50, priority=100),
        ContextItem("medium", "M" * 50, priority=50),
    ]
    result = ContextPolicy(budget_chars=110, separator="|").pack(items)
    assert result.used_chars <= 110
    assert "high" in result.selected_ids
    assert "medium" in result.selected_ids
    assert "low" in result.dropped_ids
    # Final presentation preserves the source order of selected items.
    assert result.text == ("H" * 50) + "|" + ("M" * 50)
