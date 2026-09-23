from __future__ import annotations

from agent_lab.context import ContextItem, ContextPolicy


def demo_items() -> list[ContextItem]:
    return [
        ContextItem(
            "question",
            "Question: Why can EtherCAT DC jitter hurt CSP control?",
            100,
            "user",
        ),
        ContextItem(
            "evidence",
            "Evidence: timing jitter perturbs when cyclic setpoints arrive.",
            80,
            "notes",
        ),
        ContextItem(
            "noise",
            "Unrelated: image classification uses convolution filters.",
            5,
            "old-note",
        ),
        ContextItem(
            "constraint",
            "Constraint: distinguish bus timing from servo-loop bandwidth.",
            90,
            "spec",
        ),
    ]


def run() -> None:
    policy = ContextPolicy(budget_chars=190)
    packed = policy.pack(demo_items())
    print(packed.text)
    print(f"selected={packed.selected_ids}")
    print(f"dropped={packed.dropped_ids}")
    print(f"truncated={packed.truncated_ids}")
    print(f"budget={packed.used_chars}/{packed.budget_chars} chars")
