from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    input: str
    expected_contains: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


def load_jsonl(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cases.append(
                EvalCase(
                    case_id=row.get("case_id", f"line-{line_no}"),
                    input=row["input"],
                    expected_contains=tuple(row.get("expected_contains", [])),
                    metadata=row.get("metadata"),
                )
            )
    return cases


def contains_all(output: str, expected: Iterable[str]) -> bool:
    lowered = output.lower()
    return all(term.lower() in lowered for term in expected)
