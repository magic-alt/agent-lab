from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextItem:
    item_id: str
    text: str
    priority: int = 0
    source: str = "unknown"


@dataclass(frozen=True)
class PackedContext:
    text: str
    selected_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    truncated_ids: tuple[str, ...]
    used_chars: int
    budget_chars: int


class ContextPolicy:
    """Deterministic priority-first context packer with a hard character budget.

    If the next highest-priority item does not fully fit, it is truncated and packing stops.
    This prevents small low-priority noise from backfilling space ahead of more important evidence.
    """

    def __init__(
        self,
        budget_chars: int = 6000,
        separator: str = "\n\n---\n\n",
        min_truncated_chars: int = 24,
    ) -> None:
        if budget_chars <= 0:
            raise ValueError("budget_chars must be positive")
        if min_truncated_chars <= 0:
            raise ValueError("min_truncated_chars must be positive")
        self.budget_chars = budget_chars
        self.separator = separator
        self.min_truncated_chars = min_truncated_chars

    def pack(self, items: list[ContextItem]) -> PackedContext:
        indexed = list(enumerate(items))
        ranked = sorted(indexed, key=lambda pair: (-pair[1].priority, pair[0]))

        selected_indexes: list[int] = []
        selected_chunks: dict[int, str] = {}
        truncated: set[int] = set()
        used = 0
        for idx, item in ranked:
            prefix_cost = len(self.separator) if selected_indexes else 0
            remaining = self.budget_chars - used - prefix_cost
            if remaining <= 0:
                break
            if len(item.text) <= remaining:
                selected_indexes.append(idx)
                selected_chunks[idx] = item.text
                used += prefix_cost + len(item.text)
                continue
            if remaining >= self.min_truncated_chars:
                chunk = item.text[:remaining]
                if remaining >= 2:
                    chunk = item.text[: remaining - 1] + "…"
                selected_indexes.append(idx)
                selected_chunks[idx] = chunk
                truncated.add(idx)
            break

        selected_indexes.sort()
        text = self.separator.join(selected_chunks[i] for i in selected_indexes)
        selected = set(selected_indexes)
        dropped = tuple(item.item_id for i, item in indexed if i not in selected)
        return PackedContext(
            text=text,
            selected_ids=tuple(items[i].item_id for i in selected_indexes),
            dropped_ids=dropped,
            truncated_ids=tuple(items[i].item_id for i in selected_indexes if i in truncated),
            used_chars=len(text),
            budget_chars=self.budget_chars,
        )
