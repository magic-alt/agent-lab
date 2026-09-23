from __future__ import annotations

import ast
import operator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .registry import ToolDefinition, ToolRegistry


class CalculateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=200)


class TextStatsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=20_000)


class ReadTextFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=500)
    max_chars: int | None = Field(ge=1, le=50_000)


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str) -> str:
    """Evaluate arithmetic only; names, calls, attributes and containers are rejected."""
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("power exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](visit(node.operand))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    result = visit(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def text_stats(text: str) -> dict[str, int]:
    return {
        "characters": len(text),
        "characters_no_spaces": len("".join(text.split())),
        "words_whitespace": len(text.split()),
        "lines": text.count("\n") + 1,
    }


def make_read_text_file(root: Path):
    root = root.resolve()

    def read_text_file(path: str, max_chars: int | None) -> str:
        candidate = (root / path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("path escapes configured tool root") from exc
        data = candidate.read_text(encoding="utf-8")
        max_chars = 8000 if max_chars is None else max_chars
        if len(data) > max_chars:
            return data[:max_chars] + "\n...[truncated]"
        return data

    return read_text_file


def build_default_registry(*, file_root: Path | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="calculate",
            description="Evaluate a basic arithmetic expression. No variables or function calls.",
            input_model=CalculateInput,
            handler=safe_calculate,
        )
    )
    registry.register(
        ToolDefinition(
            name="text_stats",
            description="Return deterministic character, word and line counts for text.",
            input_model=TextStatsInput,
            handler=text_stats,
        )
    )
    if file_root is not None:
        registry.register(
            ToolDefinition(
                name="read_text_file",
                description="Read a UTF-8 text file inside the configured read-only root.",
                input_model=ReadTextFileInput,
                handler=make_read_text_file(file_root),
            )
        )
    return registry
