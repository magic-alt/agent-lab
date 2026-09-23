from pathlib import Path

from agent_lab.tools.builtins import build_default_registry, safe_calculate


def test_safe_calculate_arithmetic_only():
    assert safe_calculate("(17.5 * 8) / 7") == "20"


def test_safe_calculate_rejects_code_execution():
    try:
        safe_calculate("abs(-1)")
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("function calls were accepted")


def test_registry_rejects_extra_arguments():
    registry = build_default_registry()
    result = registry.execute(
        call_id="call_1",
        name="calculate",
        arguments_json='{"expression":"2+2","unexpected":true}',
    )
    assert not result.ok
    assert "ValidationError" in result.output


def test_file_tool_is_jailed(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("hello", encoding="utf-8")
    registry = build_default_registry(file_root=tmp_path)
    ok = registry.execute(
        call_id="a",
        name="read_text_file",
        arguments_json='{"path":"ok.txt","max_chars":10}',
    )
    assert ok.ok and ok.output == "hello"
    escaped = registry.execute(
        call_id="b",
        name="read_text_file",
        arguments_json='{"path":"../secret.txt","max_chars":null}',
    )
    assert not escaped.ok


def test_openai_strict_tool_schemas_require_all_properties():
    registry = build_default_registry(file_root=Path.cwd())
    for tool in registry.schemas():
        params = tool["parameters"]
        assert params["additionalProperties"] is False
        assert set(params.get("required", [])) == set(params.get("properties", {}))
        assert tool["strict"] is True
