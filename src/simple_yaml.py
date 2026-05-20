from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        return parse_simple_yaml(path.read_text(encoding="utf-8"))

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if data is None else data


def parse_simple_yaml(text: str) -> dict[str, Any]:
    lines = _tokenize(text)

    if not lines:
        return {}

    value, index = _parse_block(lines, 0, lines[0][0])

    if index != len(lines):
        raise ValueError("Could not parse the complete YAML document")

    if not isinstance(value, dict):
        raise ValueError("Top-level YAML document must be a mapping")

    return value


def _tokenize(text: str) -> list[tuple[int, str]]:
    lines = []

    for raw_line in text.splitlines():
        clean_line = _strip_comment(raw_line).rstrip()

        if not clean_line.strip():
            continue

        indent = len(clean_line) - len(clean_line.lstrip(" "))
        lines.append((indent, clean_line.strip()))

    return lines


def _parse_block(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if lines[index][1] == "-" or lines[index][1].startswith("- "):
        return _parse_list(lines, index, indent)

    return _parse_mapping(lines, index, indent)


def _parse_mapping(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    result = {}

    while index < len(lines):
        line_indent, content = lines[index]

        if line_indent < indent:
            break

        if line_indent > indent:
            raise ValueError(f"Unexpected indentation near: {content}")

        key, separator, value = content.partition(":")

        if separator == "":
            raise ValueError(f"Expected mapping entry near: {content}")

        key = key.strip()
        value = value.strip()

        if value:
            result[key] = _parse_scalar(value)
            index += 1
            continue

        index += 1

        if index >= len(lines) or lines[index][0] <= indent:
            result[key] = {}
            continue

        result[key], index = _parse_block(lines, index, lines[index][0])

    return result, index


def _parse_list(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    result = []

    while index < len(lines):
        line_indent, content = lines[index]

        if line_indent < indent:
            break

        if (
            line_indent > indent
            or (content != "-" and not content.startswith("- "))
        ):
            raise ValueError(f"Unexpected list entry near: {content}")

        value = "" if content == "-" else content[2:].strip()

        if value:
            result.append(_parse_scalar(value))
            index += 1
            continue

        index += 1

        if index >= len(lines) or lines[index][0] <= indent:
            result.append(None)
            continue

        child, index = _parse_block(lines, index, lines[index][0])
        result.append(child)

    return result, index


def _parse_scalar(value: str) -> Any:
    value = value.strip()

    if value in {"null", "None", "~"}:
        return None

    if value in {"true", "True"}:
        return True

    if value in {"false", "False"}:
        return False

    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()

        if not inner:
            return []

        return [_parse_scalar(item) for item in _split_inline_items(inner)]

    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)

    if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", value):
        return float(value)

    return value


def _split_inline_items(value: str) -> list[str]:
    items = []
    current = []
    quote = None
    bracket_depth = 0

    for char in value:
        if quote is not None:
            current.append(char)

            if char == quote:
                quote = None

            continue

        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue

        if char == "[":
            bracket_depth += 1
            current.append(char)
            continue

        if char == "]":
            bracket_depth -= 1
            current.append(char)
            continue

        if char == "," and bracket_depth == 0:
            items.append("".join(current).strip())
            current = []
            continue

        current.append(char)

    if current:
        items.append("".join(current).strip())

    return items


def _strip_comment(line: str) -> str:
    quote = None

    for index, char in enumerate(line):
        if quote is not None:
            if char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
            continue

        if char == "#":
            return line[:index]

    return line
