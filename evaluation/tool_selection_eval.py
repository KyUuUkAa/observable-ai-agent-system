"""Backward-compatible entry point for the unified regression pipeline."""

from evaluation.agent_regression import main
from evaluation.regression_core import classify_tool_calls, parse_tool_trace


def extract_called_tools(result: dict) -> list[str]:
    return [call["name"] for call in parse_tool_trace(result)]


def classify_result(expected_tool, called_tools):
    expected_tools = [] if expected_tool is None else [expected_tool]
    status, _ = classify_tool_calls(expected_tools, called_tools)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
