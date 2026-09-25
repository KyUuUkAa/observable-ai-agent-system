from __future__ import annotations

import ast
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2
CORRECT_STATUS = "correct"


def load_case_suite(path: str | Path) -> dict:
    suite_path = Path(path)
    with suite_path.open("r", encoding="utf-8") as case_file:
        suite = json.load(case_file)

    if not isinstance(suite, dict):
        raise ValueError("测试用例文件根节点必须是 JSON 对象。")
    if not isinstance(suite.get("cases"), list) or not suite["cases"]:
        raise ValueError("测试用例文件必须包含非空 cases 数组。")

    normalized_cases = []
    seen_ids = set()
    for raw_case in suite["cases"]:
        if not isinstance(raw_case, dict):
            raise ValueError("每条测试用例必须是 JSON 对象。")

        case_id = str(raw_case.get("id", "")).strip()
        query = str(raw_case.get("query", "")).strip()
        if not case_id or not query:
            raise ValueError("每条测试用例都必须包含非空 id 和 query。")
        if case_id in seen_ids:
            raise ValueError(f"测试用例 id 重复：{case_id}")
        seen_ids.add(case_id)

        expected_tools = raw_case.get("expected_tools")
        if expected_tools is None and "expected_tool" in raw_case:
            expected_tool = raw_case.get("expected_tool")
            expected_tools = [] if expected_tool is None else [expected_tool]
        if not isinstance(expected_tools, list) or any(
            not isinstance(tool, str) or not tool.strip()
            for tool in expected_tools
        ):
            raise ValueError(
                f"用例 {case_id} 的 expected_tools 必须是字符串数组。"
            )

        normalized_cases.append(
            {
                **raw_case,
                "id": case_id,
                "query": query,
                "expected_tools": [tool.strip() for tool in expected_tools],
                "tags": [str(tag) for tag in raw_case.get("tags", [])],
                "enforce_order": bool(raw_case.get("enforce_order", True)),
                "allow_tool_error": bool(raw_case.get("allow_tool_error", False)),
                "required_output_terms": [
                    str(term) for term in raw_case.get("required_output_terms", [])
                ],
                "forbidden_output_terms": [
                    str(term) for term in raw_case.get("forbidden_output_terms", [])
                ],
                "expected_tool_arguments": raw_case.get(
                    "expected_tool_arguments", {}
                ),
            }
        )

    return {
        **suite,
        "schema_version": suite.get("schema_version", SCHEMA_VERSION),
        "suite_name": suite.get("suite_name", suite_path.stem),
        "cases": normalized_cases,
        "source_path": str(suite_path),
    }


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value


def _tool_call_has_error(call: dict) -> bool:
    status = call.get("output_status")
    if isinstance(status, str) and status.lower() in {"error", "failed"}:
        return True

    output = call.get("output")
    if isinstance(output, str):
        normalized = output.strip().lower()
        return normalized.startswith(("error", "failed", "错误", "失败"))
    return False


def parse_tool_trace(run_result: dict) -> list[dict]:
    """Normalize Harness logs into ordered tool-call records."""

    calls: list[dict] = []
    for log in run_result.get("tool_logs", []):
        if not isinstance(log, dict):
            continue

        log_type = log.get("type")
        if log_type == "tool_call":
            name = log.get("tool_name") or log.get("name")
            if name:
                calls.append(
                    {
                        "name": str(name),
                        "arguments": _parse_json_value(log.get("arguments")),
                        "output": None,
                        "output_status": None,
                    }
                )
        elif log_type == "tool_output" and calls:
            target = next(
                (
                    call
                    for call in reversed(calls)
                    if call["output"] is None
                ),
                calls[-1],
            )
            output = log.get("output")
            parsed_output = _parse_json_value(output)
            target["output"] = parsed_output
            if isinstance(parsed_output, dict):
                target["output_status"] = parsed_output.get("status")

    return calls


def classify_tool_calls(
    expected_tools: list[str],
    actual_tools: list[str],
    *,
    enforce_order: bool = True,
) -> tuple[str, list[str]]:
    failure_types: list[str] = []
    expected_counts = Counter(expected_tools)
    actual_counts = Counter(actual_tools)

    if not expected_tools:
        if not actual_tools:
            return CORRECT_STATUS, failure_types
        return "false_tool_call", ["unexpected_tool"]

    if not actual_tools:
        return "missed_tool_call", ["missing_tool"]

    unexpected = [
        tool for tool in actual_tools if tool not in expected_counts
    ]
    missing = [
        tool
        for tool, count in expected_counts.items()
        if actual_counts[tool] < count
    ]
    duplicated = [
        tool
        for tool, count in actual_counts.items()
        if count > expected_counts.get(tool, 0)
        and tool in expected_counts
    ]

    if unexpected:
        failure_types.append("unexpected_tool")
    if missing:
        failure_types.append("missing_tool")
    if duplicated:
        failure_types.append("duplicate_tool")
    if (
        enforce_order
        and not unexpected
        and not missing
        and not duplicated
        and actual_tools != expected_tools
    ):
        failure_types.append("wrong_tool_order")

    if "unexpected_tool" in failure_types:
        return "wrong_tool", failure_types
    if "missing_tool" in failure_types:
        return "missed_tool_call", failure_types
    if "duplicate_tool" in failure_types:
        return "duplicate_tool_call", failure_types
    if "wrong_tool_order" in failure_types:
        return "wrong_tool", failure_types
    return CORRECT_STATUS, failure_types


def evaluate_case(case: dict, run_result: dict) -> dict:
    tool_calls = parse_tool_trace(run_result)
    actual_tools = [call["name"] for call in tool_calls]
    expected_tools = list(case["expected_tools"])

    if run_result.get("status") != "success":
        status = "run_failed"
        failure_types = ["agent_run_failed"]
    else:
        status, failure_types = classify_tool_calls(
            expected_tools,
            actual_tools,
            enforce_order=case.get("enforce_order", True),
        )

        tool_error = any(_tool_call_has_error(call) for call in tool_calls)
        if tool_error and not case.get("allow_tool_error", False):
            failure_types.append("tool_execution_error")
            if status == CORRECT_STATUS:
                status = "tool_execution_error"

        output_text = str(run_result.get("output") or "")
        missing_output_terms = [
            term
            for term in case.get("required_output_terms", [])
            if term.casefold() not in output_text.casefold()
        ]
        forbidden_output_terms = [
            term
            for term in case.get("forbidden_output_terms", [])
            if term.casefold() in output_text.casefold()
        ]
        if missing_output_terms:
            failure_types.append("missing_required_output")
            if status == CORRECT_STATUS:
                status = "output_assertion_failed"
        if forbidden_output_terms:
            failure_types.append("forbidden_output")
            if status == CORRECT_STATUS:
                status = "output_assertion_failed"

        argument_mismatches = []
        for tool_name, expected_arguments in case.get(
            "expected_tool_arguments", {}
        ).items():
            matching_call = next(
                (call for call in tool_calls if call["name"] == tool_name),
                None,
            )
            actual_arguments = (
                matching_call.get("arguments") if matching_call else None
            )
            if not isinstance(actual_arguments, dict):
                argument_mismatches.append(
                    {"tool": tool_name, "expected": expected_arguments, "actual": actual_arguments}
                )
                continue
            output = matching_call.get("output") if matching_call else None
            effective_filters = (
                output.get("filters", {}) if isinstance(output, dict) else {}
            )
            effective_arguments = {
                **effective_filters,
                **actual_arguments,
            }
            if isinstance(output, dict) and "format" in output:
                effective_arguments.setdefault("output_format", output["format"])
            mismatched_keys = {
                key: {
                    "expected": value,
                    "actual": effective_arguments.get(key),
                }
                for key, value in expected_arguments.items()
                if (
                    effective_arguments.get(key) not in value
                    if isinstance(value, list)
                    else effective_arguments.get(key) != value
                )
            }
            if mismatched_keys:
                argument_mismatches.append(
                    {"tool": tool_name, "keys": mismatched_keys}
                )
        if argument_mismatches:
            failure_types.append("tool_argument_mismatch")
            if status == CORRECT_STATUS:
                status = "tool_argument_mismatch"

    if run_result.get("session_cleanup_error"):
        failure_types.append("session_cleanup_error")
        if status == CORRECT_STATUS:
            status = "session_cleanup_error"

    return {
        "id": case["id"],
        "query": case["query"],
        "tags": case.get("tags", []),
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "tool_calls": tool_calls,
        "status": status,
        "failure_types": failure_types,
        "missing_output_terms": missing_output_terms if run_result.get("status") == "success" else [],
        "forbidden_output_terms": forbidden_output_terms if run_result.get("status") == "success" else [],
        "argument_mismatches": argument_mismatches if run_result.get("status") == "success" else [],
        "passed": status == CORRECT_STATUS,
        "run_id": run_result.get("run_id"),
        "conversation_id": run_result.get("conversation_id"),
        "latency_seconds": _number_or_none(run_result.get("latency")),
        "agent_latency_seconds": _number_or_none(
            run_result.get("agent_latency")
        ),
        "agent_output": run_result.get("output"),
        "error": run_result.get("error"),
        "session_cleanup_error": run_result.get("session_cleanup_error"),
    }


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def summarize_results(results: Iterable[dict]) -> dict:
    records = list(results)
    total = len(records)
    passed = sum(1 for record in records if record["passed"])
    latencies = [
        record["latency_seconds"]
        for record in records
        if record.get("latency_seconds") is not None
    ]
    status_counts = Counter(record["status"] for record in records)
    failure_counts = Counter(
        failure_type
        for record in records
        for failure_type in record.get("failure_types", [])
    )
    tool_counts = Counter(
        tool
        for record in records
        for tool in record.get("actual_tools", [])
    )

    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "status_counts": dict(sorted(status_counts.items())),
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "tool_call_counts": dict(sorted(tool_counts.items())),
        "total_tool_calls": sum(tool_counts.values()),
        "latency": {
            "sample_count": len(latencies),
            "mean_seconds": statistics.mean(latencies) if latencies else None,
            "median_seconds": statistics.median(latencies) if latencies else None,
            "p95_seconds": _percentile(latencies, 0.95),
            "min_seconds": min(latencies) if latencies else None,
            "max_seconds": max(latencies) if latencies else None,
        },
    }


def compare_with_baseline(
    current_report: dict,
    baseline_report: dict | None,
    *,
    latency_tolerance: float = 0.20,
) -> dict:
    if baseline_report is None:
        return {"status": "no_baseline"}

    current_summary = current_report["summary"]
    baseline_summary = baseline_report.get(
        "summary",
        baseline_report.get("metrics", {}),
    )
    current_cases = {
        case["id"]: case
        for case in current_report.get("cases", [])
    }
    baseline_cases = {
        case["id"]: case
        for case in baseline_report.get(
            "cases",
            baseline_report.get("results", []),
        )
    }
    case_sets_match = current_cases.keys() == baseline_cases.keys()

    regressions = []
    improvements = []
    for case_id in sorted(current_cases.keys() & baseline_cases.keys()):
        before = baseline_cases[case_id].get("status")
        after = current_cases[case_id].get("status")
        if before == CORRECT_STATUS and after != CORRECT_STATUS:
            regressions.append(
                {"id": case_id, "before": before, "after": after}
            )
        elif before != CORRECT_STATUS and after == CORRECT_STATUS:
            improvements.append(
                {"id": case_id, "before": before, "after": after}
            )

    pass_rate_before = (
        baseline_summary.get("pass_rate") if case_sets_match else None
    )
    pass_rate_after = current_summary.get("pass_rate")
    pass_rate_delta = None
    if isinstance(pass_rate_before, (int, float)):
        pass_rate_delta = pass_rate_after - float(pass_rate_before)

    latency_comparisons = {}
    for metric in ("mean_seconds", "p95_seconds") if case_sets_match else ():
        before = baseline_summary.get("latency", {}).get(metric)
        after = current_summary.get("latency", {}).get(metric)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            ratio = (after - before) / before if before else None
            latency_comparisons[metric] = {
                "before": before,
                "after": after,
                "delta_seconds": after - before,
                "delta_ratio": ratio,
                "regressed": ratio is not None and ratio > latency_tolerance,
            }

    metric_regression = (
        pass_rate_delta is not None and pass_rate_delta < 0
    ) or any(
        comparison["regressed"]
        for comparison in latency_comparisons.values()
    )

    if regressions or metric_regression:
        status = "regressed"
    elif not case_sets_match:
        status = "expanded"
    elif improvements or (pass_rate_delta is not None and pass_rate_delta > 0):
        status = "improved"
    else:
        status = "stable"

    return {
        "status": status,
        "latency_tolerance": latency_tolerance,
        "case_sets_match": case_sets_match,
        "pass_rate": {
            "before": pass_rate_before,
            "after": pass_rate_after,
            "delta": pass_rate_delta,
        },
        "latency": latency_comparisons,
        "case_regressions": regressions,
        "case_improvements": improvements,
        "new_case_ids": sorted(current_cases.keys() - baseline_cases.keys()),
        "removed_case_ids": sorted(baseline_cases.keys() - current_cases.keys()),
    }


def build_report(
    *,
    suite: dict,
    case_results: list[dict],
    baseline: dict | None,
    latency_tolerance: float,
) -> dict:
    report = {
        "schema_version": SCHEMA_VERSION,
        "suite_name": suite["suite_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_source": suite.get("source_path"),
        "summary": summarize_results(case_results),
        "cases": case_results,
    }
    report["baseline_comparison"] = compare_with_baseline(
        report,
        baseline,
        latency_tolerance=latency_tolerance,
    )
    return report


def load_optional_json(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def write_json_report(report: dict, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return output_path


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "-"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown_report(report: dict) -> str:
    summary = report["summary"]
    comparison = report["baseline_comparison"]
    latency = summary["latency"]
    lines = [
        f"# Agent Regression Report: {report['suite_name']}",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Cases: **{summary['total_cases']}**",
        f"- Passed: **{summary['passed_cases']}**",
        f"- Pass rate: **{summary['pass_rate']:.2%}**",
        f"- Baseline status: **{comparison['status']}**",
        "",
        "## Latency",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean | {_markdown_cell(latency['mean_seconds'])} s |",
        f"| Median | {_markdown_cell(latency['median_seconds'])} s |",
        f"| P95 | {_markdown_cell(latency['p95_seconds'])} s |",
        f"| Min | {_markdown_cell(latency['min_seconds'])} s |",
        f"| Max | {_markdown_cell(latency['max_seconds'])} s |",
        "",
        "## Failure Types",
        "",
    ]

    if summary["failure_type_counts"]:
        lines.extend(["| Failure type | Count |", "|---|---:|"])
        lines.extend(
            f"| {failure_type} | {count} |"
            for failure_type, count in summary["failure_type_counts"].items()
        )
    else:
        lines.append("No failures detected.")

    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            f"- Status: **{comparison['status']}**",
        ]
    )
    if comparison.get("pass_rate"):
        pass_rate = comparison["pass_rate"]
        lines.append(
            "- Pass rate: "
            f"{_markdown_cell(pass_rate['before'])} → "
            f"{_markdown_cell(pass_rate['after'])}"
        )
    for regression in comparison.get("case_regressions", []):
        lines.append(
            f"- Regression `{regression['id']}`: "
            f"{regression['before']} → {regression['after']}"
        )
    for improvement in comparison.get("case_improvements", []):
        lines.append(
            f"- Improvement `{improvement['id']}`: "
            f"{improvement['before']} → {improvement['after']}"
        )

    lines.extend(
        [
            "",
            "## Case Results",
            "",
            "| ID | Expected | Actual | Status | Latency (s) | Failures |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for case in report["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(case["id"]),
                    _markdown_cell(case["expected_tools"]),
                    _markdown_cell(case["actual_tools"]),
                    _markdown_cell(case["status"]),
                    _markdown_cell(case["latency_seconds"]),
                    _markdown_cell(case["failure_types"]),
                ]
            )
            + " |"
        )

    lines.append("")
    return "\n".join(lines)


def write_markdown_report(report: dict, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(render_markdown_report(report))
    return output_path
