import json
import asyncio
import tempfile
import unittest
from pathlib import Path

from evaluation.regression_core import (
    build_report,
    classify_tool_calls,
    evaluate_case,
    load_case_suite,
    parse_tool_trace,
    render_markdown_report,
    summarize_results,
    write_json_report,
    write_markdown_report,
)
from evaluation.agent_regression import build_baseline_snapshot, execute_case


def make_run(*tool_names, status="success", latency=1.0):
    logs = []
    for tool_name in tool_names:
        logs.extend(
            [
                {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "arguments": json.dumps({"value": 1}),
                },
                {
                    "type": "tool_output",
                    "output": json.dumps({"status": "success"}),
                },
            ]
        )
    return {
        "status": status,
        "latency": latency,
        "agent_latency": latency - 0.1,
        "tool_logs": logs,
        "output": "ok" if status == "success" else None,
        "error": None if status == "success" else "failed",
    }


class RegressionCoreTests(unittest.TestCase):
    def test_load_case_suite_normalizes_legacy_expected_tool(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.json"
            path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "one",
                                "query": "test",
                                "expected_tool": "calculator",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            suite = load_case_suite(path)

        self.assertEqual(suite["cases"][0]["expected_tools"], ["calculator"])

    def test_parse_tool_trace_pairs_arguments_and_output(self):
        calls = parse_tool_trace(make_run("calculator"))
        self.assertEqual(calls[0]["name"], "calculator")
        self.assertEqual(calls[0]["arguments"], {"value": 1})
        self.assertEqual(calls[0]["output_status"], "success")

    def test_python_literal_tool_error_is_detected(self):
        case = {
            "id": "tool_error",
            "query": "run tool",
            "expected_tools": ["calculator"],
            "tags": [],
            "enforce_order": True,
        }
        run = make_run("calculator")
        run["tool_logs"][1]["output"] = "{'status': 'error', 'message': 'bad input'}"
        result = evaluate_case(case, run)

        self.assertEqual(result["status"], "tool_execution_error")
        self.assertIn("tool_execution_error", result["failure_types"])

    def test_expected_tool_error_can_be_allowed(self):
        case = {
            "id": "expected_error",
            "query": "invalid image",
            "expected_tools": ["recognize_oracle_image"],
            "tags": [],
            "enforce_order": True,
            "allow_tool_error": True,
        }
        run = make_run("recognize_oracle_image")
        run["tool_logs"][1]["output"] = json.dumps(
            {"status": "error", "error": "引用已过期"}
        )
        result = evaluate_case(case, run)

        self.assertTrue(result["passed"])

    def test_tool_argument_and_output_assertions(self):
        case = {
            "id": "filters",
            "query": "export",
            "expected_tools": ["export_oracle_records"],
            "tags": [],
            "enforce_order": True,
            "expected_tool_arguments": {
                "export_oracle_records": {"days": 7, "output_format": "csv"}
            },
            "required_output_terms": ["下载"],
            "forbidden_output_terms": ["现代汉字"],
        }
        run = make_run("export_oracle_records")
        run["tool_logs"][0]["arguments"] = json.dumps(
            {"days": 30, "output_format": "csv"}
        )
        run["output"] = "已生成文件"
        result = evaluate_case(case, run)

        self.assertEqual(result["status"], "output_assertion_failed")
        self.assertIn("missing_required_output", result["failure_types"])
        self.assertIn("tool_argument_mismatch", result["failure_types"])

    def test_effective_default_filters_satisfy_argument_assertion(self):
        case = {
            "id": "default_filter",
            "query": "export all",
            "expected_tools": ["export_oracle_records"],
            "tags": [],
            "enforce_order": True,
            "expected_tool_arguments": {
                "export_oracle_records": {
                    "review_status": "all",
                    "output_format": "csv",
                }
            },
        }
        run = make_run("export_oracle_records")
        run["tool_logs"][0]["arguments"] = json.dumps({"days": 1})
        run["tool_logs"][1]["output"] = json.dumps(
            {
                "status": "success",
                "format": "csv",
                "filters": {"review_status": "all", "days": 1},
            }
        )

        result = evaluate_case(case, run)

        self.assertTrue(result["passed"])

    def test_classification_matrix(self):
        self.assertEqual(classify_tool_calls([], [])[0], "correct")
        self.assertEqual(
            classify_tool_calls([], ["calculator"])[0],
            "false_tool_call",
        )
        self.assertEqual(
            classify_tool_calls(["calculator"], [])[0],
            "missed_tool_call",
        )
        self.assertEqual(
            classify_tool_calls(["calculator"], ["search_resume"])[0],
            "wrong_tool",
        )
        self.assertEqual(
            classify_tool_calls(
                ["calculator"],
                ["calculator", "calculator"],
            )[0],
            "duplicate_tool_call",
        )

    def test_evaluate_and_summarize(self):
        case = {
            "id": "calc",
            "query": "1+1",
            "expected_tools": ["calculator"],
            "tags": [],
            "enforce_order": True,
        }
        passed = evaluate_case(case, make_run("calculator", latency=1.2))
        duplicate = evaluate_case(
            case,
            make_run("calculator", "calculator", latency=2.0),
        )
        summary = summarize_results([passed, duplicate])

        self.assertTrue(passed["passed"])
        self.assertEqual(duplicate["status"], "duplicate_tool_call")
        self.assertEqual(summary["passed_cases"], 1)
        self.assertEqual(summary["failure_type_counts"]["duplicate_tool"], 1)
        self.assertAlmostEqual(summary["latency"]["mean_seconds"], 1.6)

    def test_report_detects_case_regression_and_writes_both_formats(self):
        suite = {"suite_name": "demo", "source_path": "cases.json"}
        case = {
            "id": "calc",
            "query": "1+1",
            "expected_tools": ["calculator"],
            "tags": [],
            "enforce_order": True,
        }
        result = evaluate_case(case, make_run(status="failed"))
        baseline = {
            "summary": {"pass_rate": 1.0, "latency": {}},
            "cases": [{"id": "calc", "status": "correct"}],
        }
        report = build_report(
            suite=suite,
            case_results=[result],
            baseline=baseline,
            latency_tolerance=0.20,
        )

        self.assertEqual(report["baseline_comparison"]["status"], "regressed")
        self.assertIn("calc", render_markdown_report(report))

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = write_json_report(report, Path(temp_dir) / "report.json")
            md_path = write_markdown_report(report, Path(temp_dir) / "report.md")
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

    def test_execute_case_cleans_temporary_session(self):
        class FakeHarness:
            def __init__(self):
                self.cleared = []

            async def run(self, user_input, conversation_id):
                return {
                    "status": "success",
                    "conversation_id": conversation_id,
                    "tool_logs": [],
                }

            async def clear_session(self, conversation_id):
                self.cleared.append(conversation_id)

        harness = FakeHarness()
        result = asyncio.run(
            execute_case(
                harness,
                {"query": "hello"},
                1.0,
            )
        )

        self.assertEqual(harness.cleared, [result["conversation_id"]])

    def test_baseline_snapshot_does_not_store_tool_outputs(self):
        report = {
            "schema_version": 2,
            "suite_name": "demo",
            "generated_at": "now",
            "summary": {
                "total_cases": 1,
                "passed_cases": 1,
                "failed_cases": 0,
                "pass_rate": 1.0,
                "status_counts": {"correct": 1},
                "failure_type_counts": {},
                "tool_call_counts": {"query_review_queue": 1},
                "total_tool_calls": 1,
                "latency": {},
            },
            "cases": [
                {
                    "id": "private",
                    "status": "correct",
                    "tool_calls": [{"output": "private-record"}],
                }
            ],
        }

        snapshot = build_baseline_snapshot(report)

        self.assertEqual(snapshot["cases"], [{"id": "private", "status": "correct"}])
        self.assertNotIn("private-record", json.dumps(snapshot))


if __name__ == "__main__":
    unittest.main()
