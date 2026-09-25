from __future__ import annotations

import argparse
import asyncio
import io
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from evaluation.regression_core import (
    build_report,
    evaluate_case,
    load_case_suite,
    load_optional_json,
    write_json_report,
    write_markdown_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "cases" / "tool_routing.json"
DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "evaluation"
    / "baselines"
    / "tool_routing.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "agent_regression"


def build_baseline_snapshot(report: dict) -> dict:
    """Keep regression evidence without persisting tool outputs or record data."""

    summary = report["summary"]
    return {
        "schema_version": report.get("schema_version"),
        "suite_name": report.get("suite_name"),
        "generated_at": report.get("generated_at"),
        "notes": "Sanitized accepted Agent regression baseline.",
        "summary": {
            key: summary.get(key)
            for key in (
                "total_cases",
                "passed_cases",
                "failed_cases",
                "pass_rate",
                "status_counts",
                "failure_type_counts",
                "tool_call_counts",
                "total_tool_calls",
                "latency",
            )
        },
        "cases": [
            {"id": case["id"], "status": case["status"]}
            for case in report.get("cases", [])
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Agent regression cases, classify tool-call failures, "
            "compare a baseline and generate JSON/Markdown reports."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--keep-sessions",
        action="store_true",
        help="Keep temporary Agent database sessions for manual debugging.",
    )
    parser.add_argument(
        "--latency-tolerance",
        type=float,
        default=0.20,
        help="Allowed relative latency growth before baseline regression.",
    )
    parser.add_argument(
        "--write-baseline",
        nargs="?",
        const=DEFAULT_BASELINE,
        type=Path,
        default=None,
        help=(
            "After review, save this run as the baseline. "
            "Optionally provide a different output path."
        ),
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 2 when the baseline comparison regresses.",
    )
    parser.add_argument(
        "--fail-on-case-failure",
        action="store_true",
        help="Exit with code 3 when one or more cases fail.",
    )
    return parser


def select_cases(suite: dict, case_ids: list[str], tags: list[str]) -> list[dict]:
    selected = list(suite["cases"])
    if case_ids:
        wanted_ids = set(case_ids)
        selected = [case for case in selected if case["id"] in wanted_ids]
        missing = wanted_ids - {case["id"] for case in selected}
        if missing:
            raise ValueError(
                "未找到测试用例：" + ", ".join(sorted(missing))
            )
    if tags:
        wanted_tags = set(tags)
        selected = [
            case
            for case in selected
            if wanted_tags.intersection(case.get("tags", []))
        ]
    if not selected:
        raise ValueError("筛选后没有可运行的测试用例。")
    return selected


async def execute_case(
    harness,
    case: dict,
    timeout_seconds: float,
    *,
    keep_session: bool = False,
) -> dict:
    conversation_id = str(uuid.uuid4())
    user_input = case["query"]
    if "{{oracle_image_id}}" in user_input:
        from PIL import Image, ImageDraw

        from oracle_recognition import cache_oracle_image

        image = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(image)
        draw.line((72, 38, 152, 186), fill="black", width=9)
        draw.line((151, 40, 74, 184), fill="black", width=9)
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_id = cache_oracle_image(image_buffer.getvalue())
        user_input = user_input.replace("{{oracle_image_id}}", image_id)
    try:
        run_result = await asyncio.wait_for(
            harness.run(
                user_input=user_input,
                conversation_id=conversation_id,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        run_result = {
            "run_id": None,
            "conversation_id": conversation_id,
            "status": "failed",
            "output": None,
            "latency": timeout_seconds,
            "agent_latency": timeout_seconds,
            "tool_logs": [],
            "error": f"case timed out after {timeout_seconds:.1f}s",
        }
    except Exception as error:
        run_result = {
            "run_id": None,
            "conversation_id": conversation_id,
            "status": "failed",
            "output": None,
            "latency": None,
            "agent_latency": None,
            "tool_logs": [],
            "error": str(error),
        }

    if not keep_session:
        try:
            await harness.clear_session(conversation_id)
        except Exception as cleanup_error:
            run_result["session_cleanup_error"] = str(cleanup_error)

    return run_result


async def run_pipeline(args: argparse.Namespace) -> tuple[dict, Path, Path]:
    suite = load_case_suite(args.cases)
    suite["cases"] = select_cases(suite, args.case_id, args.tag)
    baseline = load_optional_json(args.baseline)

    # Heavy model, database and embedding imports stay out of regression_core,
    # so its classification/report logic remains unit-testable offline.
    from agent import agent
    from harness import AgentHarness

    args.output_dir.mkdir(parents=True, exist_ok=True)
    harness_log = args.output_dir / "harness.jsonl"
    harness = AgentHarness(agent, log_file=str(harness_log))
    case_results = []

    try:
        for index, case in enumerate(suite["cases"], start=1):
            print(
                f"[{index}/{len(suite['cases'])}] "
                f"{case['id']} expected={case['expected_tools']}"
            )
            run_result = await execute_case(
                harness,
                case,
                args.timeout,
                keep_session=args.keep_sessions,
            )
            evaluated = evaluate_case(case, run_result)
            case_results.append(evaluated)
            print(
                f"  actual={evaluated['actual_tools']} "
                f"status={evaluated['status']} "
                f"latency={evaluated['latency_seconds']}"
            )
    finally:
        await harness.close()

    report = build_report(
        suite=suite,
        case_results=case_results,
        baseline=baseline,
        latency_tolerance=args.latency_tolerance,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_stem = f"{suite['suite_name']}-{timestamp}"
    json_path = write_json_report(
        report,
        args.output_dir / f"{report_stem}.json",
    )
    markdown_path = write_markdown_report(
        report,
        args.output_dir / f"{report_stem}.md",
    )
    write_json_report(report, args.output_dir / "latest.json")
    write_markdown_report(report, args.output_dir / "latest.md")

    if args.write_baseline:
        write_json_report(build_baseline_snapshot(report), args.write_baseline)

    return report, json_path, markdown_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0。")
    if args.latency_tolerance < 0:
        parser.error("--latency-tolerance 不能小于 0。")

    try:
        report, json_path, markdown_path = asyncio.run(run_pipeline(args))
    except (OSError, ValueError) as error:
        print(f"Agent regression configuration error: {error}", file=sys.stderr)
        return 1

    summary = report["summary"]
    comparison = report["baseline_comparison"]
    print("\nAgent Regression Summary")
    print(f"Cases: {summary['total_cases']}")
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    print(f"Baseline: {comparison['status']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")

    if args.fail_on_regression and comparison["status"] == "regressed":
        return 2
    if args.fail_on_case_failure and summary["failed_cases"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
