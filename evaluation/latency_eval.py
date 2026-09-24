import asyncio
import statistics
import time
import uuid

from agent import agent
from harness import AgentHarness


RUNS_PER_CASE = 5


RUNS_PER_CASE = 5


TEST_CASES = [
    {
        "name": "no_tool",
        "query": "Python 中 list 和 tuple 有什么区别？",
        "expected_tool_calls": 0,
    },
    {
        "name": "rag_tool",
        "query": "我的YOLO项目主要做了什么？",
        "expected_tool_calls": 1,
    },
    {
        "name": "calculator",
        "query": "计算 5548 * 9999",
        "expected_tool_calls": 1,
    },
]

async def main():

    harness = AgentHarness(agent)

    all_results = []

    try:

        for case in TEST_CASES:

            print("\n")
            print("=" * 60)
            print(f"测试场景: {case['name']}")
            print(f"问题: {case['query']}")
            print("=" * 60)

            latencies = []
            tool_call_results = []

            for run_index in range(
                1,
                RUNS_PER_CASE + 1
            ):

                # 每次都生成新的 Conversation ID
                # 防止 Session 历史污染测试结果
                conversation_id = str(
                    uuid.uuid4()
                )

                print(
                    f"\nRun {run_index}/"
                    f"{RUNS_PER_CASE}"
                )

                start_time = (
                    time.perf_counter()
                )

                result = await harness.run(
                    user_input=case["query"],
                    conversation_id=conversation_id,
                )

                latency = (
                    time.perf_counter()
                    - start_time
                )

                tool_call_count = result.get(
                    "tool_call_count",
                    0
                )
                tool_logs = result.get(
                    "tool_logs",
                    []
                )

                called_tools = []

                for log in tool_logs:
                    if log.get("type") == "tool_call":
                        tool_name = log.get("name")

                        if tool_name:
                            called_tools.append(
                                tool_name
                            )

                print(
                    f"[EVAL] "
                    f"latency={latency:.4f}s | "
                    f"tool_calls={tool_call_count} | "
                    f"tools={called_tools}"
                )
                latencies.append(
                    latency
                )

                tool_call_results.append(
                    tool_call_count
                )

                print(
                    f"[EVAL] latency="
                    f"{latency:.4f}s | "
                    f"tool_calls="
                    f"{tool_call_count}"
                )


            mean_latency = (
                statistics.mean(
                    latencies
                )
            )

            min_latency = min(
                latencies
            )

            max_latency = max(
                latencies
            )

            expected = (
                case[
                    "expected_tool_calls"
                ]
            )

            tool_success_count = sum(
                1
                for count
                in tool_call_results
                if count == expected
            )

            tool_success_rate = (
                tool_success_count
                / RUNS_PER_CASE
            )


            summary = {
                "name":
                    case["name"],

                "mean_latency":
                    mean_latency,

                "min_latency":
                    min_latency,

                "max_latency":
                    max_latency,

                "tool_success_rate":
                    tool_success_rate,
            }

            all_results.append(
                summary
            )


        print("\n")
        print("=" * 60)
        print("Latency Benchmark Summary")
        print("=" * 60)

        for result in all_results:

            print(
                f"\n"
                f"{result['name']}\n"
                f"Mean: "
                f"{result['mean_latency']:.4f}s\n"
                f"Min: "
                f"{result['min_latency']:.4f}s\n"
                f"Max: "
                f"{result['max_latency']:.4f}s\n"
                f"Tool Success Rate: "
                f"{result['tool_success_rate']:.2%}"
            )

    finally:

        await harness.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )