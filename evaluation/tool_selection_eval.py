import asyncio
import uuid

from agent import agent
from harness import AgentHarness


TEST_CASES = [
    # =====================================================
    # search_resume
    # =====================================================
    {
        "id": "rag_01",
        "query": "我的YOLO项目主要做了什么？",
        "expected_tool": "search_resume",
    },
    {
        "id": "rag_02",
        "query": "我的Agent项目用了哪些技术？",
        "expected_tool": "search_resume",
    },
    {
        "id": "rag_03",
        "query": "我的多视图聚类项目主要解决什么问题？",
        "expected_tool": "search_resume",
    },
    {
        "id": "rag_04",
        "query": "根据我的简历介绍一下我的项目经历。",
        "expected_tool": "search_resume",
    },

    # =====================================================
    # calculator
    # =====================================================
    {
        "id": "calc_01",
        "query": "计算 1234 * 5678",
        "expected_tool": "calculator",
    },
    {
        "id": "calc_02",
        "query": "帮我算 9876 + 5432",
        "expected_tool": "calculator",
    },
    {
        "id": "calc_03",
        "query": "10000 除以 25 等于多少？",
        "expected_tool": "calculator",
    },
    {
        "id": "calc_04",
        "query": "计算 8888 - 4567",
        "expected_tool": "calculator",
    },

    # =====================================================
    # no tool
    # =====================================================
    {
        "id": "none_01",
        "query": "Python 中 list 和 tuple 有什么区别？",
        "expected_tool": None,
    },
    {
        "id": "none_02",
        "query": "什么是 Transformer？",
        "expected_tool": None,
    },
    {
        "id": "none_03",
        "query": "解释一下什么是 REST API。",
        "expected_tool": None,
    },
    {
        "id": "none_04",
        "query": "Python 的字典和集合有什么区别？",
        "expected_tool": None,
    },
]


def extract_called_tools(result: dict) -> list[str]:
    """
    从 Harness 返回结果中提取实际调用的工具名称。
    """

    tool_logs = result.get(
        "tool_logs",
        []
    )

    called_tools = []

    for log in tool_logs:

        if log.get("type") != "tool_call":
            continue

        # 兼容不同日志字段名称
        tool_name = (
            log.get("name")
            or log.get("tool_name")
        )

        if tool_name:
            called_tools.append(
                tool_name
            )

    return called_tools


def classify_result(
    expected_tool,
    called_tools
):
    """
    Tool Selection 分类。
    """

    # =============================================
    # 本来不应该调用工具
    # =============================================

    if expected_tool is None:

        if len(called_tools) == 0:
            return "correct"

        return "false_tool_call"


    # =============================================
    # 应该调用工具，但完全没有调用
    # =============================================

    if len(called_tools) == 0:
        return "missed_tool_call"


    # =============================================
    # 出现了错误工具
    # =============================================

    if any(
        tool != expected_tool
        for tool in called_tools
    ):
        return "wrong_tool"


    # =============================================
    # 正确工具只调用了一次
    # =============================================

    if len(called_tools) == 1:
        return "correct"


    # =============================================
    # 正确工具被重复调用
    # =============================================

    return "duplicate_tool_call"



async def main():

    harness = AgentHarness(
        agent
    )

    results = []

    try:

        for index, case in enumerate(
            TEST_CASES,
            start=1
        ):

            print("\n")
            print("=" * 70)
            print(
                f"[{index}/{len(TEST_CASES)}] "
                f"{case['id']}"
            )
            print(
                f"Query: {case['query']}"
            )
            print(
                f"Expected Tool: "
                f"{case['expected_tool']}"
            )
            print("=" * 70)

            # 每条测试都是全新 Session
            conversation_id = str(
                uuid.uuid4()
            )

            result = await harness.run(
                user_input=case["query"],
                conversation_id=conversation_id,
            )

            called_tools = (
                extract_called_tools(
                    result
                )
            )

            status = classify_result(
                case["expected_tool"],
                called_tools
            )

            latency = result.get(
                "agent_latency",
                result.get(
                    "latency",
                    0
                )
            )

            record = {
                "id": case["id"],
                "query": case["query"],
                "expected_tool":
                    case["expected_tool"],
                "called_tools":
                    called_tools,
                "status":
                    status,
                "latency":
                    latency,
            }

            results.append(
                record
            )

            print(
                f"[EVAL] "
                f"expected="
                f"{case['expected_tool']} | "
                f"actual="
                f"{called_tools} | "
                f"status="
                f"{status} | "
                f"latency="
                f"{latency:.4f}s"
            )


        # =================================================
        # Summary
        # =================================================

        total = len(results)

        correct = sum(
            1
            for item in results
            if item["status"]
            == "correct"
        )

        false_tool_calls = sum(
            1
            for item in results
            if item["status"]
            == "false_tool_call"
        )

        missed_tool_calls = sum(
            1
            for item in results
            if item["status"]
            == "missed_tool_call"
        )

        wrong_tools = sum(
            1
            for item in results
            if item["status"]
            == "wrong_tool"
        )


        accuracy = (
            correct / total
            if total
            else 0
        )


        print("\n")
        print("=" * 70)
        print(
            "Tool Selection Evaluation Summary"
        )
        print("=" * 70)

        print(
            f"Total Cases: "
            f"{total}"
        )

        print(
            f"Correct: "
            f"{correct}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.2%}"
        )

        print(
            f"False Tool Calls: "
            f"{false_tool_calls}"
        )

        print(
            f"Missed Tool Calls: "
            f"{missed_tool_calls}"
        )

        print(
            f"Wrong Tool Calls: "
            f"{wrong_tools}"
        )


        print("\nDetailed Results")
        print("-" * 70)

        for item in results:

            print(
                f"{item['id']:8s} | "
                f"expected="
                f"{str(item['expected_tool']):15s} | "
                f"actual="
                f"{str(item['called_tools']):25s} | "
                f"{item['status']}"
            )

    finally:

        await harness.close()
        duplicate_tool_calls = sum(
        1
        for item in results
        if item["status"]
        == "duplicate_tool_call"
    )
        print(
        f"Duplicate Tool Calls: "
        f"{duplicate_tool_calls}"
    )

if __name__ == "__main__":

    asyncio.run(
        main()
    )