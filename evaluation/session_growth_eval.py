import asyncio
import csv
import json
import statistics
import uuid
from pathlib import Path

from agent import agent
from harness import AgentHarness
from database import get_connection


# =========================================================
# 配置
# =========================================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = REPORT_DIR / "session_growth_eval.json"
REPORT_CSV = REPORT_DIR / "session_growth_eval.csv"
BASE_DIR = Path(__file__).resolve().parent

REPORT_DIR = (
    BASE_DIR
    / "reports"
)

REPORT_DIR.mkdir(
    exist_ok=True
)

JSON_PATH = (
    REPORT_DIR
    / "session_growth_eval.json"
)

CSV_PATH = (
    REPORT_DIR
    / "session_growth_eval.csv"
)


# =========================================================
# 测试轮次
#
# 同一个 Conversation 连续运行。
#
# 为了避免只测一种任务导致结论偏差，
# 使用：
#
# no_tool
# rag
# calculator
#
# 三种任务循环。
# =========================================================

TEST_ROUNDS = [

    {
        "round": 1,
        "type": "no_tool",
        "query": "Python 中 list 和 tuple 有什么区别？",
    },

    {
        "round": 2,
        "type": "rag",
        "query": "我的Agent项目主要做了什么？",
    },

    {
        "round": 3,
        "type": "calculator",
        "query": "计算 1234 * 5678",
    },

    {
        "round": 4,
        "type": "no_tool",
        "query": "Python 中字典和集合有什么区别？",
    },

    {
        "round": 5,
        "type": "rag",
        "query": "我的YOLO项目主要做了什么？",
    },

    {
        "round": 6,
        "type": "calculator",
        "query": "计算 2468 * 1357",
    },

    {
        "round": 7,
        "type": "no_tool",
        "query": "什么是 REST API？",
    },

    {
        "round": 8,
        "type": "rag",
        "query": "我的多视图聚类项目用了哪些技术？",
    },

    {
        "round": 9,
        "type": "calculator",
        "query": "计算 8765 + 4321",
    },

    {
        "round": 10,
        "type": "no_tool",
        "query": "Transformer 中 Attention 的作用是什么？",
    },

    {
        "round": 11,
        "type": "rag",
        "query": "我有没有使用 FastAPI 和 PostgreSQL？",
    },

    {
        "round": 12,
        "type": "calculator",
        "query": "计算 9999 - 4321",
    },
]


# =========================================================
# 提取实际调用工具
# =========================================================

def extract_called_tools(
    result: dict
) -> list[str]:

    tool_logs = result.get(
        "tool_logs",
        []
    )

    called_tools = []

    for log in tool_logs:

        if log.get("type") != "tool_call":
            continue

        tool_name = (
            log.get("name")
            or
            log.get("tool_name")
        )

        if tool_name:
            called_tools.append(
                tool_name
            )

    return called_tools


# =========================================================
# 获取 PostgreSQL Session 统计
#
# message_count:
# 当前 session 有多少 agent_messages
#
# payload_chars:
# message_data 转成文本后的字符数
#
# payload_bytes:
# message_data 实际 PostgreSQL 存储大小
#
# row_bytes:
# agent_messages 整行数据的大致存储大小
# =========================================================

def get_session_stats(
    session_id: str
) -> dict:

    conn = get_connection()

    try:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    COUNT(*) AS message_count,

                    COALESCE(
                        SUM(
                            LENGTH(
                                message_data::text
                            )
                        ),
                        0
                    ) AS payload_chars,

                    COALESCE(
                        SUM(
                            pg_column_size(
                                message_data
                            )
                        ),
                        0
                    ) AS payload_bytes,

                    COALESCE(
                        SUM(
                            pg_column_size(am)
                        ),
                        0
                    ) AS row_bytes

                FROM agent_messages AS am

                WHERE session_id = %s
                """,
                (
                    session_id,
                )
            )

            row = cursor.fetchone()

            message_count = int(
                row[0] or 0
            )

            payload_chars = int(
                row[1] or 0
            )

            payload_bytes = int(
                row[2] or 0
            )

            row_bytes = int(
                row[3] or 0
            )


            # =============================================
            # agent_sessions 是否存在
            # =============================================

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM agent_sessions
                WHERE session_id = %s
                """,
                (
                    session_id,
                )
            )

            session_exists = (
                cursor.fetchone()[0]
                > 0
            )


            return {

                "session_exists":
                    session_exists,

                "message_count":
                    message_count,

                "payload_chars":
                    payload_chars,

                "payload_bytes":
                    payload_bytes,

                "row_bytes":
                    row_bytes,

                "payload_kb":
                    payload_bytes
                    / 1024,

                "row_kb":
                    row_bytes
                    / 1024,
            }

    finally:

        conn.close()


# =========================================================
# 保存 JSON
# =========================================================

def save_json(
    output: dict
):

    JSON_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


# =========================================================
# 保存 CSV
# =========================================================

def save_csv(
    records: list[dict]
):

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "round",
            "type",
            "query",
            "agent_latency",
            "total_latency",
            "tool_call_count",
            "called_tools",
            "message_count",
            "messages_added",
            "payload_chars",
            "payload_bytes",
            "payload_kb",
            "payload_bytes_added",
            "row_bytes",
            "row_kb",
        ])

        for item in records:

            writer.writerow([
                item["round"],
                item["type"],
                item["query"],
                item["agent_latency"],
                item["total_latency"],
                item["tool_call_count"],
                item["called_tools"],
                item["message_count"],
                item["messages_added"],
                item["payload_chars"],
                item["payload_bytes"],
                item["payload_kb"],
                item["payload_bytes_added"],
                item["row_bytes"],
                item["row_kb"],
            ])


# =========================================================
# 打印最终分析
# =========================================================

def print_summary(
    records
):

    print("\n")
    print("=" * 80)
    print(
        "Session Growth Evaluation Summary"
    )
    print("=" * 80)


    first = records[0]
    last = records[-1]


    print(
        f"Rounds: "
        f"{len(records)}"
    )

    print(
        f"Messages: "
        f"{first['message_count']} "
        f"→ "
        f"{last['message_count']}"
    )

    print(
        f"Payload Size: "
        f"{first['payload_kb']:.2f} KB "
        f"→ "
        f"{last['payload_kb']:.2f} KB"
    )

    print(
        f"Row Storage: "
        f"{first['row_kb']:.2f} KB "
        f"→ "
        f"{last['row_kb']:.2f} KB"
    )


    # =====================================================
    # 平均每轮新增消息
    # =====================================================

    average_messages_added = (
        statistics.mean(
            item["messages_added"]
            for item
            in records
        )
    )

    average_bytes_added = (
        statistics.mean(
            item[
                "payload_bytes_added"
            ]
            for item
            in records
        )
    )


    print(
        f"Average Messages / Round: "
        f"{average_messages_added:.2f}"
    )

    print(
        f"Average Payload Growth / Round: "
        f"{average_bytes_added / 1024:.2f} KB"
    )


    # =====================================================
    # 前 3 轮 vs 后 3 轮延迟
    # =====================================================

    if len(records) >= 6:

        first_three_latency = (
            statistics.mean(
                item["agent_latency"]
                for item
                in records[:3]
            )
        )

        last_three_latency = (
            statistics.mean(
                item["agent_latency"]
                for item
                in records[-3:]
            )
        )

        latency_change = (
            (
                last_three_latency
                - first_three_latency
            )
            / first_three_latency
            * 100
        )


        print(
            f"First 3 Rounds Mean Latency: "
            f"{first_three_latency:.4f}s"
        )

        print(
            f"Last 3 Rounds Mean Latency: "
            f"{last_three_latency:.4f}s"
        )

        print(
            f"Latency Change: "
            f"{latency_change:+.2f}%"
        )


    # =====================================================
    # 分任务统计
    # =====================================================

    print("\n")
    print("=" * 80)
    print(
        "Latency by Task Type"
    )
    print("=" * 80)


    task_types = sorted(
        set(
            item["type"]
            for item
            in records
        )
    )


    for task_type in task_types:

        task_records = [
            item
            for item
            in records
            if item["type"]
            == task_type
        ]

        latencies = [
            item["agent_latency"]
            for item
            in task_records
        ]

        print(
            f"\n{task_type}"
        )

        print(
            f"Mean: "
            f"{statistics.mean(latencies):.4f}s"
        )

        print(
            f"Min: "
            f"{min(latencies):.4f}s"
        )

        print(
            f"Max: "
            f"{max(latencies):.4f}s"
        )


    # =====================================================
    # 逐轮表
    # =====================================================

    print("\n")
    print("=" * 80)
    print(
        "Round-by-Round Results"
    )
    print("=" * 80)

    print(
        "Round | Type       | "
        "Latency | Msgs | "
        "Added | Payload KB | Tools"
    )

    print(
        "-" * 80
    )


    for item in records:

        print(
            f"{item['round']:>5} | "
            f"{item['type']:<10} | "
            f"{item['agent_latency']:>7.2f}s | "
            f"{item['message_count']:>4} | "
            f"{item['messages_added']:>5} | "
            f"{item['payload_kb']:>10.2f} | "
            f"{item['called_tools']}"
        )


# =========================================================
# Main
# =========================================================

async def main():

    harness = AgentHarness(
        agent
    )


    # =====================================================
    # 整个实验只使用一个 Conversation
    # =====================================================

    conversation_id = str(
        uuid.uuid4()
    )


    print("\n")
    print("=" * 80)
    print(
        "Session Growth Evaluation"
    )
    print("=" * 80)

    print(
        f"Conversation ID: "
        f"{conversation_id}"
    )

    print(
        f"Rounds: "
        f"{len(TEST_ROUNDS)}"
    )


    records = []

    previous_message_count = 0
    previous_payload_bytes = 0


    try:

        for case in TEST_ROUNDS:

            print("\n")
            print("=" * 80)

            print(
                f"Round "
                f"{case['round']}"
                f"/"
                f"{len(TEST_ROUNDS)}"
            )

            print(
                f"Type: "
                f"{case['type']}"
            )

            print(
                f"Query: "
                f"{case['query']}"
            )

            print("=" * 80)


            # =============================================
            # Agent Run
            # =============================================

            result = await harness.run(
                user_input=
                    case["query"],

                conversation_id=
                    conversation_id,
            )


            # =============================================
            # Harness Metrics
            # =============================================

            agent_latency = float(
                result.get(
                    "agent_latency",
                    result.get(
                        "latency",
                        0
                    )
                )
            )

            total_latency = float(
                result.get(
                    "latency",
                    agent_latency
                )
            )

            tool_call_count = int(
                result.get(
                    "tool_call_count",
                    0
                )
            )

            called_tools = (
                extract_called_tools(
                    result
                )
            )


            # =============================================
            # PostgreSQL Session Metrics
            # =============================================

            session_stats = (
                get_session_stats(
                    conversation_id
                )
            )


            messages_added = (
                session_stats[
                    "message_count"
                ]
                - previous_message_count
            )

            payload_bytes_added = (
                session_stats[
                    "payload_bytes"
                ]
                - previous_payload_bytes
            )


            record = {

                "round":
                    case["round"],

                "type":
                    case["type"],

                "query":
                    case["query"],

                "agent_latency":
                    agent_latency,

                "total_latency":
                    total_latency,

                "tool_call_count":
                    tool_call_count,

                "called_tools":
                    called_tools,

                "message_count":
                    session_stats[
                        "message_count"
                    ],

                "messages_added":
                    messages_added,

                "payload_chars":
                    session_stats[
                        "payload_chars"
                    ],

                "payload_bytes":
                    session_stats[
                        "payload_bytes"
                    ],

                "payload_kb":
                    session_stats[
                        "payload_kb"
                    ],

                "payload_bytes_added":
                    payload_bytes_added,

                "row_bytes":
                    session_stats[
                        "row_bytes"
                    ],

                "row_kb":
                    session_stats[
                        "row_kb"
                    ],
            }


            records.append(
                record
            )


            # =============================================
            # 当前轮输出
            # =============================================

            print(
                "\n[SESSION EVAL]"
            )

            print(
                f"Agent Latency: "
                f"{agent_latency:.4f}s"
            )

            print(
                f"Tool Calls: "
                f"{tool_call_count}"
            )

            print(
                f"Called Tools: "
                f"{called_tools}"
            )

            print(
                f"Agent Messages: "
                f"{session_stats['message_count']}"
            )

            print(
                f"Messages Added: "
                f"{messages_added}"
            )

            print(
                f"Payload Size: "
                f"{session_stats['payload_kb']:.2f} KB"
            )

            print(
                f"Payload Added: "
                f"{payload_bytes_added / 1024:.2f} KB"
            )


            # =============================================
            # 更新 Previous
            # =============================================

            previous_message_count = (
                session_stats[
                    "message_count"
                ]
            )

            previous_payload_bytes = (
                session_stats[
                    "payload_bytes"
                ]
            )


        # =================================================
        # Summary
        # =================================================

        print_summary(
            records
        )


        # =================================================
        # 输出文件
        # =================================================

        output = {

            "conversation_id":
                conversation_id,

            "rounds":
                len(records),

            "results":
                records,
        }


        save_json(
            output
        )

        save_csv(
            records
        )


        print("\n")
        print("=" * 80)
        print(
            "Reports Saved"
        )
        print("=" * 80)

        print(
            f"JSON: "
            f"{JSON_PATH}"
        )

        print(
            f"CSV: "
            f"{CSV_PATH}"
        )


        print("\n")
        print(
            "测试 Session 保留在 PostgreSQL 中，"
            "便于后续人工检查："
        )

        print(
            conversation_id
        )


    finally:

        await harness.close()


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )