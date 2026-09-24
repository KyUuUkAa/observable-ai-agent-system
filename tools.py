from agents import function_tool

from rag import search_resume_rag

import time



@function_tool
def calculator(a: float, b: float, operation: str) -> str:
    """
    执行简单数学计算。

    operation 支持：
    - add 或 +
    - subtract 或 -
    - multiply、*、x、×
    - divide、/、÷
    """

    start_time = time.perf_counter()

    try:
        print(
            f"[TOOL] calculator 被调用："
            f"a={a}, b={b}, operation={operation}"
        )

        # 统一模型可能生成的不同操作符
        op = operation.strip().lower()

        operation_aliases = {
            "add": "add",
            "+": "add",

            "subtract": "subtract",
            "-": "subtract",

            "multiply": "multiply",
            "*": "multiply",
            "x": "multiply",
            "×": "multiply",

            "divide": "divide",
            "/": "divide",
            "÷": "divide",
        }

        normalized_operation = operation_aliases.get(op)

        if normalized_operation == "add":
            result = a + b

        elif normalized_operation == "subtract":
            result = a - b

        elif normalized_operation == "multiply":
            result = a * b

        elif normalized_operation == "divide":
            if b == 0:
                return "除数不能为0"

            result = a / b

        else:
            return f"不支持的操作：{operation}"

        # 如果结果本质上是整数，避免显示成 7006652.0
        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return f"计算结果：{result}"

    finally:
        latency = (
            time.perf_counter()
            - start_time
        )

        print(
            f"[PERF] calculator: "
            f"{latency:.4f}s"
        )

@function_tool

def search_resume(
    query: str
) -> str:

    start_time = time.perf_counter()

    print(
        f"[TOOL] search_resume 被调用：query={query}"
    )

    try:
        results = search_resume_rag(
            query,
            top_k=2
        )

        output_parts = []

        for index, result in enumerate(
            results,
            start=1
        ):
            output_parts.append(
                f"检索结果 Top-{index}\n"
                f"Chunk ID: {result['chunk_id']}\n"
                f"Similarity: {result['score']:.4f}\n\n"
                f"{result['content']}"
            )

        return "\n\n".join(
            output_parts
        )

    finally:
        latency = (
            time.perf_counter()
            - start_time
        )

        print(
            f"[PERF] search_resume: "
            f"{latency:.4f}s"
        )
