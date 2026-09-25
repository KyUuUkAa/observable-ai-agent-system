import json
import time
from uuid import UUID

from agents import function_tool

from rag import search_resume_rag
from database import (
    get_oracle_record_by_image_reference,
    update_oracle_review,
)
from oracle_agent_workflow import (
    OracleWorkflowTransitionError,
    complete_retrieval,
    register_classification,
    require_retrieval,
)
from oracle_domain_service import (
    create_oracle_export,
    get_oracle_quality_metrics as load_oracle_quality_metrics,
    query_oracle_records,
)
from oracle_hybrid import get_hybrid_threshold
from oracle_recognition import (
    OracleImageReferenceError,
    OracleRecognitionError,
    get_cached_oracle_image,
    recognize_oracle_image as classify_oracle_image,
    retrieve_oracle_candidates as retrieve_visual_candidates,
)



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


@function_tool
def recognize_oracle_image(image_id: str) -> str:
    """
    识别用户已经通过系统上传的一张甲骨文单字图片。

    Args:
        image_id: 系统提供的短期图片引用。只能使用消息附件中的 image_id，
            不要编造，也不要传入本地文件路径或 URL。

    Returns:
        JSON 文本，包含预测类别编码、置信度、Top-5、融合路由、
        可视化字模/拓片候选和模型信息。低置信度检索候选仅用于辅助复核。
        当前类别编码尚未映射到现代汉字或释义。
    """

    start_time = time.perf_counter()
    print(f"[TOOL] recognize_oracle_image 被调用：image_id={image_id}")

    try:
        content = get_cached_oracle_image(image_id)
        result = classify_oracle_image(content, include_retrieval=False)
        workflow = register_classification(
            image_id,
            result,
            threshold=get_hybrid_threshold(),
        )
        record = get_oracle_record_by_image_reference(image_id)
        return json.dumps(
            {
                "status": "success",
                "notice": "类别编码尚未映射到现代汉字或释义。",
                "record_id": record["id"] if record else None,
                "workflow": workflow.safe_summary(),
                "next_action": (
                    "call retrieve_oracle_candidates once"
                    if workflow.stage.value == "classified_low_confidence"
                    else "answer with classification evidence"
                ),
                **result,
            },
            ensure_ascii=False,
        )
    except OracleImageReferenceError as error:
        return json.dumps(
            {
                "status": "error",
                "error": str(error),
            },
            ensure_ascii=False,
        )
    except OracleRecognitionError as error:
        return json.dumps(
            {
                "status": "error",
                "error": str(error),
            },
            ensure_ascii=False,
        )
    finally:
        latency = time.perf_counter() - start_time
        print(f"[PERF] recognize_oracle_image: {latency:.4f}s")


@function_tool
def retrieve_oracle_candidates(
    image_id: str,
    limit: int = 5,
    force: bool = False,
) -> str:
    """检索与已识别图片相似的字模和代表拓片。

    必须先调用 recognize_oracle_image。低置信度结果允许自动检索；只有用户明确
    要求查看相似字模时，才可以将 force 设为 true。对同一图片只能调用一次。
    """

    started = time.perf_counter()
    try:
        workflow = require_retrieval(image_id, force=force)
        content = get_cached_oracle_image(image_id)
        routing, candidates = retrieve_visual_candidates(
            content,
            workflow.classification,
            limit=limit,
            force=force,
        )
        completed = complete_retrieval(
            image_id,
            candidates,
            forced=force,
        )
        return json.dumps(
            {
                "status": "success",
                "routing": routing,
                "candidates": candidates,
                "workflow": completed.safe_summary(),
                "notice": "候选仅用于图形比对，不代表现代汉字释义。",
            },
            ensure_ascii=False,
        )
    except (
        OracleImageReferenceError,
        OracleRecognitionError,
        OracleWorkflowTransitionError,
    ) as error:
        return json.dumps(
            {"status": "error", "error": str(error)},
            ensure_ascii=False,
        )
    finally:
        print(
            f"[PERF] retrieve_oracle_candidates: "
            f"{time.perf_counter() - started:.4f}s"
        )


@function_tool
def query_review_queue(
    review_status: str = "all",
    class_code: str = "",
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    days: int = 30,
    conflicts_only: bool = False,
    limit: int = 20,
) -> str:
    """按复核状态、类别、置信度、时间和分类/检索冲突查询识别记录。

    review_status 默认 all；待复核=pending，高置信度自动通过=auto_accepted，
    人工确认=accepted，驳回=rejected。今天必须传 days=1，本周传 days=7。
    """

    try:
        records = query_oracle_records(
            review_status=review_status,
            class_code=class_code,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            days=days,
            conflicts_only=conflicts_only,
            limit=limit,
        )
        return json.dumps(
            {
                "status": "success",
                "count": len(records),
                "records": records,
                "filters": {
                    "review_status": review_status,
                    "class_code": class_code or None,
                    "min_confidence": min_confidence,
                    "max_confidence": max_confidence,
                    "days": days,
                    "conflicts_only": conflicts_only,
                },
            },
            ensure_ascii=False,
        )
    except ValueError as error:
        return json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False)


@function_tool
def update_review_result(
    record_id: str,
    status: str,
    notes: str = "",
) -> str:
    """在用户明确要求后，将一条识别记录确认或驳回。

    status 只允许 accepted 或 rejected。不得根据模型输出自行替用户复核。
    """

    normalized = status.strip().lower()
    if normalized not in {"accepted", "rejected"}:
        return json.dumps(
            {"status": "error", "error": "复核操作只允许 accepted 或 rejected。"},
            ensure_ascii=False,
        )
    try:
        normalized_record_id = str(UUID(record_id))
    except ValueError:
        return json.dumps(
            {"status": "error", "error": "record_id 必须是有效的 UUID。"},
            ensure_ascii=False,
        )
    record = update_oracle_review(
        normalized_record_id,
        review_status=normalized,
        review_notes=notes.strip() or None,
    )
    if record is None:
        return json.dumps(
            {"status": "not_found", "record_id": record_id},
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "status": "success",
            "record_id": record["id"],
            "review_status": record["review_status"],
            "review_notes": record["review_notes"],
            "reviewed_at": record["reviewed_at"],
        },
        ensure_ascii=False,
    )


@function_tool
def export_oracle_records(
    output_format: str = "csv",
    review_status: str = "all",
    class_code: str = "",
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    days: int = 30,
    conflicts_only: bool = False,
) -> str:
    """按指定过滤条件生成甲骨文识别记录 CSV 或 JSON 导出文件。"""

    try:
        result = create_oracle_export(
            output_format=output_format,
            review_status=review_status,
            class_code=class_code,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            days=days,
            conflicts_only=conflicts_only,
        )
        return json.dumps({"status": "success", **result}, ensure_ascii=False)
    except ValueError as error:
        return json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False)


@function_tool
def get_oracle_quality_metrics() -> str:
    """查询分类、检索、置信度校准、模型版本和Agent回归评估指标。"""

    return json.dumps(
        {"status": "success", **load_oracle_quality_metrics()},
        ensure_ascii=False,
    )
