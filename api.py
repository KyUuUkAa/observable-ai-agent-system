from dotenv import load_dotenv

# ============================================================
# Environment
# ============================================================

# 必须尽量早加载环境变量，
# 因为 agent.py / harness.py 会读取数据库和模型相关配置。
load_dotenv(override=True)


# ============================================================
# Imports
# ============================================================

import time
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import agent
from harness import AgentHarness

from oracle_recognition import (
    OracleImageError,
    OracleModelUnavailableError,
    OracleRecognitionError,
    cache_oracle_image,
    get_model_status,
    has_cached_oracle_image,
    recognize_oracle_image,
)
from oracle_hybrid import (
    OracleCandidateNotFoundError,
    OracleRetrievalUnavailableError,
    get_candidate_image,
)
from oracle_workflow import (
    MAX_BATCH_FILES,
    build_recognition_trace,
    get_default_review_threshold,
    oracle_records_to_csv,
    oracle_records_to_json,
    validate_review_threshold,
)

from database import (
    attach_oracle_agent_trace,
    create_oracle_recognition_record,
    create_conversation,
    get_oracle_recognition_image,
    get_oracle_recognition_record,
    get_oracle_review_summary,
    get_conversations,
    add_message,
    get_messages,
    list_oracle_recognition_records,
    update_conversation_title,
    update_oracle_review,
    delete_conversation,
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Observable AI Agent System API",
    description=(
        "Backend API for the Observable AI Agent System. "
        "Provides Agent interaction, tool calling, "
        "conversation persistence, execution tracing, "
        "and RAG-based retrieval."
    ),
    version="1.1.0",
    openapi_tags=[
        {
            "name": "System",
            "description": (
                "System health checks and backend service status."
            ),
        },
        {
            "name": "Agent",
            "description": (
                "Agent interaction, tool calling, "
                "RAG retrieval and execution trace."
            ),
        },
        {
            "name": "Conversations",
            "description": (
                "Conversation lifecycle, message history "
                "and Agent Session management."
            ),
        },
        {
            "name": "Oracle Recognition",
            "description": (
                "Single-glyph Oracle Bone Script image classification."
            ),
        },
    ],
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Agent Harness
# ============================================================

harness = AgentHarness(agent)


# ============================================================
# Request Models
# ============================================================

class ChatRequest(BaseModel):
    """
    Agent Chat 请求体。
    """

    conversation_id: str = Field(
        ...,
        description=(
            "Conversation ID used for multi-turn "
            "Agent Session persistence."
        ),
        examples=[
            "7a8d3c17-eb7b-4fc5-a66c-d653b91d6675"
        ],
    )

    message: str = Field(
        ...,
        description="User message sent to the Agent.",
        examples=[
            "我的YOLO项目主要做了什么？"
        ],
    )

    oracle_image_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{32}$",
        description=(
            "Optional short-lived reference returned by /oracle/recognize. "
            "When present, the Agent must invoke the Oracle recognition tool."
        ),
        examples=[
            "6c77965e2f174638b99b6ec0ea8f5771"
        ],
    )


class OracleReviewRequest(BaseModel):
    status: Literal["pending", "accepted", "rejected"] = Field(
        ...,
        description="Human review decision.",
    )
    notes: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional reviewer notes.",
    )


# ============================================================
# Helper Functions
# ============================================================

def generate_conversation_title(
    message: str
) -> str:
    """
    根据首条用户消息生成简短会话标题。

    当前采用轻量规则生成标题，
    后续可以替换为 LLM Title Generator。
    """

    text = message.strip()
    lower_text = text.lower()

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    if "yolo" in lower_text:
        if (
            "优化" in text
            or "改进" in text
        ):
            return "YOLO 项目优化"

        return "YOLO 项目介绍"

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    if (
        "agent" in lower_text
        or "智能体" in text
    ):
        return "Agent 项目介绍"

    # --------------------------------------------------------
    # 大模型
    # --------------------------------------------------------

    if (
        "大模型" in text
        or "llm" in lower_text
    ):
        return "大模型应用项目"

    # --------------------------------------------------------
    # 多视图聚类
    # --------------------------------------------------------

    if (
        "多视图" in text
        or "聚类" in text
    ):
        return "多视图聚类项目"

    # --------------------------------------------------------
    # 甲骨文
    # --------------------------------------------------------

    if "甲骨文" in text:
        return "甲骨文项目"

    # --------------------------------------------------------
    # 简历
    # --------------------------------------------------------

    if "简历" in text:
        return "简历分析"

    # --------------------------------------------------------
    # 通用 fallback
    # --------------------------------------------------------

    title = text

    for suffix in [
        "？",
        "?",
        "。",
        "！",
        "!",
    ]:
        title = title.rstrip(suffix)

    max_length = 20

    if len(title) > max_length:
        title = (
            title[:max_length]
            + "..."
        )

    if not title:
        return "New Conversation"

    return title


# ============================================================
# System API
# ============================================================

@app.get(
    "/health",
    tags=["System"],
    summary="Health Check",
    description=(
        "Check whether the FastAPI backend "
        "is running normally."
    ),
)
def health():
    """
    后端健康检查。
    """

    return {
        "status": "ok"
    }


# ============================================================
# Oracle Bone Script Recognition API
# ============================================================

MAX_ORACLE_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_ORACLE_REVIEW_THRESHOLD = get_default_review_threshold()


async def read_oracle_upload(file: UploadFile) -> bytes:
    if (
        file.content_type
        and not file.content_type.startswith("image/")
        and file.content_type != "application/octet-stream"
    ):
        raise HTTPException(
            status_code=400,
            detail="只支持图片文件。",
        )

    content = await file.read(MAX_ORACLE_UPLOAD_BYTES + 1)
    if len(content) > MAX_ORACLE_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="图片不能超过 10 MB。",
        )
    return content


def recognize_and_store_oracle(
    *,
    content: bytes,
    filename: str,
    content_type: str | None,
    review_threshold: float,
    source: str,
    conversation_id: str | None,
):
    started_at = time.perf_counter()
    result = recognize_oracle_image(content)
    image_id = cache_oracle_image(content)
    trace = build_recognition_trace(
        started_at=started_at,
        source=source,
        filename=filename,
        routing=result.get("routing"),
    )
    record = create_oracle_recognition_record(
        image_content=content,
        original_filename=filename,
        content_type=content_type,
        image_reference_id=image_id,
        recognition=result,
        review_threshold=review_threshold,
        source=source,
        execution_trace=trace,
        conversation_id=conversation_id,
    )
    return {
        "filename": filename,
        "image_id": image_id,
        "record_id": record["id"],
        "review_status": record["review_status"],
        "review_threshold": record["review_threshold"],
        **result,
    }


@app.get(
    "/oracle/health",
    tags=["Oracle Recognition"],
    summary="Oracle Classifier Health",
    description=(
        "Check whether the local single-glyph classifier is configured. "
        "This endpoint does not load the model checkpoint."
    ),
)
def oracle_health():
    status = get_model_status()
    if status["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail=status["detail"],
        )
    return status


@app.post(
    "/oracle/recognize",
    tags=["Oracle Recognition"],
    summary="Recognize One Oracle Bone Script Glyph",
    description=(
        "Upload one cropped glyph image. The response contains the "
        "predicted class code, confidence and Top-5 candidates."
    ),
)
async def recognize_oracle(
    file: UploadFile = File(...),
    review_threshold: float = Form(DEFAULT_ORACLE_REVIEW_THRESHOLD),
    conversation_id: str | None = Form(default=None),
):
    filename = file.filename or "oracle-image"
    content_type = file.content_type
    try:
        threshold = validate_review_threshold(review_threshold)
        content = await read_oracle_upload(file)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await file.close()

    try:
        return await run_in_threadpool(
            recognize_and_store_oracle,
            content=content,
            filename=filename,
            content_type=content_type,
            review_threshold=threshold,
            source="single",
            conversation_id=conversation_id,
        )
    except OracleImageError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except OracleModelUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error
    except OracleRecognitionError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error


@app.post(
    "/oracle/batch",
    tags=["Oracle Recognition"],
    summary="Recognize A Batch Of Oracle Glyphs",
)
async def recognize_oracle_batch(
    files: list[UploadFile] = File(...),
    review_threshold: float = Form(DEFAULT_ORACLE_REVIEW_THRESHOLD),
    conversation_id: str | None = Form(default=None),
):
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一张图片。")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多上传 {MAX_BATCH_FILES} 张图片。",
        )

    try:
        threshold = validate_review_threshold(review_threshold)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    results = []
    succeeded = 0
    failed = 0

    for file in files:
        filename = file.filename or "oracle-image"
        try:
            content = await read_oracle_upload(file)
            result = await run_in_threadpool(
                recognize_and_store_oracle,
                content=content,
                filename=filename,
                content_type=file.content_type,
                review_threshold=threshold,
                source="batch",
                conversation_id=conversation_id,
            )
            results.append({"status": "success", **result})
            succeeded += 1
        except (
            HTTPException,
            OracleImageError,
            OracleModelUnavailableError,
            OracleRecognitionError,
        ) as error:
            detail = error.detail if isinstance(error, HTTPException) else str(error)
            results.append(
                {
                    "status": "failed",
                    "filename": filename,
                    "error": detail,
                }
            )
            failed += 1
        except Exception as error:
            results.append(
                {
                    "status": "failed",
                    "filename": filename,
                    "error": str(error),
                }
            )
            failed += 1
        finally:
            await file.close()

    return {
        "total": len(files),
        "succeeded": succeeded,
        "failed": failed,
        "review_threshold": threshold,
        "results": results,
    }


@app.get(
    "/oracle/candidates/{candidate_id}/{kind}",
    tags=["Oracle Recognition"],
    summary="Get A Retrieved Oracle Candidate Image",
)
def oracle_candidate_image(candidate_id: str, kind: Literal["glyph", "rubbing"]):
    try:
        content, media_type = get_candidate_image(candidate_id, kind)
    except OracleCandidateNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except OracleRetrievalUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get(
    "/oracle/review-summary",
    tags=["Oracle Recognition"],
    summary="Get Oracle Review Queue Summary",
)
def oracle_review_summary():
    return get_oracle_review_summary()


@app.get(
    "/oracle/records/export",
    tags=["Oracle Recognition"],
    summary="Export Oracle Recognition Records",
)
def export_oracle_records(
    format: Literal["json", "csv"] = Query(default="json"),
    review_status: str | None = Query(default=None),
    class_code: str | None = Query(default=None),
):
    try:
        records = list_oracle_recognition_records(
            review_status=review_status,
            class_code=class_code,
            limit=10_000,
            offset=0,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if format == "csv":
        return Response(
            content="\ufeff" + oracle_records_to_csv(records),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="oracle-records-{timestamp}.csv"'
                )
            },
        )

    return Response(
        content=oracle_records_to_json(records),
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="oracle-records-{timestamp}.json"'
            )
        },
    )


@app.get(
    "/oracle/records",
    tags=["Oracle Recognition"],
    summary="List Oracle Recognition Records",
)
def oracle_records(
    review_status: str | None = Query(default=None),
    class_code: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    try:
        records = list_oracle_recognition_records(
            review_status=review_status,
            class_code=class_code,
            limit=limit,
            offset=offset,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"items": records, "limit": limit, "offset": offset}


@app.get(
    "/oracle/records/{record_id}",
    tags=["Oracle Recognition"],
    summary="Get One Oracle Recognition Record",
)
def oracle_record(record_id: UUID):
    record = get_oracle_recognition_record(str(record_id))
    if record is None:
        raise HTTPException(status_code=404, detail="识别记录不存在。")
    return record


@app.get(
    "/oracle/records/{record_id}/image",
    tags=["Oracle Recognition"],
    summary="Get The Original Oracle Image",
)
def oracle_record_image(record_id: UUID):
    image = get_oracle_recognition_image(str(record_id))
    if image is None:
        raise HTTPException(status_code=404, detail="识别记录不存在。")
    return Response(
        content=image["content"],
        media_type=image["content_type"],
        headers={"Cache-Control": "private, max-age=300"},
    )


@app.patch(
    "/oracle/records/{record_id}/review",
    tags=["Oracle Recognition"],
    summary="Review An Oracle Recognition Record",
)
def review_oracle_record(
    record_id: UUID,
    request: OracleReviewRequest,
):
    try:
        record = update_oracle_review(
            str(record_id),
            review_status=request.status,
            review_notes=request.notes,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if record is None:
        raise HTTPException(status_code=404, detail="识别记录不存在。")
    return record


# ============================================================
# Agent API
# ============================================================

@app.post(
    "/chat",
    tags=["Agent"],
    summary="Run Agent Chat",
    description=(
        "Send a user message to the Agent. "
        "The request is associated with a Conversation ID, "
        "allowing PostgreSQL-backed multi-turn Agent Sessions. "
        "The Agent may invoke tools such as RAG retrieval "
        "or calculator depending on the request."
    ),
)
async def chat(
    request: ChatRequest
):
    """
    执行一次 Agent 对话。

    执行流程：

    1. 保存用户消息
    2. 自动生成 Conversation 标题
    3. 调用 Agent Harness
    4. Agent 执行 Tool Calling / RAG
    5. 保存 Assistant 回复
    6. 返回 Agent Result + Execution Trace
    """

    print(
        "Conversation ID:",
        request.conversation_id,
    )

    print(
        "User Input:",
        request.message,
    )

    if (
        request.oracle_image_id
        and not has_cached_oracle_image(
            request.oracle_image_id
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "甲骨文图片引用不存在或已过期，"
                "请重新上传并识别图片。"
            ),
        )

    # ========================================================
    # 1. 保存用户消息
    # ========================================================

    add_message(
        conversation_id=request.conversation_id,
        role="user",
        content=request.message,
    )

    print(
        "User message saved."
    )

    # ========================================================
    # 2. 自动生成 Conversation 标题
    # ========================================================

    title = generate_conversation_title(
        request.message
    )

    updated_conversation = (
        update_conversation_title(
            conversation_id=(
                request.conversation_id
            ),
            title=title,
        )
    )

    if updated_conversation:
        print(
            "Conversation title updated:",
            updated_conversation[
                "title"
            ],
        )

    # ========================================================
    # 3. 执行 Agent
    # ========================================================

    agent_input = request.message
    if request.oracle_image_id:
        agent_input = (
            f"{request.message}\n\n"
            "[ORACLE_IMAGE_ATTACHMENT]\n"
            f"image_id: {request.oracle_image_id}\n"
            "这是服务端验证的本次消息图片附件。"
            "必须调用 recognize_oracle_image，"
            "并且只能使用上面的 image_id。"
        )

    result = await harness.run(
        user_input=agent_input,
        conversation_id=(
            request.conversation_id
        ),
    )

    if request.oracle_image_id:
        agent_trace = {
            "run_id": result.get("run_id"),
            "status": result.get("status"),
            "latency": result.get("latency"),
            "agent_latency": result.get("agent_latency"),
            "tool_call_count": result.get("tool_call_count", 0),
            "tool_logs": result.get("tool_logs", []),
            "error": result.get("error"),
        }
        try:
            await run_in_threadpool(
                attach_oracle_agent_trace,
                request.oracle_image_id,
                conversation_id=request.conversation_id,
                agent_trace=agent_trace,
            )
        except Exception as error:
            print("Oracle record trace update failed:", error)

    # ========================================================
    # 4. 保存 Agent 回复
    # ========================================================

    if (
        result.get("status") == "success"
        and result.get("output")
    ):
        add_message(
            conversation_id=(
                request.conversation_id
            ),
            role="assistant",
            content=result["output"],
        )

        print(
            "Assistant message saved."
        )

    # ========================================================
    # 5. 返回 Agent Harness Result
    # ========================================================

    return result


# ============================================================
# Conversation API
# ============================================================

@app.post(
    "/conversations",
    tags=["Conversations"],
    summary="Create Conversation",
    description=(
        "Create a new conversation and return "
        "its Conversation ID."
    ),
)
def new_conversation():
    """
    创建新会话。
    """

    conversation = create_conversation(
        title="New Conversation"
    )

    return conversation


@app.get(
    "/conversations",
    tags=["Conversations"],
    summary="List Conversations",
    description=(
        "Return the conversation list stored "
        "in PostgreSQL."
    ),
)
def list_conversations():
    """
    获取历史会话列表。
    """

    conversations = (
        get_conversations()
    )

    return conversations


@app.get(
    "/conversations/{conversation_id}/messages",
    tags=["Conversations"],
    summary="Get Conversation Messages",
    description=(
        "Return all user and assistant messages "
        "belonging to the specified conversation."
    ),
)
def conversation_messages(
    conversation_id: UUID
):
    """
    获取指定 Conversation 的历史消息。
    """

    messages = get_messages(
        str(conversation_id)
    )

    return messages


@app.delete(
    "/conversations/{conversation_id}",
    tags=["Conversations"],
    summary="Delete Conversation",
    description=(
        "Delete the business conversation data "
        "and clear the corresponding Agent Session."
    ),
)
async def remove_conversation(
    conversation_id: UUID
):
    """
    删除指定 Conversation。

    同时执行：

    1. 删除 conversations / messages
    2. 清理对应 Agent Session
    3. 清除 Agent runtime context
    """

    conversation_id_str = str(
        conversation_id
    )

    # ========================================================
    # 1. 删除业务 Conversation
    # ========================================================

    deleted = delete_conversation(
        conversation_id_str
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found"
            ),
        )

    # ========================================================
    # 2. 清理 Agent Session
    # ========================================================

    session_cleared = True

    try:
        await harness.clear_session(
            conversation_id_str
        )

    except Exception as error:
        session_cleared = False

        print(
            "Agent Session clear failed:",
            error,
        )

    # ========================================================
    # 3. 返回删除结果
    # ========================================================

    return {
        "status": "success",
        "conversation_id": (
            conversation_id_str
        ),
        "session_cleared": (
            session_cleared
        ),
    }
