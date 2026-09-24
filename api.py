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

from uuid import UUID

from fastapi import FastAPI, File, HTTPException, UploadFile
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

from database import (
    create_conversation,
    get_conversations,
    add_message,
    get_messages,
    update_conversation_title,
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
    version="1.0.0",
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
    file: UploadFile = File(...)
):
    if (
        file.content_type
        and not file.content_type.startswith("image/")
        and file.content_type != "application/octet-stream"
    ):
        raise HTTPException(
            status_code=400,
            detail="只支持图片文件。",
        )

    try:
        content = await file.read(
            MAX_ORACLE_UPLOAD_BYTES + 1
        )
    finally:
        await file.close()

    if len(content) > MAX_ORACLE_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="图片不能超过 10 MB。",
        )

    try:
        result = await run_in_threadpool(
            recognize_oracle_image,
            content,
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

    return {
        "filename": file.filename,
        "image_id": cache_oracle_image(content),
        **result,
    }


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
