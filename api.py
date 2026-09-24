from dotenv import load_dotenv
from database import create_conversation, get_conversations
load_dotenv(override=True)
from uuid import UUID
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from agent import agent
from harness import AgentHarness

from database import (
    create_conversation,
    get_conversations,
    add_message,
    get_messages,
    update_conversation_title,
    delete_conversation,
)
app = FastAPI(
    title="Career Agent API",
    description="AI Career Agent Backend",
    version="0.1.0"
)


# ==============================
# CORS
# ==============================

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


harness = AgentHarness(agent)


class ChatRequest(BaseModel):
    conversation_id: str
    message: str



@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/chat")
async def chat(
    request: ChatRequest
):

    print(
        "Conversation ID:",
        request.conversation_id
    )

    print(
        "User Input:",
        request.message
    )

    # ==============================
    # 1. 保存用户消息
    # ==============================

    add_message(
        conversation_id=request.conversation_id,
        role="user",
        content=request.message
    )
    # ==============================
# 自动生成 Conversation 标题
# ==============================

    title = generate_conversation_title(
        request.message
    )

    updated_conversation = (
        update_conversation_title(
            conversation_id=request.conversation_id,
            title=title
        )
    )

    if updated_conversation:

        print(
            "Conversation title updated:",
            updated_conversation["title"]
        )

    print(
        "User message saved."
    )


    # ==============================
    # 2. 执行 Agent
    # ==============================

    result = await harness.run(
        user_input=request.message,
        conversation_id=request.conversation_id
    )


    # ==============================
    # 3. 保存 Agent 回复
    # ==============================

    if (
        result.get("status") == "success"
        and result.get("output")
    ):

        add_message(
            conversation_id=request.conversation_id,
            role="assistant",
            content=result["output"]
        )

        print(
            "Assistant message saved."
        )

    return result

@app.post("/conversations")
def new_conversation():

    conversation = create_conversation(
        title="New Conversation"
    )

    return conversation


@app.get("/conversations")
def list_conversations():

    conversations = get_conversations()

    return conversations

@app.get("/conversations/{conversation_id}/messages")
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


def generate_conversation_title(
    message: str
) -> str:
    """
    根据首条用户消息生成简短会话标题。
    当前使用轻量规则，后续可替换为 LLM Title Generator。
    """

    text = message.strip()

    lower_text = text.lower()


    # YOLO
    if "yolo" in lower_text:

        if (
            "优化" in text
            or "改进" in text
        ):
            return "YOLO 项目优化"

        return "YOLO 项目介绍"


    # Agent
    if (
        "agent" in lower_text
        or "智能体" in text
    ):
        return "Agent 项目介绍"


    # 大模型
    if (
        "大模型" in text
        or "llm" in lower_text
    ):
        return "大模型应用项目"


    # 多视图聚类
    if (
        "多视图" in text
        or "聚类" in text
    ):
        return "多视图聚类项目"


    # 甲骨文
    if "甲骨文" in text:
        return "甲骨文项目"


    # 简历
    if "简历" in text:
        return "简历分析"


    # ==============================
    # 通用 fallback
    # ==============================

    title = text

    # 去掉常见问句结尾
    for suffix in [
        "？",
        "?",
        "。",
        "！",
        "!"
    ]:
        title = title.rstrip(suffix)

    # 防止标题过长
    max_length = 20

    if len(title) > max_length:
        title = (
            title[:max_length]
            + "..."
        )

    return title



@app.delete("/conversations/{conversation_id}")
def remove_conversation(
    conversation_id: UUID
):
    """
    删除指定 Conversation。
    """

    deleted = delete_conversation(
        str(conversation_id)
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    return {
        "status": "success",
        "conversation_id": str(
            conversation_id
        )
    }


@app.delete("/conversations/{conversation_id}")
async def remove_conversation(
    conversation_id: UUID
):
    conversation_id_str = str(
        conversation_id
    )

    deleted = delete_conversation(
        conversation_id_str
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    session_cleared = True

    try:
        await harness.clear_session(
            conversation_id_str
        )

    except Exception as error:
        session_cleared = False

        print(
            "Agent Session clear failed:",
            error
        )

    return {
        "status": "success",
        "conversation_id":
            conversation_id_str,
        "session_cleared":
            session_cleared,
    }