import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
import uuid

# ==============================
# 加载 .env
# ==============================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True
)


# ==============================
# 数据库连接
# ==============================

def get_connection():

    password = os.getenv(
        "POSTGRES_PASSWORD"
    )

    if not password:
        raise RuntimeError(
            "没有读取到 POSTGRES_PASSWORD，"
            "请检查 .env 文件。"
        )

    return psycopg.connect(
        host=os.getenv(
            "POSTGRES_HOST",
            "127.0.0.1"
        ),

        port=os.getenv(
            "POSTGRES_PORT",
            "5432"
        ),

        dbname=os.getenv(
            "POSTGRES_DB"
        ),

        user=os.getenv(
            "POSTGRES_USER"
        ),

        password=password,
    )
def create_conversation(title: str):
    """
    创建一条 conversation 记录。
    """

    conversation_id = uuid.uuid4()

    sql = """
    INSERT INTO conversations (
        id,
        title
    )
    VALUES (
        %s,
        %s
    )
    RETURNING
        id,
        title,
        created_at,
        updated_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                sql,
                (
                    conversation_id,
                    title
                )
            )

            row = cursor.fetchone()

    return {
        "id": str(row[0]),
        "title": row[1],
        "created_at": row[2].isoformat(),
        "updated_at": row[3].isoformat(),
    }

# ==============================
# 测试连接
# ==============================

def test_connection():

    print(
        ".env 路径：",
        ENV_PATH
    )

    print(
        ".env 是否存在：",
        ENV_PATH.exists()
    )

    print(
        "POSTGRES_PASSWORD 已加载：",
        bool(
            os.getenv(
                "POSTGRES_PASSWORD"
            )
        )
    )

    with get_connection() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                "SELECT current_database();"
            )

            row = cursor.fetchone()

            print(
                "PostgreSQL连接成功：",
                row[0]
            )
def get_conversations():
    """
    获取所有会话。
    """

    sql = """
    SELECT
        id,
        title,
        created_at,
        updated_at
    FROM conversations
    ORDER BY created_at DESC;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:

            cursor.execute(sql)

            rows = cursor.fetchall()

    conversations = []

    for row in rows:
        conversations.append({
            "id": str(row[0]),
            "title": row[1],
            "created_at": row[2].isoformat(),
            "updated_at": row[3].isoformat(),
        })

    return conversations

def add_message(
    conversation_id: str,
    role: str,
    content: str
):
    """
    向指定 Conversation 写入一条聊天消息。

    role:
        user
        assistant
    """

    if role not in ("user", "assistant"):
        raise ValueError(
            "role 必须是 'user' 或 'assistant'"
        )

    sql = """
    INSERT INTO messages (
        conversation_id,
        role,
        content
    )
    VALUES (
        %s,
        %s,
        %s
    )
    RETURNING
        id,
        conversation_id,
        role,
        content,
        created_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                sql,
                (
                    conversation_id,
                    role,
                    content
                )
            )

            row = cursor.fetchone()

    return {
        "id": row[0],
        "conversation_id": str(row[1]),
        "role": row[2],
        "content": row[3],
        "created_at": row[4].isoformat(),
    }
def get_messages(
    conversation_id: str
):
    """
    获取指定 Conversation 的全部聊天消息。
    """

    sql = """
    SELECT
        id,
        conversation_id,
        role,
        content,
        created_at
    FROM messages
    WHERE conversation_id = %s
    ORDER BY id ASC;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                sql,
                (conversation_id,)
            )

            rows = cursor.fetchall()

    messages = []

    for row in rows:
        messages.append({
            "id": row[0],
            "conversation_id": str(row[1]),
            "role": row[2],
            "content": row[3],
            "created_at": row[4].isoformat(),
        })

    return messages


def update_conversation_title(
    conversation_id: str,
    title: str
):
    """
    仅当当前标题还是 New Conversation 时更新标题。
    """

    sql = """
    UPDATE conversations
    SET
        title = %s,
        updated_at = CURRENT_TIMESTAMP
    WHERE
        id = %s
        AND title = 'New Conversation'
    RETURNING
        id,
        title,
        created_at,
        updated_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                sql,
                (
                    title,
                    conversation_id
                )
            )

            row = cursor.fetchone()

    if row is None:
        return None

    return {
        "id": str(row[0]),
        "title": row[1],
        "created_at": row[2].isoformat(),
        "updated_at": row[3].isoformat(),
    }


def delete_conversation(
    conversation_id: str
) -> bool:
    """
    删除 Conversation。

    messages 表通过 ON DELETE CASCADE
    自动删除对应聊天消息。
    """

    sql = """
    DELETE FROM conversations
    WHERE id = %s
    RETURNING id;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                sql,
                (conversation_id,)
            )

            row = cursor.fetchone()

    return row is not None

if __name__ == "__main__":

    # ==============================
    # 获取最近一个 Conversation
    # ==============================

    conversations = get_conversations()

    if not conversations:
        print("当前没有 Conversation")
        raise SystemExit

    conversation_id = conversations[0]["id"]

    print(
        "测试 Conversation ID:",
        conversation_id
    )


    # ==============================
    # 写入 user 消息
    # ==============================

    user_message = add_message(
        conversation_id=conversation_id,
        role="user",
        content="我的YOLO项目主要做了什么？"
    )

    print("\n写入 User 消息成功：")
    print(user_message)


    # ==============================
    # 写入 assistant 消息
    # ==============================

    assistant_message = add_message(
        conversation_id=conversation_id,
        role="assistant",
        content="YOLO项目主要完成了数据清洗、模型训练、推理以及误检漏检分析。"
    )

    print("\n写入 Assistant 消息成功：")
    print(assistant_message)


    # ==============================
    # 读取消息
    # ==============================

    messages = get_messages(
        conversation_id
    )

    print("\n当前 Conversation 消息：")

    for message in messages:
        print(
            f"[{message['role']}] "
            f"{message['content']}"
        )