import hashlib
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb
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


# ==============================
# 甲骨文识别记录
# ==============================

ORACLE_REVIEW_STATUSES = {
    "pending",
    "auto_accepted",
    "accepted",
    "rejected",
}


def _serialize_oracle_record(row):
    if row is None:
        return None

    return {
        "id": str(row[0]),
        "conversation_id": str(row[1]) if row[1] else None,
        "image_reference_id": row[2],
        "original_filename": row[3],
        "content_type": row[4],
        "image_sha256": row[5],
        "image_width": row[6],
        "image_height": row[7],
        "top1_class_id": row[8],
        "top1_class_code": row[9],
        "top1_confidence": float(row[10]),
        "top5": row[11],
        "model_info": row[12],
        "model_version": row[13],
        "review_threshold": float(row[14]),
        "review_status": row[15],
        "review_notes": row[16],
        "reviewed_at": row[17].isoformat() if row[17] else None,
        "execution_trace": row[18],
        "source": row[19],
        "created_at": row[20].isoformat(),
        "updated_at": row[21].isoformat(),
    }


ORACLE_RECORD_SELECT = """
    id,
    conversation_id,
    image_reference_id,
    original_filename,
    content_type,
    image_sha256,
    image_width,
    image_height,
    top1_class_id,
    top1_class_code,
    top1_confidence,
    top5,
    model_info,
    model_version,
    review_threshold,
    review_status,
    review_notes,
    reviewed_at,
    execution_trace,
    source,
    created_at,
    updated_at
"""


def create_oracle_recognition_record(
    *,
    image_content: bytes,
    original_filename: str,
    content_type: str | None,
    image_reference_id: str,
    recognition: dict,
    review_threshold: float,
    source: str,
    execution_trace: dict,
    conversation_id: str | None = None,
):
    """Persist one recognition result and its original image."""

    prediction = recognition["prediction"]
    image = recognition["image"]
    model_info = recognition["model"]
    confidence = float(prediction["confidence"])
    review_status = (
        "pending"
        if confidence < review_threshold
        else "auto_accepted"
    )
    model_version = (
        model_info.get("checkpoint_sha256")
        or model_info.get("version")
        or "unknown"
    )
    record_id = uuid.uuid4()

    sql = f"""
    INSERT INTO oracle_recognition_records (
        id,
        conversation_id,
        image_reference_id,
        original_filename,
        content_type,
        image_sha256,
        image_data,
        image_width,
        image_height,
        top1_class_id,
        top1_class_code,
        top1_confidence,
        top5,
        model_info,
        model_version,
        review_threshold,
        review_status,
        execution_trace,
        source
    )
    VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    RETURNING {ORACLE_RECORD_SELECT};
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    record_id,
                    conversation_id,
                    image_reference_id,
                    original_filename,
                    content_type or "application/octet-stream",
                    hashlib.sha256(image_content).hexdigest(),
                    image_content,
                    image["width"],
                    image["height"],
                    prediction["class_id"],
                    prediction["class_code"],
                    confidence,
                    Jsonb(recognition["top5"]),
                    Jsonb(model_info),
                    model_version,
                    review_threshold,
                    review_status,
                    Jsonb(execution_trace),
                    source,
                ),
            )
            row = cursor.fetchone()

    return _serialize_oracle_record(row)


def list_oracle_recognition_records(
    *,
    review_status: str | None = None,
    class_code: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    conditions = []
    parameters = []

    if review_status:
        if review_status not in ORACLE_REVIEW_STATUSES:
            raise ValueError("不支持的复核状态。")
        conditions.append("review_status = %s")
        parameters.append(review_status)

    if class_code:
        conditions.append("top1_class_code = %s")
        parameters.append(class_code)

    where_clause = (
        "WHERE " + " AND ".join(conditions)
        if conditions
        else ""
    )
    sql = f"""
    SELECT {ORACLE_RECORD_SELECT}
    FROM oracle_recognition_records
    {where_clause}
    ORDER BY created_at DESC
    LIMIT %s OFFSET %s;
    """
    parameters.extend([limit, offset])

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, parameters)
            rows = cursor.fetchall()

    return [_serialize_oracle_record(row) for row in rows]


def get_oracle_recognition_record(record_id: str):
    sql = f"""
    SELECT {ORACLE_RECORD_SELECT}
    FROM oracle_recognition_records
    WHERE id = %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (record_id,))
            row = cursor.fetchone()

    return _serialize_oracle_record(row)


def get_oracle_recognition_image(record_id: str):
    sql = """
    SELECT image_data, content_type, original_filename
    FROM oracle_recognition_records
    WHERE id = %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (record_id,))
            row = cursor.fetchone()

    if row is None:
        return None
    return {
        "content": bytes(row[0]),
        "content_type": row[1],
        "filename": row[2],
    }


def update_oracle_review(
    record_id: str,
    *,
    review_status: str,
    review_notes: str | None,
):
    if review_status not in {"pending", "accepted", "rejected"}:
        raise ValueError("复核状态必须是 pending、accepted 或 rejected。")

    sql = f"""
    UPDATE oracle_recognition_records
    SET
        review_status = %s,
        review_notes = %s,
        reviewed_at = CASE
            WHEN %s = 'pending' THEN NULL
            ELSE CURRENT_TIMESTAMP
        END,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = %s
    RETURNING {ORACLE_RECORD_SELECT};
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    review_status,
                    review_notes,
                    review_status,
                    record_id,
                ),
            )
            row = cursor.fetchone()

    return _serialize_oracle_record(row)


def attach_oracle_agent_trace(
    image_reference_id: str,
    *,
    conversation_id: str,
    agent_trace: dict,
):
    """Attach the later Agent run to the record created during upload."""

    sql = f"""
    UPDATE oracle_recognition_records
    SET
        conversation_id = %s,
        execution_trace = execution_trace || %s,
        updated_at = CURRENT_TIMESTAMP
    WHERE image_reference_id = %s
    RETURNING {ORACLE_RECORD_SELECT};
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                sql,
                (
                    conversation_id,
                    Jsonb({"agent": agent_trace}),
                    image_reference_id,
                ),
            )
            row = cursor.fetchone()

    return _serialize_oracle_record(row)


def get_oracle_review_summary():
    sql = """
    SELECT review_status, COUNT(*)
    FROM oracle_recognition_records
    GROUP BY review_status;
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()

    counts = {status: 0 for status in ORACLE_REVIEW_STATUSES}
    for status, count in rows:
        counts[status] = count
    counts["total"] = sum(counts.values())
    return counts

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
