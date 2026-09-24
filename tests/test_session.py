import asyncio
import os

from dotenv import load_dotenv
from agents.extensions.memory import SQLAlchemySession


load_dotenv()


async def main():
    database_url = os.getenv(
        "AGENT_DATABASE_URL"
    )

    if not database_url:
        raise RuntimeError(
            "没有读取到 AGENT_DATABASE_URL"
        )

    print(
        "开始连接 PostgreSQL SQLAlchemySession..."
    )

    session = SQLAlchemySession.from_url(
        "sqlalchemy-test-session",
        url=database_url,

        # 开发阶段让 SDK 自动建表
        create_tables=True,

        # 中文 JSON 在数据库里更容易阅读
        ensure_ascii=False,
    )

    # 调一次 Session API，
    # 触发表结构初始化
    items = await session.get_items()

    print(
        "SQLAlchemySession 连接成功"
    )

    print(
        "当前 Session items:",
        items
    )

    # 关闭底层 AsyncEngine
    await session.engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())