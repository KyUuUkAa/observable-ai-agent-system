import json
import os
import time
import uuid

from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine

from agents import Runner
from agents.extensions.memory import SQLAlchemySession

from deterministic_routing import argument_hints_for_input, required_tool_for_input


# =========================================================
# 环境变量
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True
)


# =========================================================
# Agent Harness
# =========================================================

class AgentHarness:

    def __init__(
        self,
        agent,
        log_file="logs.jsonl"
    ):
        self.agent = agent

        # -----------------------------------------
        # 日志文件
        # -----------------------------------------

        self.log_file = str(
            BASE_DIR / log_file
        )

        # -----------------------------------------
        # PostgreSQL Agent Session
        # -----------------------------------------

        database_url = os.getenv(
            "AGENT_DATABASE_URL"
        )

        if not database_url:
            raise RuntimeError(
                "没有读取到 AGENT_DATABASE_URL，"
                "请检查项目根目录 .env 文件。"
            )

        if not database_url.startswith(
            "postgresql+asyncpg://"
        ):
            raise RuntimeError(
                "AGENT_DATABASE_URL 必须使用 "
                "postgresql+asyncpg://"
            )

        # 一个 Harness 共用一个 AsyncEngine
        self.engine = create_async_engine(
            database_url,
            pool_pre_ping=True,
        )

        print(
            "Agent Session Backend: PostgreSQL"
        )


    # =====================================================
    # 创建 Agent Session
    # =====================================================

    def get_session(
        self,
        conversation_id: str
    ) -> SQLAlchemySession:

        return SQLAlchemySession(
            conversation_id,

            engine=self.engine,

            # agent_sessions / agent_messages
            # 已经存在，所以这里不自动创建
            create_tables=False,

            # 中文 JSON 在数据库里保持可读
            ensure_ascii=False,
        )


    # =====================================================
    # 保存 Harness 日志
    # =====================================================

    def save_log(
        self,
        log_data: dict
    ):

        with open(
            self.log_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(
                    log_data,
                    ensure_ascii=False
                )
                + "\n"
            )


    # =====================================================
    # 提取 Tool Call Trace
    # =====================================================

    def extract_tool_logs(
        self,
        result
    ):

        tool_logs = []

        for item in result.new_items:

            item_type = getattr(
                item,
                "type",
                None
            )

            # -----------------------------------------
            # Tool Call
            # -----------------------------------------

            if item_type == "tool_call_item":

                raw_item = getattr(
                    item,
                    "raw_item",
                    None
                )

                tool_logs.append({
                    "type":
                        "tool_call",

                    "tool_name":
                        getattr(
                            raw_item,
                            "name",
                            None
                        ),

                    "arguments":
                        getattr(
                            raw_item,
                            "arguments",
                            None
                        )
                })


            # -----------------------------------------
            # Tool Output
            # -----------------------------------------

            elif (
                item_type
                == "tool_call_output_item"
            ):

                tool_logs.append({
                    "type":
                        "tool_output",

                    "output":
                        str(
                            getattr(
                                item,
                                "output",
                                None
                            )
                        )
                })

        return tool_logs


    # =====================================================
    # 清理 Agent Session
    # =====================================================

    async def clear_session(
        self,
        conversation_id: str
    ):
        """
        清理指定 Conversation 对应的
        PostgreSQL Agent Session。

        会清理 SQLAlchemySession 使用的
        agent_sessions / agent_messages 中
        当前 session_id 对应的数据。
        """

        session = self.get_session(
            conversation_id
        )

        await session.clear_session()

        print(
            "PostgreSQL Agent Session cleared:",
            conversation_id
        )


    # =====================================================
    # Agent Run
    # =====================================================

    async def run(
        self,
        user_input: str,
        conversation_id: str
    ):

        run_id = str(
            uuid.uuid4()
        )

        start_time = time.time()

        print(
            "\n=============================="
        )

        print(
            f"Run ID: {run_id}"
        )

        print(
            f"Conversation ID: "
            f"{conversation_id}"
        )

        print(
            f"User Input: {user_input}"
        )

        print(
            "=============================="
        )


        # -----------------------------------------
        # PostgreSQL Agent Session
        # -----------------------------------------

        session = self.get_session(
            conversation_id
        )
        agent_start = time.perf_counter()

        try:

            # =====================================
            # 执行 Agent
            # =====================================

            required_tool = required_tool_for_input(user_input)
            routed_agent = self.agent
            if required_tool:
                argument_hints = argument_hints_for_input(user_input)
                argument_constraint = (
                    "\nRequired argument mapping: " + "; ".join(argument_hints)
                    if argument_hints
                    else ""
                )
                route_instruction = (
                    "\n\n[RUNTIME ROUTE CONSTRAINT]\n"
                    f"The current request has been deterministically routed to "
                    f"{required_tool}. Your next action MUST be calling exactly this "
                    "tool. Do not answer, describe the call, or choose another tool "
                    "before that call. Extract arguments from the original user input."
                    + argument_constraint
                )
                allowed_tool_names = {required_tool}
                if required_tool == "recognize_oracle_image":
                    allowed_tool_names.add("retrieve_oracle_candidates")
                routed_agent = self.agent.clone(
                    instructions=str(self.agent.instructions) + route_instruction,
                    tools=[
                        tool
                        for tool in self.agent.tools
                        if tool.name in allowed_tool_names
                    ],
                    model_settings=self.agent.model_settings.resolve(
                        {"tool_choice": required_tool}
                    )
                )
                print("Deterministic tool route:", required_tool)

            result = await Runner.run(
                routed_agent,
                user_input,
                session=session
            )
            if required_tool:
                first_tool_names = {
                    log.get("tool_name")
                    for log in self.extract_tool_logs(result)
                    if log.get("type") == "tool_call"
                }
                if required_tool not in first_tool_names:
                    print("Required tool was skipped; retrying once:", required_tool)
                    result = await Runner.run(
                        routed_agent,
                        (
                            "[RUNTIME RETRY] The previous response skipped the "
                            f"required tool. Call {required_tool} now using the "
                            "arguments from the user's preceding request. Do not "
                            "describe the call."
                        ),
                        session=session,
                    )
            agent_latency = (
             time.perf_counter()
                - agent_start
            )
            print(
              "[PERF] Agent Runner:",
                f"{agent_latency:.4f}s"
            )

            # =====================================
            # Latency
            # =====================================

            latency = (
                time.time()
                - start_time
            )


            # =====================================
            # Tool Trace
            # =====================================

            tool_logs = (
                self.extract_tool_logs(
                    result
                )
            )


            tool_call_count = len([
                item
                for item in tool_logs
                if (
                    item["type"]
                    == "tool_call"
                )
            ])


            # =====================================
            # Success Log
            # =====================================

            log_data = {

                "run_id":
                    run_id,

                "conversation_id":
                    conversation_id,

                "timestamp":
                    datetime.now()
                    .isoformat(),

                "status":
                    "success",

                "input":
                    user_input,

                "output":
                    result.final_output,

                "latency":
                    round(
                        latency,
                        4
                    ),
                "agent_latency":
                    round(agent_latency, 4),


                "tool_call_count":
                    tool_call_count,

                "tool_logs":
                    tool_logs,
            }


            self.save_log(
                log_data
            )


            print(
                "Agent Run Success"
            )

            print(
                f"Latency: "
                f"{round(latency, 2)}s"
            )

            print(
                f"Tool Calls: "
                f"{tool_call_count}"
            )


            return log_data


        except Exception as error:

            # =====================================
            # Failed Run
            # =====================================

            latency = (
                time.time()
                - start_time
            )


            log_data = {

                "run_id":
                    run_id,

                "conversation_id":
                    conversation_id,

                "timestamp":
                    datetime.now()
                    .isoformat(),

                "status":
                    "failed",

                "input":
                    user_input,

                "output":
                    None,

                "latency":
                    round(
                        latency,
                        4
                    ),

                "tool_call_count":
                    0,

                "tool_logs":
                    [],

                "error":
                    str(error),
            }


            self.save_log(
                log_data
            )


            print(
                "Agent Run Failed:"
            )

            print(
                str(error)
            )


            return log_data


    # =====================================================
    # 关闭数据库 Engine
    # =====================================================

    async def close(self):
        """
        应用退出时关闭 PostgreSQL AsyncEngine。
        """

        await self.engine.dispose()

        print(
            "Agent PostgreSQL Engine closed."
        )
