from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

from tools import calculator, search_resume
from agents.agent import StopAtTools
print("calculator type:", type(calculator))
print("search_resume type:", type(search_resume))
# ==============================
# 关闭 OpenAI tracing
# ==============================

set_tracing_disabled(True)


# ==============================
# Ollama Client
# ==============================

ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


# ==============================
# Local Qwen Model
# ==============================

local_model = OpenAIChatCompletionsModel(
    model="qwen3:4b",
    openai_client=ollama_client
)


# ==============================
# Agent
# ==============================

agent = Agent(
    name="Career Agent",

    instructions = """
你是一个 AI Agent，需要根据用户问题决定是否调用工具。

工具使用规则：

1. 当用户询问自己的简历、项目经历、技术栈、项目成果时，
   必须调用 search_resume。
   不允许凭空编造用户个人经历。

2. 当用户要求执行明确的数学计算时，
   必须调用 calculator。

3. 对同一个数学计算任务，calculator 只能调用一次。
   calculator 返回结果后，直接使用该结果回答用户。
   不要再次调用 calculator 进行验证、复算或确认。

4. 对同一个简历问题，search_resume 通常只调用一次。
   如果检索结果已经足够回答问题，不要重复调用。

5. 对 Python、Transformer、REST API 等通用知识问题，
   直接回答，不调用工具。

6. 工具成功返回结果后，如果结果已经足够回答用户，
   应立即生成最终答案，不要重复调用相同工具。

回答要求：
- 简洁、准确。
- 不编造用户个人经历。
- 不重复执行已经成功完成的工具调用。
""",

    model=local_model,

    tools=[
        calculator,
        search_resume,
    ],

    tool_use_behavior=StopAtTools(
        stop_at_tool_names=[
            "calculator"
        ]
    ),
)