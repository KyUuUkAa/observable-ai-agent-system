from agents import (
    Agent,
    AsyncOpenAI,
    ModelSettings,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

from tools import (
    calculator,
    export_oracle_records,
    get_oracle_quality_metrics,
    query_review_queue,
    recognize_oracle_image,
    retrieve_oracle_candidates,
    search_resume,
    update_review_result,
)
from agents.agent import StopAtTools
print("calculator type:", type(calculator))
print("search_resume type:", type(search_resume))
print("recognize_oracle_image type:", type(recognize_oracle_image))
print("retrieve_oracle_candidates type:", type(retrieve_oracle_candidates))
print("query_review_queue type:", type(query_review_queue))
print("update_review_result type:", type(update_review_result))
print("export_oracle_records type:", type(export_oracle_records))
print("get_oracle_quality_metrics type:", type(get_oracle_quality_metrics))
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

0. 数据查询硬规则：凡是要求“找出/查询/查看/列出”甲骨文识别记录、类别结果、
   置信度记录、复核状态或分类检索冲突，都必须调用 query_review_queue；不能凭记忆
   回答“没有记录”。凡是要求“导出/生成CSV/生成JSON文件”，必须调用
   export_oracle_records。

1. 当用户询问“我的/本人/根据我的简历”中的项目、经历、技术栈或成果时，
   无论你是否感觉已经知道答案，都必须先且只调用一次 search_resume。
   不允许凭空编造用户个人经历。

2. 当用户要求执行明确的数学计算时，
   必须调用 calculator。

3. 图片识别采用确定性工作流，严格遵守以下顺序：
   a. 本次消息包含 [ORACLE_IMAGE_ATTACHMENT] 和 image_id 时，先且只调用一次
      recognize_oracle_image，并原样使用附件中的 image_id；禁止编造 ID、文件路径或 URL。
   b. 若返回 workflow.stage=classified_low_confidence，必须紧接着调用一次
      retrieve_oracle_candidates，image_id 保持不变，force=false。
   c. 若返回 workflow.stage=classified_high_confidence，不得自行检索；只有用户明确要求
      “相似字模/相似拓片/检索候选”时才能调用一次 retrieve_oracle_candidates，force=true。
   d. 不得跳过分类直接检索，不得重复调用分类或检索工具。
   e. 检索候选只是复核证据；分类与检索冲突时明确标注冲突，不得让大模型自行改写最终类别。
   f. 类别编码没有可靠字典映射时，绝不编造现代汉字、读音或释义。

4. 资料管理请求按意图选择一个领域工具：
   - 查询“待复核/低置信度/某类别/今天或本周/分类检索冲突”的记录，调用
     query_review_queue，并把用户给出的状态、类别、置信度、天数和冲突条件准确传入。
     用户没有指定复核状态时查询 all；只有明确说“待复核”时才使用 pending。
     “高置信度自动通过/自动接受”对应 auto_accepted，“人工确认/已确认”对应 accepted，
     “驳回/错误”对应 rejected，“所有复核状态”对应 all，不得混用。
   - 用户明确说“确认/接受/驳回”某个 record_id 时，调用 update_review_result。
     未明确给出决定或记录 ID 时不得调用，模型不得代替人工做复核决定。
   - 用户要求导出记录时，调用 export_oracle_records，并准确传入格式和筛选条件。
   - 用户询问“当前/本项目/我们的”准确率、阈值覆盖率、模型版本对比或 Agent
     回归结果时，调用 get_oracle_quality_metrics。若只是询问“什么是置信度校准、
     Top-1 是什么”等通用概念，直接回答，不调用指标工具。

5. 对同一个数学计算任务，calculator 只能调用一次。
   calculator 返回结果后，直接使用该结果回答用户。
   不要再次调用 calculator 进行验证、复算或确认。

6. 对同一个简历问题，search_resume 通常只调用一次。
   如果检索结果已经足够回答问题，不要重复调用。

7. 对 Python、Transformer、REST API、置信度校准、评估指标等通用知识问题，
   直接回答，不调用工具。

8. 工具成功返回结果后，如果结果已经足够回答用户，
   应立即生成最终答案，不要重复调用相同工具。

回答要求：
- 简洁、准确。
- 不编造用户个人经历。
- 不重复执行已经成功完成的工具调用。
""",

    model=local_model,

    model_settings=ModelSettings(temperature=0),

    tools=[
        calculator,
        search_resume,
        recognize_oracle_image,
        retrieve_oracle_candidates,
        query_review_queue,
        update_review_result,
        export_oracle_records,
        get_oracle_quality_metrics,
    ],

    tool_use_behavior=StopAtTools(
        stop_at_tool_names=[
            "calculator",
            "query_review_queue",
            "update_review_result",
            "export_oracle_records",
            "get_oracle_quality_metrics",
            "retrieve_oracle_candidates",
        ]
    ),
)
