# Agent Evaluation & Performance Optimization

## 1. Overview

本项目针对基于 **OpenAI Agents SDK + Ollama + Qwen3:4B** 构建的本地 Agent 应用系统，设计并实现了一套可重复执行的 Evaluation Pipeline，用于系统评估：

- Tool Selection Accuracy
- Duplicate Tool Calling
- End-to-End Latency
- Tool Execution Latency
- RAG Retrieval Quality
- Hybrid Retrieval Optimization
- Multi-turn Session Growth
- PostgreSQL Session Storage Composition

系统当前主要由以下模块组成：

```text
User
  ↓
Vue Frontend
  ↓
FastAPI Backend
  ↓
Agent Harness
  ↓
OpenAI Agents SDK
  ↓
Ollama / Qwen3:4B
  ↓
Function Tools
  ├── search_resume
  ├── calculator
  ├── recognize_oracle_image
  ├── retrieve_oracle_candidates
  ├── query_review_queue
  ├── update_review_result
  ├── export_oracle_records
  └── get_oracle_quality_metrics
  ↓
PostgreSQL Session
```

其中：

- **OpenAI Agents SDK**：负责 Agent 编排、Tool Calling 和 Session 管理；
- **Qwen3:4B**：通过 Ollama 进行本地部署；
- **BAAI/bge-small-zh-v1.5**：负责 RAG Embedding；
- **FastAPI**：提供后端接口；
- **Vue**：实现聊天界面和 Execution Trace；
- **PostgreSQL**：存储业务会话和 Agent Session；
- **SQLAlchemySession**：用于 OpenAI Agents SDK 多轮上下文持久化。

为了避免只关注模型回答效果而忽略系统执行质量，本项目将 Agent Evaluation 拆分为：

```text
Tool Routing
↓
Execution Latency
↓
Retrieval Quality
↓
Session Growth
```

形成完整的 Agent 工程评估链路。

---

## Automated Regression Runner

The repeatable regression entry point is:

```powershell
python -m evaluation.agent_regression --fail-on-regression --fail-on-case-failure
```

It implements one closed pipeline:

```text
read versioned JSON cases
        ↓
run the real Agent in isolated conversations
        ↓
parse tool arguments and outputs from Execution Trace
        ↓
classify correct / false / missed / wrong / duplicate calls
        ↓
verify filter arguments and answer safety assertions
        ↓
aggregate latency, tool errors, and run failures
        ↓
compare case-level results and pass rate with the baseline
        ↓
write JSON + Markdown reports
```

Inputs and outputs:

```text
evaluation/cases/tool_routing.json       # editable test cases
evaluation/baselines/tool_routing.json   # versioned accepted baseline
reports/agent_regression/                # ignored generated reports
```

The current suite contains 60 cases. Each case declares `expected_tools`, including an empty list for a no-tool query, and may declare partial `expected_tool_arguments`, expected error paths, required answer terms, or forbidden answer terms. The report preserves the raw tool trace needed to diagnose wrong arguments, repeated calls, tool errors, and routing regressions. Temporary PostgreSQL Agent sessions are removed after each case; use `--keep-sessions` only when database-level debugging is needed. Use `--case-id` or `--tag` for focused diagnosis, and update the baseline only after reviewing a complete successful run with `--write-baseline`.

For explicit source-of-truth intents, the Harness applies a deterministic routing guard: it constrains the run to the required tool schema and retries once if the local model skips the required call. General questions are not forced. This guard makes route failures reproducible without allowing the LLM to answer database queries from memory.

The accepted 60-case baseline currently passes **60/60** with 52 tool calls, zero wrong/missed/duplicate calls, 6.96 s mean latency and 14.46 s P95 latency on the local Qwen3:4B environment. The committed baseline is sanitized and intentionally excludes tool outputs and recognition-record data.

The legacy command remains available and delegates to the same runner:

```powershell
python -m evaluation.tool_selection_eval
```

---

# 2. Evaluation Architecture

项目通过自定义 `AgentHarness` 对每次 Agent Run 进行统一记录。

主要记录信息包括：

```text
Run ID
Timestamp
Conversation ID
Success / Failed
Agent Latency
Total Latency
Tool Name
Tool Arguments
Tool Output
Execution Trace
```

运行日志保存为：

```text
logs.jsonl
```

同时通过独立 Evaluation Script 对不同模块进行离线测试：

```text
tool_selection_eval.py
latency_eval.py
retrieval_eval.py
session_growth_eval.py
```

实验输出统一保存至：

```text
reports/
```

包括 JSON 和 CSV 结果，便于后续进行复现和对比。

---

# 3. Tool Selection Evaluation

## 3.1 Evaluation Objective

Tool Calling 是 Agent 系统区别于普通 Chatbot 的核心能力之一。

本实验主要评估：

> Agent 是否能够在合适的问题上选择正确 Tool，同时避免不必要、错误或重复的 Tool Calling。

测试类别包括：

| Task Type | Expected Behavior |
|---|---|
| Resume / Project Query | `search_resume` |
| Mathematical Calculation | `calculator` |
| General Knowledge | No Tool |

构建 12 条测试样本：

```text
4 × search_resume
4 × calculator
4 × no-tool
```

每条测试使用独立 Conversation ID，避免历史 Session 对 Tool Selection 产生干扰。

---

## 3.2 Evaluation Metrics

Tool Selection Evaluation 将结果分为：

```text
correct
false_tool_call
missed_tool_call
wrong_tool
duplicate_tool_call
```

判断逻辑：

- 正确调用一次目标 Tool：`correct`
- 不需要 Tool 却调用 Tool：`false_tool_call`
- 应调用 Tool 但没有调用：`missed_tool_call`
- 调用了错误 Tool：`wrong_tool`
- 同一个任务重复调用正确 Tool：`duplicate_tool_call`

---

## 3.3 Initial Result

初始测试显示：

- `search_resume` 路由正常；
- `no-tool` 路由正常；
- Calculator Tool 能够被正确选择；
- 但 Calculator 出现明显重复调用。

典型情况：

```text
Expected:
calculator

Actual:
calculator
calculator
```

4 条 Calculator 测试中：

```text
3 / 4
```

出现重复调用。

因此：

```text
Calculator Duplicate Tool Call Rate
= 75%
```

这说明问题并不是：

```text
Tool Selection Error
```

而是：

```text
Tool Execution Loop Redundancy
```

即 Agent 已经获得正确计算结果，但模型仍然再次决定调用相同 Tool。

---

# 4. Prompt Constraint Experiment

## 4.1 Optimization Attempt

首先通过 Agent Instructions 添加软约束：

```text
数学计算问题调用 calculator。

同一个计算任务只允许调用一次 calculator。

calculator 返回结果后直接使用结果回答，
不要再次调用 calculator 验证结果。

search_resume 通常只调用一次。

如果 Tool Result 已经能够回答问题，
立即生成最终回答，不重复执行 Tool。
```

---

## 4.2 Result

加入 Prompt Constraint 后：

```text
Calculator Duplicate Tool Calls
仍然为 3 / 4
```

即重复调用率仍约：

```text
75%
```

因此实验表明：

> Prompt 级软约束能够描述期望行为，但无法稳定控制本地模型的 Tool Calling 执行次数。

这说明仅依赖 Prompt 无法彻底解决执行链路中的重复调用问题。

---

# 5. Runtime Termination Optimization

## 5.1 Optimization Method

进一步从 Runtime 层控制 Agent 执行过程。

针对 Calculator Tool 配置：

```python
from agents.agent import StopAtTools

tool_use_behavior = StopAtTools(
    stop_at_tool_names=["calculator"]
)
```

执行逻辑由：

```text
User
↓
LLM
↓
calculator
↓
Tool Result
↓
LLM
↓
可能再次 calculator
```

改为：

```text
User
↓
LLM
↓
calculator
↓
Tool Result
↓
END
```

Calculator 返回值直接作为最终结果，避免 Tool Output 再次进入模型决策循环。

需要注意：

```text
search_resume
```

没有配置 `StopAtTools`。

原因是 RAG Tool 返回的是检索 Evidence，仍需要 LLM 根据检索结果进行自然语言组织。

---

## 5.2 Tool Selection Result

Runtime Termination 优化后：

```text
Total Cases:            12
Correct:                12
Accuracy:               100%
False Tool Call:        0
Missed Tool Call:       0
Wrong Tool:             0
Duplicate Tool Call:    0
```

Calculator Duplicate Tool Call：

```text
Before: 75%
After:   0%
```

即：

```text
75% → 0%
```

Runtime-level termination 成功消除了当前测试集中的 Calculator 重复调用。

---

# 6. Latency Evaluation

## 6.1 Evaluation Setup

为了分析 Agent 不同执行路径的性能，构建三类 Latency Benchmark：

```text
No Tool
RAG Tool
Calculator Tool
```

每种任务重复执行多次，并使用新的 Conversation ID，避免 Session 历史长度成为干扰变量。

---

## 6.2 Initial Benchmark

优化前测试结果：

| Task | Mean Latency |
|---|---:|
| No Tool | 4.64 s |
| RAG Tool | 15.54 s |
| Calculator | 22.12 s |

其中：

```text
Calculator
```

具有最高平均延迟。

同时 Tool Selection Evaluation 已经发现其存在重复 Tool Calling，因此进一步分析 Tool 本身是否为性能瓶颈。

---

# 7. Tool Execution Profiling

## 7.1 RAG Tool

对 RAG 内部执行时间进一步拆分：

```text
Query Embedding ≈ 0.0065 s
Vector Search   ≈ 0.0001 s
Total           ≈ 0.0066 s
```

也就是说：

```text
RAG Retrieval ≈ 6~8 ms
```

---

## 7.2 Calculator Tool

Calculator 自身执行时间：

```text
≈ 0.0001 s
```

因此 Calculator 本身几乎没有计算开销。

---

## 7.3 Bottleneck Analysis

上述结果说明：

```text
Tool Execution Time
```

并不是 Agent 端到端性能瓶颈。

主要耗时更可能来自：

```text
Local LLM Inference
+
Agent Decision Loop
+
Tool Calling Round Trips
```

尤其是重复 Tool Calling，会导致额外的：

```text
LLM Inference
Tool Call
Tool Output
LLM Decision
```

从而显著增加响应时间。

---

# 8. Latency Result After Runtime Optimization

Runtime Termination 后的一组 Benchmark 结果：

| Task | Mean Latency |
|---|---:|
| No Tool | 3.68 s |
| RAG Tool | 6.97 s |
| Calculator | 4.53 s |

其中 Calculator：

```text
22.12 s
↓
4.53 s
```

降幅约：

```text
79.5%
```

Calculator 延迟降低与 Runtime Termination 直接相关，因为该优化缩短了 Calculator Tool Calling 执行链路。

需要注意：

> Runtime Termination 仅直接作用于 Calculator 路径，因此 No Tool 和 RAG Benchmark 的变化不能全部归因于该优化。

本地 Qwen3 推理时间还会受到以下因素影响：

```text
Model Warm-up
Generated Token Length
System Load
Context Length
Ollama Runtime State
```

因此 No Tool 和 RAG 的单次延迟存在一定自然波动。

---

# 9. RAG Retrieval Architecture

项目构建本地 Resume RAG，用于处理用户个人经历、技能和项目相关问题。

Embedding Model：

```text
BAAI/bge-small-zh-v1.5
```

Retrieval Pipeline：

```text
Resume Text
↓
Semantic Chunk Split
↓
Document Embedding
↓
Embedding Cache
↓
Query Embedding
↓
Cosine Similarity
↓
Top-K Retrieval
↓
LLM Answer Generation
```

Document Embedding 在应用初始化阶段一次性构建并缓存：

```text
Embedding Matrix
```

Query 到达后仅计算 Query Embedding。

因此检索执行时间能够维持在毫秒级。

---

# 10. RAG Corpus Optimization

早期 Resume Corpus 只有 4 个较粗粒度 Chunk：

```text
Personal Information
Agent Project
Multi-view Clustering
YOLO Project
```

存在以下问题：

```text
Chunk Semantic Overlap
Corpus Information Missing
Project Description Outdated
Ground Truth Ambiguity
```

例如：

- FastAPI 已经实际使用，但旧 Corpus 中缺失；
- PostgreSQL Session 已经完成，但旧 Corpus 中没有完整描述；
- YOLO 项目已经从早期描述发展为实际甲骨文字图像分类任务；
- Agent Engineering 和 Agent Core Logic 被混在一个 Chunk 中。

因此重新整理 Corpus。

最终采用 6 个语义 Chunk：

```text
Chunk 0
个人信息与求职方向

Chunk 1
Agent / 智能体 / Tool Calling / Evaluation

Chunk 2
Agent系统前后端 / RAG / FastAPI / Vue / PostgreSQL

Chunk 3
不完整多视图甲骨文聚类研究

Chunk 4
甲骨文字图像分类 / YOLO视觉识别

Chunk 5
技术栈与综合能力
```

通过语义职责划分降低 Chunk 间的信息重叠。

---

# 11. Manual Ground Truth Retrieval Evaluation

## 11.1 Problem with Keyword Ground Truth

初始 Retrieval Evaluation 采用：

```python
expected_keywords
```

自动推断 Relevant Chunk。

这种方式存在明显问题。

例如：

```text
Query:
我主要掌握哪些AI技术？
```

某个 Project Chunk 可能包含 `"AI"` 或 `"Transformer"`，但真正应该优先返回的是：

```text
技术栈与综合能力
```

因此 Keyword Matching 并不能可靠定义 Retrieval Ground Truth。

---

## 11.2 Manual Ground Truth

最终改为人工定义：

```python
{
    "query": "...",
    "relevant_chunk_ids": [...]
}
```

例如：

```python
{
    "query": "我的YOLO项目主要做了什么？",
    "relevant_chunk_ids": [4],
}
```

以及多 Relevant Query：

```python
{
    "query": "我有哪些计算机视觉项目经历？",
    "relevant_chunk_ids": [3, 4],
}
```

最终建立：

```text
20 Query Retrieval Benchmark
```

覆盖：

```text
Agent
RAG
Backend
YOLO
Computer Vision
Multi-view Clustering
Transformer
Diffusion
Technical Skills
LLM Application Development
```

---

# 12. Retrieval Metrics

Retrieval Evaluation 使用：

## Hit@1

Top-1 是否包含 Relevant Chunk：

```text
Hit@1
```

用于衡量首个返回结果的准确性。

---

## Hit@2

Top-2 中是否至少包含一个 Relevant Chunk：

```text
Hit@2
```

当前实际 RAG 使用：

```text
top_k = 2
```

因此该指标直接反映实际系统 Evidence Recall。

---

## MRR

Mean Reciprocal Rank：

```text
MRR = Mean(1 / First Relevant Rank)
```

用于衡量 Relevant Chunk 是否能够尽可能靠前。

---

## Top1-Top2 Margin

```text
Top1 Score - Top2 Score
```

用于观察排名第一和第二候选之间的置信度差异。

---

## Relevant Margin

定义：

```text
Best Relevant Score
-
Best Non-Relevant Score
```

若：

```text
Relevant Margin > 0
```

说明 Relevant Evidence 排名领先。

若：

```text
Relevant Margin < 0
```

说明至少有一个 Non-Relevant Chunk 排在 Relevant Chunk 前面。

---

# 13. Dense Retrieval Baseline

在最终 6-Chunk Corpus 上进行 Dense Retrieval Evaluation。

结果：

```text
Total Cases:              20
Valid Cases:              20
Invalid Cases:             0

Hit@1:                  80.00%
Hit@2:                 100.00%
MRR:                     0.9000

Mean Relevant Margin:     0.0673

Failed Top1 Cases:        4
Weak Retrieval Cases:     4
```

Dense Retriever 已经能够做到：

```text
Hit@2 = 100%
```

说明所有测试 Query 的 Relevant Evidence 均能够进入 Top2。

主要问题集中在：

```text
Top1 Ranking
```

而不是 Recall。

---

# 14. Dense Retrieval Failure Analysis

典型失败包括：

```text
Query:
我有没有做过FastAPI相关开发？
```

Dense Ranking：

```text
Rank 1: Agent Chunk
Rank 2: FastAPI / Backend Chunk
```

两者分数差仅约：

```text
0.0038
```

另一个例子：

```text
Query:
我的Agent系统前后端是怎么实现的？
```

Dense Retrieval 中正确 Chunk 同样已经到 Rank 2。

因此失败模式主要表现为：

```text
Relevant Evidence 已召回
但语义接近 Chunk 排序发生轻微颠倒
```

这说明下一步重点应从：

```text
Recall Optimization
```

转向：

```text
Ranking Optimization
```

---

# 15. Hybrid Retrieval

## 15.1 Motivation

Dense Embedding 擅长整体语义匹配，但对于部分技术专有词：

```text
RAG
FastAPI
YOLO
PostgreSQL
Transformer
Diffusion
```

纯 Dense Similarity 可能无法稳定利用精确字符串匹配信息。

因此引入轻量 Lexical Matching。

---

## 15.2 Hybrid Scoring

最终 Ranking Score：

```text
Final Score
=
0.90 × Dense Similarity
+
0.10 × Lexical Score
```

其中：

```text
Dense Similarity
```

来自 BGE Embedding Cosine Similarity。

Lexical Score 根据 Query 中出现的技术关键词与 Document 的关键词覆盖情况计算。

该方法保持架构简单，不额外引入大型 Cross Encoder 或 Reranker。

---

# 16. Hybrid Retrieval Result

Hybrid Retrieval 结果：

| Metric | Dense | Hybrid |
|---|---:|---:|
| Hit@1 | 80.00% | **95.00%** |
| Hit@2 | 100.00% | **100.00%** |
| MRR | 0.9000 | **0.9750** |
| Mean Relevant Margin | 0.0673 | **0.0938** |
| Failed Top1 Cases | 4 | **1** |

最终：

```text
Hit@1 = 95.00%

Hit@2 = 100.00%

MRR = 0.9750
```

20 条人工标注 Query 中：

```text
20 / 20
```

均能够在 Top2 中召回 Relevant Evidence。

Hybrid Retrieval 基本消除了：

```text
RAG
FastAPI
Agent Backend
```

等 Query 中的 Dense Ranking 冲突。

---

# 17. Remaining Retrieval Failure

最终仅有一条 Top1 Failure：

```text
Query:
我有哪些计算机视觉项目经历？
```

Ground Truth：

```text
Chunk 3
Chunk 4
```

Hybrid Ranking：

```text
Rank 1: Chunk 5
Rank 2: Chunk 3
Rank 3: Chunk 0
Rank 4: Chunk 4
```

Chunk 5 为：

```text
技术栈与综合能力
```

其中明确包含：

```text
计算机视觉
```

因此 Lexical Score 将该 Chunk 推至 Rank 1。

但是 Relevant Project Evidence 已在：

```text
Rank 2
```

进入实际系统使用的：

```text
Top-K = 2
```

范围。

继续针对该单一 Case 调整权重容易产生 Evaluation Overfitting，因此当前阶段停止进一步针对测试集调整 Hybrid Weight。

---

# 18. Retrieval Latency

Hybrid Retrieval 没有显著增加系统延迟。

典型结果：

```text
Embedding ≈ 5~7 ms
Search    ≈ 0.1 ms
Total     ≈ 5~7 ms
```

因此：

```text
Dense + Lexical Hybrid Retrieval
```

在提高 Ranking Quality 的同时，仍保持毫秒级检索开销。

---

# 19. Multi-turn Session Growth Evaluation

## 19.1 Objective

Agent 使用 SQLAlchemySession 保存完整多轮上下文，包括：

```text
user
reasoning
assistant
function_call
function_call_output
```

随着 Conversation 轮数增加，Session 数据将持续增长。

因此需要回答：

> Session 数据增长是否会在当前规模下造成明显延迟退化？

---

## 19.2 Evaluation Setup

使用同一个 Conversation ID 连续运行 12 轮。

任务按照：

```text
No Tool
RAG
Calculator
```

循环执行。

每轮记录：

```text
Round
Task Type
Agent Latency
Tool Calls
agent_messages Count
Messages Added
Payload Characters
Payload Bytes
Row Storage
```

---

# 20. Session Growth Result

12 轮后：

```text
Rounds:
12
```

Agent Messages：

```text
3
↓
52
```

Payload：

```text
2.56 KB
↓
41.25 KB
```

Row Storage：

```text
2.79 KB
↓
45.23 KB
```

平均每轮增加：

```text
4.33 Agent Messages
```

平均每轮 Payload 增长：

```text
3.44 KB
```

因此当前 Session 数据呈近似线性增长。

---

# 21. Message Growth by Task Type

实验中观察到稳定的 Message Growth Pattern。

## No Tool

平均每轮：

```text
+3 items
```

典型结构：

```text
user
reasoning
assistant
```

---

## RAG

平均每轮：

```text
+6 items
```

典型结构：

```text
user
reasoning
function_call
function_call_output
reasoning
assistant
```

---

## Calculator

平均每轮：

```text
+4 items
```

典型结构：

```text
user
reasoning
function_call
function_call_output
```

Calculator 配置 Runtime Termination，因此 Tool Output 返回后结束当前 Agent Run。

这意味着 Runtime Termination 不仅减少了：

```text
Latency
```

同时也减少了额外：

```text
Reasoning
Tool Call
Tool Output
Assistant Generation
```

产生的 Session 数据。

---

# 22. Session Storage Composition

对测试 Session：

```text
a91bbc67-3832-4cb2-9a11-1c9aed37f8e9
```

中的 PostgreSQL `agent_messages` 进行统计。

结果：

| Item Type | Count | Storage |
|---|---:|---:|
| reasoning | 16 | 25.47 KB |
| function_call_output | 8 | 6.68 KB |
| message | 8 | 6.61 KB |
| function_call | 8 | 1.77 KB |
| user message | 12 | 0.72 KB |
| **Total** | **52** | **≈41.25 KB** |

其中：

```text
reasoning
=
25.47 KB
```

占总 Payload：

```text
≈ 61.7%
```

因此 Reasoning Item 是当前 Session Storage Growth 的主要来源。

---

# 23. User Message Identification

数据库中存在：

```text
type = NULL
```

的 Session Item。

进一步检查发现这些记录结构为：

```json
{
    "content": "...",
    "role": "user"
}
```

共：

```text
12
```

条。

数量与 12 轮测试完全一致。

因此这些记录并非异常数据，而是：

```text
User Message
```

因为它们没有 `type` 字段，所以在基于：

```sql
message_data::jsonb ->> 'type'
```

的统计中被归类为：

```text
unknown
```

最终统计时应将其解释为：

```text
user message
```

---

# 24. Session Size vs Latency

12 轮 Session Growth 测试中：

```text
Payload
2.56 KB → 41.25 KB
```

与此同时：

```text
First 3 Rounds Mean Latency:
8.35 s

Last 3 Rounds Mean Latency:
7.34 s
```

变化：

```text
-12.07%
```

因此没有观察到：

```text
Session Size ↑
→
Latency 持续 ↑
```

的明显趋势。

---

# 25. Latency by Task Type

12 轮测试中的平均延迟：

| Task | Mean | Min | Max |
|---|---:|---:|---:|
| Calculator | 4.06 s | 3.62 s | 5.10 s |
| No Tool | 7.45 s | 5.90 s | 8.95 s |
| RAG | 11.77 s | 7.86 s | 15.98 s |

RAG 延迟波动相对明显，但并未随着 Session Size 增长呈单调上升。

例如 RAG：

```text
Round 2   12.35 s
Round 5    7.86 s
Round 8   15.98 s
Round 11  10.89 s
```

因此当前数据更支持：

```text
Local LLM inference variance
```

而不是：

```text
Session growth causes monotonic latency degradation
```

这一解释。

---

# 26. Context Compaction Decision

当前 Session 从：

```text
2.56 KB
```

增长至：

```text
41.25 KB
```

虽然存储规模近似线性增加，但没有观察到明显的端到端性能恶化。

因此当前阶段暂不引入：

```text
Context Trimming

Session Compaction

Conversation Summary

Aggressive History Removal
```

避免过早增加系统复杂度。

如果未来出现：

```text
Long Conversation
Large Context Window
Hundreds of Session Items
Latency Increase
Context Overflow
```

可优先针对：

```text
Reasoning History
```

和：

```text
Historical Context Window
```

进行优化，因为 Reasoning 当前已经占 Session Payload：

```text
≈ 61.7%
```

---

# 27. Evaluation Summary

当前 Agent Evaluation & Performance Optimization 的主要结果如下：

| Evaluation | Result |
|---|---|
| Tool Selection Test Cases | 12 |
| Tool Selection Accuracy | 100% |
| Calculator Duplicate Calls | 75% → 0% |
| Calculator Mean Latency | 22.12 s → 4.53 s |
| Calculator Latency Reduction | ≈79.5% |
| Retrieval Test Cases | 20 |
| Dense Hit@1 | 80% |
| Hybrid Hit@1 | 95% |
| Hybrid Hit@2 | 100% |
| Hybrid MRR | 0.975 |
| Mean Relevant Margin | 0.0938 |
| Hybrid Retrieval Latency | ≈5–7 ms |
| Session Test Rounds | 12 |
| Session Messages | 3 → 52 |
| Session Payload | 2.56 KB → 41.25 KB |
| Average Payload Growth | 3.44 KB / Round |
| Reasoning Payload Ratio | ≈61.7% |
| Observed Session-induced Latency Degradation | Not obvious |

---

# 28. Key Optimization Results

## Tool Calling

```text
Calculator Duplicate Tool Call Rate

75%
↓
0%
```

---

## Calculator Latency

```text
22.12 s
↓
4.53 s

≈ 79.5% reduction
```

---

## Retrieval

```text
Dense Hit@1
80%

↓

Hybrid Hit@1
95%
```

同时：

```text
Hit@2 = 100%

MRR = 0.975
```

---

## Session Growth

```text
12 Rounds

3 Messages
↓
52 Messages

2.56 KB
↓
41.25 KB
```

Reasoning：

```text
25.47 KB

≈ 61.7%
```

---

# 29. Engineering Conclusions

通过本阶段 Evaluation，可以得到以下工程结论。

### 1. Tool Calling 不能只依赖 Prompt 控制

Prompt 可以描述期望行为，但无法保证模型严格遵循 Tool Calling 次数约束。

对于具有明确终止条件的 Tool，更可靠的方法是：

```text
Runtime Control
```

而不是不断增加 Prompt 规则。

---

### 2. Tool 本身不是主要延迟瓶颈

当前：

```text
Calculator ≈ 0.1 ms

RAG Retrieval ≈ 5~7 ms
```

而 Agent 总响应时间为秒级。

因此主要性能成本位于：

```text
Local LLM Inference
Agent Decision Loop
Repeated Tool Calling
```

而不是 Tool Function 本身。

---

### 3. Retrieval Evaluation 必须使用人工 Ground Truth

仅使用 Keyword Matching 自动构造 Ground Truth 容易导致错误评价。

最终采用：

```text
Manual relevant_chunk_ids
```

能够更准确地区分：

```text
真正相关 Evidence
```

和：

```text
仅包含相同关键词的 Chunk
```

---

### 4. Dense Retrieval 负责 Recall，Lexical Signal 改善 Ranking

Dense Retrieval 已达到：

```text
Hit@2 = 100%
```

说明 Recall 已经足够。

剩余问题主要是：

```text
Top1 Ranking
```

加入轻量 Lexical Signal 后：

```text
Hit@1
80% → 95%
```

且没有显著增加 Retrieval Latency。

---

### 5. 当前没有必要过早进行 Session Compaction

虽然 Session Payload 在线性增长，但 12 轮、约 41 KB 范围内：

```text
未观察到明显的 latency degradation
```

因此现阶段继续增加 Session Compression 机制的工程收益有限。

---

# 30. Current Evaluation Status

```text
Agent Evaluation & Performance Optimization
│
├── Latency Profiling
│   └── Completed
│
├── Tool Selection Evaluation
│   └── Completed
│
├── Duplicate Tool Call Diagnosis
│   └── Completed
│
├── Prompt Constraint Experiment
│   └── Completed
│
├── Runtime Termination Optimization
│   └── Completed
│
├── RAG Retrieval Evaluation
│   └── Completed
│
├── Manual Ground Truth
│   └── Completed
│
├── Hybrid Retrieval Optimization
│   └── Completed
│
└── Session Growth Evaluation
    └── Completed
```

当前 Evaluation 阶段正式结束。

---

# 31. Next Stage

在保持当前 Evaluation Baseline 不变的前提下，下一阶段将重点从通用 Career Agent 转向实际业务场景：

```text
Oracle Bone Agent
```

计划逐步接入：

```text
Oracle Image Classification
↓
YOLO Image Classification Model
↓
Similar Oracle Retrieval
↓
Oracle Knowledge Retrieval
↓
Agent Tool Orchestration
↓
FastAPI Service
↓
Vue Frontend
↓
PostgreSQL Persistence
```

对应 Tool 可以逐步扩展为：

```text
classify_oracle_image

retrieve_similar_oracles

search_oracle_knowledge
```

最终形成：

```text
用户文本 / 图像输入
        ↓
      Agent
        ↓
Intent / Tool Routing
        ↓
┌─────────────────────────────┐
│ Oracle Image Classification │
│ Similar Oracle Retrieval    │
│ Oracle Knowledge Retrieval  │
└─────────────────────────────┘
        ↓
Evidence Fusion
        ↓
Natural Language Response
        ↓
Execution Trace / PostgreSQL
```

后续所有甲骨文业务优化应建立在当前 Evaluation Baseline 之上，并通过相同 Evaluation 思路验证业务 Tool 的：

```text
Tool Selection
Accuracy
Latency
Retrieval Quality
Session Behavior
```

从而保证系统从 Demo 逐步演进为可评估、可观测、可复现的 AI Agent Application。
