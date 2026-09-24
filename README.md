# Observable AI Agent System

> A full-stack observable AI Agent application built with OpenAI Agents SDK, Ollama, Qwen3, RAG, FastAPI, Vue, and PostgreSQL.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688" />
  <img src="https://img.shields.io/badge/Vue-Frontend-42b883" />
  <img src="https://img.shields.io/badge/PostgreSQL-Database-336791" />
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-black" />
  <img src="https://img.shields.io/badge/Qwen3-4B-purple" />
  <img src="https://img.shields.io/badge/OpenAI-Agents%20SDK-412991" />
</p>

---

## Overview

**Observable AI Agent System** is a full-stack AI Agent application focused on building an Agent that is not only functional, but also:

- observable
- measurable
- debuggable
- persistent
- evaluatable
- optimizable

The system uses **OpenAI Agents SDK** as the orchestration layer and runs **Qwen3:4B locally through Ollama**.

A local RAG pipeline based on **BAAI/bge-small-zh-v1.5** provides semantic retrieval over private knowledge, while **PostgreSQL** stores both application-level conversation data and Agent runtime session context.

The project also includes a dedicated evaluation framework covering:

- Tool Selection Evaluation
- End-to-End Latency Evaluation
- RAG Retrieval Evaluation
- Session Growth Evaluation
- Failure Case Analysis
- Runtime Optimization

The project follows the engineering principle:

```text
Build
  ↓
Observe
  ↓
Evaluate
  ↓
Find Failure Cases
  ↓
Optimize
  ↓
Re-evaluate
```

---

# Demo

## RAG Tool Calling & Execution Trace

The Agent automatically determines whether a user request requires private project knowledge.

When retrieval is required, the Agent calls the `search_resume` tool and retrieves relevant context from the local RAG knowledge base.

The right-side **Execution Trace** panel exposes runtime information including:

- Conversation ID
- Run ID
- Execution status
- End-to-end latency
- Tool call count
- Tool name
- Tool arguments
- Tool output

![RAG Tool Calling and Execution Trace](docs/images/agent-rag-trace.png)

---

## Deterministic Calculator Tool

For deterministic arithmetic tasks, the Agent invokes the `calculator` tool instead of relying on LLM-generated arithmetic.

A runtime termination strategy is applied after calculator execution to prevent unnecessary repeated tool calls.

![Calculator Tool Execution Trace](docs/images/calculator-tool-trace.png)

---

# Core Capabilities

## 1. Agent Orchestration

The application uses **OpenAI Agents SDK** to manage:

- Agent execution
- Function Tool Calling
- tool routing
- multi-turn conversations
- session persistence
- runtime control

The local LLM is exposed through an OpenAI-compatible interface:

```text
OpenAI Agents SDK
        ↓
OpenAIChatCompletionsModel
        ↓
Ollama
        ↓
Qwen3:4B
```

This allows the Agent orchestration layer to remain independent from a hosted LLM provider.

---

## 2. Local LLM Inference

The application uses:

```text
Ollama
  ↓
Qwen3:4B
```

for local inference.

Advantages include:

- no remote LLM dependency during runtime
- lower deployment cost
- local data processing
- easier Agent behavior debugging
- reproducible evaluation

---

## 3. Function Tool Calling

The current Agent contains two primary tools.

### `search_resume`

Retrieves relevant private knowledge through the RAG pipeline.

Typical use cases:

```text
我的 Agent 项目主要做了什么？
我的 YOLO 项目效果怎么样？
我有没有使用 FastAPI？
我的多视图聚类项目用了哪些技术？
```

### `calculator`

Handles deterministic arithmetic operations.

Supported operations include:

```text
add
subtract
multiply
divide
```

Example:

```text
User
  ↓
计算 1234 * 5678
  ↓
Agent
  ↓
calculator(
    a=1234,
    b=5678,
    operation="multiply"
)
  ↓
Tool Output
  ↓
计算结果：7006652
```

---

# RAG Pipeline

The project contains a fully local Retrieval-Augmented Generation pipeline.

The retrieval flow is:

```text
Private Documents
        ↓
Chunking
        ↓
BAAI/bge-small-zh-v1.5
        ↓
Normalized Embeddings
        ↓
Embedding Cache
        ↓

User Query
        ↓
Query Embedding
        ↓
Dense Similarity ────────────┐
                              │
Lexical Matching ─────────────┤
                              ↓
                       Hybrid Ranking
                              ↓
                           Top-K
                              ↓
                        Agent Context
```

The final retrieval strategy combines:

```text
Dense Retrieval
+
Lexical Matching
=
Hybrid Retrieval
```

Current weighting:

```text
Dense Weight   = 0.90
Lexical Weight = 0.10
```

The dense component captures semantic similarity, while the lexical component improves ranking for domain-specific keywords such as:

```text
Agent
RAG
FastAPI
PostgreSQL
YOLO
Transformer
Diffusion
```

---

# System Architecture

```text
┌──────────────────────────────────────────────┐
│                  Vue Frontend                │
│                                              │
│  Chat UI                                     │
│  Conversation History                        │
│  Execution Trace                             │
│  Session Management                          │
└──────────────────────┬───────────────────────┘
                       │
                       │ REST API
                       ▼
┌──────────────────────────────────────────────┐
│                 FastAPI Backend              │
│                                              │
│  /chat                                       │
│  /conversations                              │
│  /health                                     │
│                                              │
│  Conversation Service                        │
│  Agent Harness                               │
└───────────────┬───────────────────┬──────────┘
                │                   │
                │                   │
                ▼                   ▼
┌───────────────────────┐  ┌──────────────────────┐
│   OpenAI Agents SDK   │  │      PostgreSQL      │
│                       │  │                      │
│   Agent               │  │ conversations        │
│   Tool Calling        │  │ messages             │
│   Session Runtime     │  │ agent_sessions       │
│                       │  │ agent_messages       │
└───────────┬───────────┘  └──────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│                 Local LLM                    │
│                                              │
│             Ollama + Qwen3:4B                │
└───────────┬──────────────────────────────────┘
            │
            │ Tool Calling
            ▼
┌──────────────────────────────────────────────┐
│                    Tools                     │
│                                              │
│  search_resume                               │
│        │                                     │
│        └── BGE Embedding                     │
│            + Dense Retrieval                 │
│            + Lexical Retrieval               │
│            + Hybrid Ranking                  │
│                                              │
│  calculator                                  │
│        │                                     │
│        └── Deterministic Calculation         │
└──────────────────────────────────────────────┘
```

---

# Observability

A custom Agent Harness records execution metadata for every run.

Each execution can contain:

```text
Run ID
Timestamp
Success / Failed
Latency
Tool Name
Arguments
Tool Output
```

Runtime logs can be persisted to:

```text
logs.jsonl
```

The frontend exposes this information through the **Execution Trace** panel.

This makes it possible to inspect the full Agent execution chain:

```text
User Request
   ↓
Agent Reasoning
   ↓
Tool Selection
   ↓
Tool Call
   ↓
Tool Output
   ↓
Final Response
```

This observability layer was also used to identify repeated tool calls during evaluation.

---

# PostgreSQL Session Persistence

The application persists two different kinds of data.

## Application Conversation Data

```text
conversations
messages
```

These tables are used by the application layer to manage:

- conversation list
- chat history
- conversation titles
- message restoration
- conversation deletion

## Agent Runtime Context

```text
agent_sessions
agent_messages
```

These tables store OpenAI Agents SDK session context.

The architecture therefore separates:

```text
Application-level conversation state
                +
Agent runtime context
```

rather than mixing both into a single table.

---

# Evaluation

A dedicated evaluation suite was implemented instead of relying only on manual testing.

Current evaluation modules include:

```text
evaluation/
├── latency_eval.py
├── retrieval_eval.py
├── session_growth_eval.py
└── tool_selection_eval.py
```

Detailed methodology and results are documented in:

[`EVALUATION.md`](EVALUATION.md)

---

# Evaluation Results

## 1. Tool Selection Evaluation

A manually constructed benchmark was used to evaluate whether the Agent correctly selects among:

```text
search_resume
calculator
no tool
```

The benchmark contains:

```text
12 evaluation cases
```

covering:

- private project questions
- deterministic arithmetic
- general knowledge questions

### Failure Discovered

Initial testing showed that the local LLM sometimes repeatedly called the Calculator tool for a single arithmetic request.

Example:

```text
User Request
    ↓
Calculator
    ↓
Calculator
    ↓
Calculator
```

Prompt-level constraints were first introduced, but they did not eliminate the behavior.

A runtime-level termination strategy was then implemented for deterministic calculator calls.

### Result

| Metric | Before Optimization | After Optimization |
|---|---:|---:|
| Calculator duplicate tool-call rate | 75% | **0%** |
| Calculator mean latency | 22.12 s | **4.53 s** |
| Tool Selection Accuracy | - | **100%** |
| Evaluation Cases | - | 12 |

Calculator latency was reduced by approximately:

```text
79.5%
```

on the current benchmark.

The optimization process was:

```text
Evaluation
    ↓
Detect Duplicate Tool Calling
    ↓
Prompt Constraint
    ↓
No Significant Improvement
    ↓
Runtime Termination Strategy
    ↓
Duplicate Rate: 75% → 0%
```

---

# 2. RAG Retrieval Evaluation

A manually annotated retrieval benchmark containing **20 queries** was created.

The evaluation does not rely on keyword matching as ground truth.

Each query is manually mapped to one or more relevant document chunks.

Metrics include:

- Hit@1
- Hit@2
- Mean Reciprocal Rank
- Top1-Top2 Margin
- Relevant Margin
- Failed Retrieval Cases
- Weak Retrieval Cases

### Final Results

| Metric | Result |
|---|---:|
| Evaluation Queries | 20 |
| Valid Queries | 20 |
| Hit@1 | **95.00%** |
| Hit@2 | **100.00%** |
| MRR | **0.9750** |
| Mean Top1-Top2 Margin | 0.0865 |
| Mean Relevant Margin | 0.0938 |
| Failed Top1 Cases | 1 |

Typical query embedding and retrieval latency after model loading:

```text
~5-10 ms
```

The evaluation showed that all 20 benchmark queries retrieve a relevant chunk within Top-2.

One remaining Top-1 failure occurs for a broad query involving multiple computer-vision project experiences, where the technical-stack chunk ranks above the project-specific chunks.

Because the production retrieval path uses Top-K context, this case still retrieves relevant information.

---

# 3. Session Growth Evaluation

The PostgreSQL-backed Agent Session was evaluated over **12 consecutive interaction rounds**.

The benchmark contained repeated task categories:

```text
No-Tool
RAG
Calculator
```

### Session Growth

| Metric | Result |
|---|---:|
| Evaluation Rounds | 12 |
| Agent Messages | 3 → 52 |
| Payload Size | 2.56 KB → 41.25 KB |
| Average Messages / Round | 4.33 |
| Average Payload Growth / Round | 3.44 KB |

Observed message growth by task type:

```text
No-Tool
≈ +3 messages

RAG
≈ +6 messages

Calculator
≈ +4 messages
```

This corresponds to their different execution chains.

### Latency

| Metric | Result |
|---|---:|
| First 3 Rounds Mean Latency | 8.35 s |
| Last 3 Rounds Mean Latency | 7.34 s |
| Change | -12.07% |

Mean latency by task type:

| Task Type | Mean Latency |
|---|---:|
| Calculator | 4.06 s |
| No-Tool | 7.45 s |
| RAG | 11.77 s |

Session storage grew approximately linearly with conversation length.

However, within the tested range of:

```text
12 rounds
≈ 41 KB payload
```

no clear positive correlation between session size and inference latency was observed.

Therefore, context compaction was intentionally **not introduced prematurely**.

---

# Performance Analysis

The system separates tool execution latency from full Agent latency.

Observed local tool execution times are approximately:

```text
RAG Retrieval
≈ 5-10 ms

Calculator
≈ <1 ms
```

However, end-to-end Agent latency is significantly higher because the dominant cost comes from:

```text
Local Qwen3 inference
+
Agent reasoning
+
Tool-call generation
+
Post-tool response generation
```

This distinction avoids incorrectly attributing Agent latency to the retrieval subsystem.

---

# Frontend Features

The Vue frontend currently supports:

- multi-turn chat
- conversation history
- new conversation creation
- automatic conversation title generation
- message history restoration
- conversation deletion
- current session display
- execution trace visualization
- tool-call inspection
- latency display
- Agent online status

The UI follows a three-column structure:

```text
┌────────────────┬───────────────────────────┬─────────────────┐
│ Conversation   │         Chat              │ Execution Trace │
│ History        │                           │                 │
│                │ User / Assistant Messages │ Run ID          │
│ New Chat       │                           │ Status          │
│ Delete         │ Input                     │ Latency         │
│                │                           │ Tool Calls      │
└────────────────┴───────────────────────────┴─────────────────┘
```

---

# Technology Stack

## Agent / LLM

- OpenAI Agents SDK
- Ollama
- Qwen3:4B
- Function Calling
- Agent Session
- Agent Evaluation

## RAG

- BAAI/bge-small-zh-v1.5
- Sentence Transformers
- Dense Retrieval
- Cosine Similarity
- Lexical Matching
- Hybrid Retrieval
- Embedding Cache
- Top-K Search

## Backend

- Python
- FastAPI
- Uvicorn
- REST API

## Database

- PostgreSQL
- asyncpg
- SQLAlchemySession
- psycopg

## Frontend

- Vue
- TypeScript
- Vite
- CSS

## Evaluation

- Tool Selection Evaluation
- Latency Benchmark
- Retrieval Evaluation
- Session Growth Evaluation
- Failure Case Analysis

---

# Project Structure

```text
observable-ai-agent-system/
│
├── agent.py
│   └── Agent definition and local Qwen model configuration
│
├── api.py
│   └── FastAPI application and REST endpoints
│
├── database.py
│   └── PostgreSQL business conversation persistence
│
├── harness.py
│   └── Agent execution harness, session management and observability
│
├── rag.py
│   └── Local embedding, retrieval and hybrid ranking
│
├── tools.py
│   └── Agent Function Tools
│
├── main.py
│   └── Local application entry point
│
├── requirements.txt
│
├── .env.example
├── .gitignore
│
├── data/
│   ├── .gitkeep
│   └── resume.example.txt
│
├── docs/
│   └── images/
│       ├── agent-rag-trace.png
│       └── calculator-tool-trace.png
│
├── evaluation/
│   ├── __init__.py
│   ├── latency_eval.py
│   ├── retrieval_eval.py
│   ├── session_growth_eval.py
│   └── tool_selection_eval.py
│
├── tests/
│   ├── test_rag.py
│   └── test_session.py
│
├── XYNai-agent/
│   └── Vue frontend
│
├── EVALUATION.md
└── README.md
```

---

# Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/KyUuUkAa/observable-ai-agent-system.git
cd observable-ai-agent-system
```

---

## 2. Create Python Environment

Using Conda:

```bash
conda create -n agent python=3.10
conda activate agent
```

Or use an existing Python environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Install Ollama

Install Ollama from the official Ollama distribution.

Pull the local model:

```bash
ollama pull qwen3:4b
```

Verify that the model exists:

```bash
ollama list
```

Make sure the Ollama service is running before starting the backend.

---

## 4. Prepare PostgreSQL

Create a PostgreSQL database:

```sql
CREATE DATABASE career_agent;
```

The default project database name is:

```text
career_agent
```

Make sure PostgreSQL is running locally.

---

## 5. Configure Environment Variables

Copy:

```text
.env.example
```

to:

```text
.env
```

Example:

```env
AGENT_DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/career_agent

# Optional
HF_TOKEN=
```

Replace:

```text
your_password
```

with your local PostgreSQL password.

Never commit the real `.env` file.

The repository `.gitignore` already excludes it.

---

# Local RAG Data

Private resume/project data is intentionally excluded from Git.

The real local document:

```text
data/resume.txt
```

is ignored.

A public example file is included:

```text
data/resume.example.txt
```

To run your own private RAG knowledge base, create:

```text
data/resume.txt
```

and place your own content inside it.

Example structure:

```text
【Project A】
Project description...

【Project B】
Project description...

【Technical Skills】
Python, FastAPI, PostgreSQL...
```

Separate sections with blank lines so they can be chunked independently.

---

# Start Backend

From the project root:

```bash
uvicorn api:app --reload
```

Default backend address:

```text
http://127.0.0.1:8000
```

Health check:

```text
GET /health
```

---

# Start Frontend

Open another terminal:

```bash
cd XYNai-agent
npm install
npm run dev
```

Vite will normally start the frontend on:

```text
http://localhost:5173
```

or another available local Vite port.

---

# Main API Endpoints

## Health Check

```http
GET /health
```

---

## Send Chat Message

```http
POST /chat
```

The backend:

```text
receives user message
        ↓
loads Agent session
        ↓
runs Agent
        ↓
executes tools if required
        ↓
records execution trace
        ↓
stores messages
        ↓
returns response
```

---

## Create Conversation

```http
POST /conversations
```

---

## List Conversations

```http
GET /conversations
```

---

## Retrieve Conversation Messages

```http
GET /conversations/{conversation_id}/messages
```

---

## Delete Conversation

```http
DELETE /conversations/{conversation_id}
```

Deletion clears both:

```text
business conversation data
+
Agent session context
```

for the selected conversation.

---

# Run Evaluation

## Tool Selection Evaluation

```bash
python -m evaluation.tool_selection_eval
```

Evaluates:

```text
search_resume
calculator
no-tool
```

routing behavior.

---

## Latency Evaluation

```bash
python -m evaluation.latency_eval
```

Measures end-to-end Agent latency across task categories.

---

## Retrieval Evaluation

```bash
python -m evaluation.retrieval_eval
```

Outputs metrics including:

```text
Hit@1
Hit@2
MRR
Relevant Margin
Weak Retrieval Cases
Failed Retrieval Cases
```

---

## Session Growth Evaluation

```bash
python -m evaluation.session_growth_eval
```

Measures:

```text
Agent message count
Session payload size
PostgreSQL storage
Round-by-round latency
Task-specific latency
```

---

# Evaluation Methodology

The evaluation suite intentionally separates several system layers.

## Tool Layer

Measures:

```text
tool selection
duplicate calls
wrong tool calls
missed tool calls
```

## Retrieval Layer

Measures:

```text
retrieval ranking quality
Top-K coverage
ranking confidence
failure cases
```

## Agent Layer

Measures:

```text
end-to-end latency
tool execution chain
runtime behavior
```

## Session Layer

Measures:

```text
context growth
payload growth
message growth
latency trend
```

This separation makes it easier to determine whether a failure comes from:

```text
LLM reasoning
Tool Selection
RAG Retrieval
Database Session
or Application Logic
```

---

# Engineering Decisions

## Why Local Qwen3 Instead of a Hosted LLM?

The project is designed to support:

- local inference
- reproducible evaluation
- lower runtime cost
- easier debugging
- local private-data processing

---

## Why PostgreSQL Instead of Only In-Memory Sessions?

Persistent sessions allow:

- application restart recovery
- long-lived conversation history
- multi-turn Agent context
- session growth analysis
- production-like state management

---

## Why Separate Business Messages and Agent Messages?

The frontend does not need every internal Agent runtime item.

Therefore the project separates:

```text
Frontend / Product Conversation
```

from:

```text
Agent Runtime Context
```

This keeps the data model cleaner and makes session behavior observable independently.

---

## Why Hybrid Retrieval?

Dense retrieval performs well on semantic similarity, but some engineering terms are highly lexical.

For example:

```text
FastAPI
PostgreSQL
YOLO
RAG
Agent
```

Hybrid retrieval combines semantic similarity and explicit keyword evidence.

The final benchmark improved to:

```text
Hit@1 = 95%
Hit@2 = 100%
MRR   = 0.975
```

---

## Why Runtime Constraints for Calculator?

Prompt instructions are probabilistic.

For a deterministic tool such as a calculator, repeated execution provides no benefit.

The evaluation showed:

```text
Prompt Constraint
    ↓
Duplicate Calls Still Exist

Runtime Constraint
    ↓
Duplicate Rate = 0%
```

Therefore deterministic behavior is enforced at the runtime layer rather than relying only on LLM compliance.

---

# Current Limitations

The current project is still an engineering prototype.

Known limitations include:

- evaluation datasets are relatively small
- Qwen3:4B inference speed depends on local hardware
- RAG currently uses a small local document corpus
- no distributed Agent execution
- no production authentication system
- no containerized deployment yet
- no streaming response implementation
- no large-scale vector database

The evaluation metrics in this repository should therefore be interpreted as results for the current benchmark and environment rather than universal model performance claims.

---

# Future Work

Planned extensions include:

```text
Docker deployment
        ↓
Unified startup scripts
        ↓
Improved API documentation
        ↓
Streaming Agent responses
        ↓
Larger RAG knowledge base
        ↓
Context management / compaction if required
        ↓
Multimodal Agent tools
```

A future application direction is to extend the current Agent infrastructure into an Oracle Bone Script multimodal system with tools such as:

```text
classify_oracle_image
retrieve_similar_oracles
search_oracle_knowledge
```

The existing Agent orchestration, PostgreSQL persistence, observability and evaluation framework can be reused for these multimodal capabilities.

---

# Security

Sensitive files are intentionally excluded from Git.

Ignored content includes:

```text
.env
private RAG documents
runtime logs
local databases
generated evaluation reports
node_modules
Python cache files
```

Public examples are provided through:

```text
.env.example
data/resume.example.txt
```

Never commit:

```text
database passwords
API keys
access tokens
private resume data
```

---

# Reproducibility

The project provides:

- explicit environment configuration
- local model configuration
- reusable evaluation scripts
- manually defined retrieval ground truth
- evaluation reports
- deterministic tool logic
- PostgreSQL-backed session persistence

This allows system changes to be evaluated quantitatively instead of relying only on manual observation.

---

# What This Project Demonstrates

This project is intended to demonstrate an end-to-end AI application engineering workflow rather than only an LLM API demo.

It covers:

```text
LLM Integration
        ↓
Agent Orchestration
        ↓
Tool Calling
        ↓
RAG
        ↓
Backend API
        ↓
Database Persistence
        ↓
Frontend Integration
        ↓
Observability
        ↓
Evaluation
        ↓
Performance Optimization
```

The main engineering focus is:

> Build Agent systems whose behavior can be inspected, measured, evaluated and improved.

---

# Author

**香一宁**

Target roles:

- AI Application Engineer
- AI Full-Stack Engineer
- Algorithm Engineer

Main technical interests:

- AI Agent
- RAG
- Computer Vision
- Multimodal Learning
- Transformer
- Diffusion Models

---

# License

This repository is currently intended for portfolio, learning, research and demonstration purposes.

If a formal open-source license is added later, this section will be updated accordingly.