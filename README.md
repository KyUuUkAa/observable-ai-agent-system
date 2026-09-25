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

## API Documentation

The backend provides a REST API implemented with FastAPI.

Interactive Swagger documentation is automatically generated through OpenAPI.

Main API groups include:

- **System** — backend health monitoring
- **Agent** — Agent execution, Tool Calling and RAG
- **Conversations** — conversation and session lifecycle management
- **Oracle Recognition** — local single-glyph Oracle Bone Script classification

### Swagger UI

![Swagger API Documentation](docs/images/swagger-api-docs.png)

Main endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Backend health check |
| POST | `/chat` | Run Agent conversation |
| POST | `/conversations` | Create a conversation |
| GET | `/conversations` | List conversations |
| GET | `/conversations/{conversation_id}/messages` | Load conversation history |
| DELETE | `/conversations/{conversation_id}` | Delete conversation and clear Agent Session |
| GET | `/oracle/health` | Check whether the local classifier is configured |
| POST | `/oracle/recognize` | Recognize one glyph and persist its result |
| POST | `/oracle/batch` | Recognize and persist up to 50 cropped glyphs |
| GET | `/oracle/records` | List and filter recognition records |
| PATCH | `/oracle/records/{record_id}/review` | Confirm or reject a recognition result |
| GET | `/oracle/records/export` | Export filtered records as CSV or JSON |

When the backend is running locally, the interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

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

The current Agent contains three primary tools.

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

### `recognize_oracle_image`

Classifies one Oracle Bone Script glyph that was uploaded through the validated backend pipeline. The tool accepts only a short-lived opaque `image_id`; it cannot read arbitrary local paths or remote URLs.

The tool returns the predicted dataset class code, confidence and Top-5 candidates. Because the current delivery does not include a verified code-to-modern-character dictionary, the Agent is instructed not to invent a character or interpretation.

Frontend flow:

```text
Upload and recognize one glyph
        ↓
Receive short-lived image_id
        ↓
Click "交给 Agent 分析"
        ↓
Agent calls recognize_oracle_image
        ↓
Execution Trace records the tool call and output
```

## 4. Oracle Digitization Workflow

The Oracle module is designed as an assisted curation workflow rather than an automatic decipherment claim:

```text
Single or batch upload
        ↓
Top-1 / Top-5 classification
        ↓
PostgreSQL record + model fingerprint + trace
        ↓
Low-confidence result enters the review queue
        ↓
Human confirms, rejects, or adds notes
        ↓
Filtered CSV / JSON export
```

The default review threshold is `0.85` and can be changed with `ORACLE_REVIEW_THRESHOLD`. Original images, predictions, confidence, model version, timestamps, review decisions, and the later Agent execution trace are kept together for auditability.

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
├── oracle_recognition.py
│   └── Lazy-loaded Oracle Bone Script single-glyph classifier
│
├── oracle_workflow.py
│   └── Review threshold, trace construction and CSV/JSON export
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
├── models/oracle/
│   ├── .gitkeep
│   └── README.md
│
├── docs/
│   └── images/
│       ├── agent-rag-trace.png
│       ├── calculator-tool-trace.png
│       └── swagger-api-docs.png
│
├── evaluation/
│   ├── __init__.py
│   ├── agent_regression.py
│   ├── regression_core.py
│   ├── cases/tool_routing.json
│   ├── baselines/tool_routing.json
│   ├── latency_eval.py
│   ├── retrieval_eval.py
│   ├── session_growth_eval.py
│   └── tool_selection_eval.py
│
├── tests/
│   ├── test_agent_regression.py
│   ├── test_oracle_workflow.py
│   ├── test_rag.py
│   └── test_session.py
│
├── scripts/
│   ├── check_environment.ps1
│   ├── init_database.sql
│   ├── start_backend.ps1
│   ├── start_frontend.ps1
│   └── start_all.ps1
│
├── XYNai-agent/
│   └── Vue frontend
│
├── EVALUATION.md
└── README.md
```

---

# Quick Start / Run

The commands below target PowerShell on Windows. Required local services and tools:

- Python 3.10+; Conda is optional but recommended
- PostgreSQL server (the `psql` client is useful for initialization)
- Ollama with the `qwen3:4b` model
- Node.js and npm; use a version accepted by `XYNai-agent/package.json`

## 1. Clone the Repository

```powershell
git clone https://github.com/KyUuUkAa/observable-ai-agent-system.git
Set-Location observable-ai-agent-system
```

## 2. Create the Python Environment

Using Conda:

```powershell
conda create -n agent python=3.10 -y
conda activate agent
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

An existing Python environment can also be used. The launchers use the active environment by default and also accept `-CondaEnv <name>` or `-PythonPath <path>`.

## 3. Prepare PostgreSQL

Start PostgreSQL, then create the database and the required tables:

```powershell
psql -U postgres -c "CREATE DATABASE career_agent;"
psql -U postgres -d career_agent -f .\scripts\init_database.sql
```

If the database already exists, skip the first command but run `init_database.sql` again after pulling this version. You can run the SQL file through pgAdmin instead when `psql` is not on `PATH`. The script is idempotent and adds the recognition-record/review table without deleting existing data.

## 4. Configure `.env`

```powershell
Copy-Item .env.example .env
```

Edit `.env` locally and replace every password placeholder. Keep both database configurations consistent:

```env
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=career_agent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_postgres_password
AGENT_DATABASE_URL=postgresql+asyncpg://postgres:your_postgres_password@127.0.0.1:5432/career_agent
HF_TOKEN=
ORACLE_MODEL_PATH=models/oracle/best_portable.pt
ORACLE_DEVICE=cpu
ORACLE_IMGSZ=224
ORACLE_REVIEW_THRESHOLD=0.85
```

If the password contains reserved URL characters, URL-encode it in `AGENT_DATABASE_URL`. Never commit the real `.env`; it is ignored by Git.

## 5. Prepare Local RAG Data

`data/resume.txt` is private and ignored by Git. Create it from the safe example, then replace the example text with local content:

```powershell
Copy-Item .\data\resume.example.txt .\data\resume.txt
```

Separate document sections with blank lines so the RAG loader can chunk them independently.

## 6. Install Ollama and Qwen3

Install Ollama, ensure its local service is running, then pull and verify the model:

```powershell
ollama pull qwen3:4b
ollama list
```

The backend expects Ollama at `http://127.0.0.1:11434` and the exact model family `qwen3:4b`.

## 7. Add the Oracle Classifier Weight

The classifier weight is a local runtime asset and is intentionally excluded from Git. Copy the portable weight into the expected directory:

```powershell
Copy-Item "<path-to-oracle-delivery>\runs\preserve_shape\weights\best_portable.pt" `
  ".\models\oracle\best_portable.pt"
```

The default configuration uses CPU inference. To keep the weight elsewhere, set `ORACLE_MODEL_PATH` in `.env` to an absolute path or a path relative to the project root. Do not commit model weights unless you have explicitly chosen an appropriate model-distribution strategy.

The current model classifies one already-cropped glyph. Its output labels are dataset codes such as `001000`; a code-to-modern-character mapping is not included in the source delivery, so the UI displays class codes and confidence values.

## 8. Install Frontend Dependencies

```powershell
Set-Location .\XYNai-agent
npm install
Set-Location ..
```

## 9. Check the Environment

Run the complete preflight check from the project root:

```powershell
.\scripts\check_environment.ps1
```

It checks Python/Conda, Python packages, `.env`, private RAG data, the Oracle classifier weight, PostgreSQL connectivity and schema, Ollama and `qwen3:4b`, Node.js, npm, and frontend dependencies. It never prints secret values.

To select a Conda environment without activating it:

```powershell
.\scripts\check_environment.ps1 -CondaEnv agent
```

## 10. Start Backend and Frontend Separately

Backend terminal:

```powershell
.\scripts\start_backend.ps1
```

Frontend terminal:

```powershell
.\scripts\start_frontend.ps1
```

Equivalent manual commands are:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload

Set-Location .\XYNai-agent
npm run dev -- --host 127.0.0.1 --port 5173
```

Default URLs:

- Frontend: `http://127.0.0.1:5173`
- Backend: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`
- Oracle model health: `http://127.0.0.1:8000/oracle/health`
- Swagger: `http://127.0.0.1:8000/docs`

## 11. One-Command Start

After completing the setup above, launch both services in separate PowerShell windows:

```powershell
.\scripts\start_all.ps1
```

For a named Conda environment:

```powershell
.\scripts\start_all.ps1 -CondaEnv agent
```

If local policy blocks project scripts, allow them only for the current shell and retry:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Stop the services with `Ctrl+C` in their respective windows.

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

## Recognize an Oracle Bone Script Glyph

```http
POST /oracle/recognize
Content-Type: multipart/form-data
```

Upload one PNG, JPEG, BMP, or other Pillow-supported image in the `file` field. The backend validates that the upload is an image, rejects files larger than 10 MB, pads the glyph to a square without stretching it, and lazily loads the local model on the first recognition request.

The response includes the persisted record ID, review state, Top-1 class code, Top-5 predictions and confidence values. Example:

```json
{
  "filename": "glyph.png",
  "image_id": "6c77965e2f174638b99b6ec0ea8f5771",
  "record_id": "7eb52723-c043-4564-a1e2-d0a5d56e03da",
  "review_status": "auto_accepted",
  "review_threshold": 0.85,
  "image": { "width": 224, "height": 180 },
  "prediction": { "class_id": 12, "class_code": "038000", "confidence": 0.91 },
  "top5": [
    { "class_id": 12, "class_code": "038000", "confidence": 0.91 }
  ],
  "model": {
    "task": "single_glyph_classification",
    "class_count": 39,
    "image_size": 224,
    "device": "cpu"
  }
}
```

`image_id` is kept only in process memory for 15 minutes and is intended for the Agent Tool. To invoke the Agent with this attachment, include it in the normal chat request:

```json
{
  "conversation_id": "7a8d3c17-eb7b-4fc5-a66c-d653b91d6675",
  "message": "请分析这张甲骨文单字图片。",
  "oracle_image_id": "6c77965e2f174638b99b6ec0ea8f5771"
}
```

Model availability can be checked without loading the weight:

```http
GET /oracle/health
```

## Batch Recognition, Review, and Export

The frontend's **批量处理与复核** tab exposes the complete workflow. The equivalent APIs are:

```http
POST /oracle/batch
GET /oracle/review-summary
GET /oracle/records?review_status=pending
PATCH /oracle/records/{record_id}/review
GET /oracle/records/export?format=csv&review_status=pending
GET /oracle/records/export?format=json
```

`POST /oracle/batch` accepts repeated multipart fields named `files`, with at most 50 images per request. A review update accepts `status` (`pending`, `accepted`, or `rejected`) and optional `notes`.

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

## Unified Agent Regression Evaluation

Run the versioned JSON case suite, compare it with the committed baseline, and generate timestamped plus `latest` JSON/Markdown reports:

```bash
python -m evaluation.agent_regression --fail-on-regression --fail-on-case-failure
```

The pipeline reads `evaluation/cases/tool_routing.json`, runs every case through the real Agent, parses the execution trace, detects correct/wrong/missed/false/duplicate tool calls, aggregates latency and failure types, and compares the result with `evaluation/baselines/tool_routing.json`. Temporary database sessions are removed after each case unless `--keep-sessions` is supplied. Generated reports are written to the ignored `reports/agent_regression/` directory.

Useful options:

```bash
python -m evaluation.agent_regression --case-id calc_01
python -m evaluation.agent_regression --tag calculator
python -m evaluation.agent_regression --keep-sessions --case-id rag_01
python -m evaluation.agent_regression --write-baseline
```

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
- Oracle recognition accepts cropped glyphs rather than locating multiple glyphs in a full rubbing
- Oracle classifier labels are dataset codes until a verified code-to-character dictionary is added
- Agent image references are short-lived and process-local; persisted recognition records and original images remain available after a restart

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
Oracle character dictionary and multi-glyph detection
```

The standalone Oracle classifier is now available as the `recognize_oracle_image` Agent Tool. Future capabilities can include:

```text
retrieve_similar_oracles
search_oracle_knowledge
detect_multiple_oracle_glyphs
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
