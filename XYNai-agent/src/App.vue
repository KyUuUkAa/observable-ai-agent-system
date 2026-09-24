<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

// ========================================
// API 配置
// ========================================

const API_BASE_URL = 'http://127.0.0.1:8000'

// ========================================
// 类型定义
// ========================================

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface ToolLog {
  type: string
  tool_name?: string
  arguments?: string
  output?: string
}

interface Conversation {
  id: string
  title: string
  created_at: string
  updated_at: string
}

interface AgentResponse {
  run_id: string
  timestamp: string
  status: string
  input: string
  output: string | null
  latency: number
  tool_call_count: number
  tool_logs: ToolLog[]
  error?: string
}

interface StoredMessage {
  id: number
  conversation_id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

interface OraclePrediction {
  class_id: number
  class_code: string
  confidence: number
}

interface OracleRecognitionResponse {
  filename: string
  image: {
    width: number
    height: number
  }
  prediction: OraclePrediction
  top5: OraclePrediction[]
  model: {
    task: string
    class_count: number
    image_size: number
    device: string
  }
}

// ========================================
// 页面状态
// ========================================

const input = ref<string>('')

const loading = ref<boolean>(false)

const conversationCreating = ref<boolean>(false)

const historyLoading = ref<boolean>(false)

const conversationId = ref<string | null>(null)

const lastRun = ref<AgentResponse | null>(null)

const conversations = ref<Conversation[]>([])

const messages = ref<Message[]>([
  {
    role: 'assistant',
    content: '请选择左侧历史会话，或点击「+ 新建对话」开始新的会话。',
  },
])

const activeMode = ref<'chat' | 'oracle'>('chat')

const oracleFile = ref<File | null>(null)

const oraclePreviewUrl = ref<string | null>(null)

const oracleLoading = ref<boolean>(false)

const oracleError = ref<string>('')

const oracleResult = ref<OracleRecognitionResponse | null>(null)

// ========================================
// 是否允许发送
// ========================================

const canSend = computed(() => {
  const text = input.value ?? ''

  return (
    text.trim().length > 0 &&
    !loading.value &&
    !historyLoading.value &&
    !conversationCreating.value &&
    conversationId.value !== null
  )
})

const canRecognizeOracle = computed(() => {
  return oracleFile.value !== null && !oracleLoading.value
})

// ========================================
// 加载历史 Conversation
// ========================================

async function loadConversations() {
  console.log('开始加载历史会话')

  try {
    const response = await fetch(`${API_BASE_URL}/conversations`)

    console.log('GET /conversations status:', response.status)

    if (!response.ok) {
      throw new Error(`加载历史会话失败：HTTP ${response.status}`)
    }

    const data: Conversation[] = await response.json()

    conversations.value = data

    console.log('历史会话列表:', conversations.value)
  } catch (error) {
    console.error('加载历史会话失败:', error)
  }
}

// ========================================
// 加载某个 Conversation 的历史消息
// ========================================

async function loadMessages(targetConversationId: string) {
  console.log('开始加载历史消息:', targetConversationId)

  historyLoading.value = true

  try {
    const response = await fetch(`${API_BASE_URL}/conversations/${targetConversationId}/messages`)

    console.log('GET messages status:', response.status)

    if (!response.ok) {
      throw new Error(`加载历史消息失败：HTTP ${response.status}`)
    }

    const data: StoredMessage[] = await response.json()

    console.log('历史消息:', data)

    messages.value = data.map((message) => ({
      role: message.role,
      content: message.content,
    }))

    // 新会话没有任何消息
    if (messages.value.length === 0) {
      messages.value = [
        {
          role: 'assistant',
          content: '这是一个新的会话，你可以开始提问。',
        },
      ]
    }
  } catch (error) {
    console.error('加载历史消息失败:', error)
  } finally {
    historyLoading.value = false
  }
}

// ========================================
// 切换历史 Conversation
// ========================================

async function selectConversation(conversation: Conversation) {
  if (loading.value) {
    console.warn('Agent 正在执行，暂时不能切换会话')
    return
  }

  if (historyLoading.value) {
    return
  }

  console.log('切换 Conversation:', conversation.id)

  // 更新当前 Conversation
  conversationId.value = conversation.id

  // 清理上一个 Conversation 的 Trace
  lastRun.value = null

  // 清空输入框
  input.value = ''

  // 从 PostgreSQL 恢复消息
  await loadMessages(conversation.id)
}

// ========================================
// 创建 Conversation
// ========================================

async function createConversation() {
  if (conversationCreating.value || loading.value) {
    return
  }

  console.log('开始创建 Conversation')

  conversationCreating.value = true

  try {
    const response = await fetch(`${API_BASE_URL}/conversations`, {
      method: 'POST',
    })

    console.log('POST /conversations status:', response.status)

    if (!response.ok) {
      throw new Error(`创建会话失败：HTTP ${response.status}`)
    }

    const data: Conversation = await response.json()

    // 切换到刚创建的 Conversation
    conversationId.value = data.id

    // 清理旧 Trace
    lastRun.value = null

    // 清理输入框
    input.value = ''

    // 新会话欢迎消息
    messages.value = [
      {
        role: 'assistant',
        content: '你好，我是 Career Agent。你可以询问我的项目经历、技术栈和相关能力。',
      },
    ]

    console.log('Conversation created:', data.id)

    // 更新左侧列表
    await loadConversations()
  } catch (error) {
    console.error('创建 Conversation 失败:', error)
  } finally {
    conversationCreating.value = false
  }
}

// ========================================
// 删除 Conversation
// ========================================

async function deleteConversation(conversation: Conversation) {
  if (loading.value) {
    console.warn('Agent 正在执行，暂时不能删除会话')
    return
  }

  if (historyLoading.value) {
    return
  }

  const confirmed = window.confirm(
    `确定删除「${conversation.title}」吗？\n\n该会话的聊天记录也会一起删除。`,
  )

  if (!confirmed) {
    return
  }

  try {
    const response = await fetch(`${API_BASE_URL}/conversations/${conversation.id}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      throw new Error(`删除失败：HTTP ${response.status}`)
    }

    console.log('Conversation deleted:', conversation.id)

    // 删除的是当前正在查看的 Conversation
    if (conversationId.value === conversation.id) {
      conversationId.value = null

      lastRun.value = null

      input.value = ''

      messages.value = [
        {
          role: 'assistant',
          content: '请选择左侧历史会话，或点击「+ 新建对话」开始新的会话。',
        },
      ]
    }

    // 更新左侧列表
    await loadConversations()
  } catch (error) {
    console.error('删除 Conversation 失败:', error)

    window.alert('删除会话失败，请检查后端服务。')
  }
}

// ========================================
// 发送消息
// ========================================

async function sendMessage() {
  const text = (input.value ?? '').trim()

  if (!text) {
    return
  }

  if (loading.value || historyLoading.value || conversationCreating.value) {
    return
  }

  if (!conversationId.value) {
    console.error('Conversation 尚未创建')
    return
  }

  // ----------------------------------------
  // 判断是否需要刷新自动标题
  // ----------------------------------------

  const currentConversation = conversations.value.find(
    (conversation) => conversation.id === conversationId.value,
  )

  const shouldRefreshTitle = currentConversation?.title === 'New Conversation'

  console.log('当前 Conversation ID:', conversationId.value)

  // ----------------------------------------
  // 页面先显示 User 消息
  // ----------------------------------------

  messages.value.push({
    role: 'user',
    content: text,
  })

  input.value = ''

  loading.value = true

  try {
    // ----------------------------------------
    // 调用 Agent
    // ----------------------------------------

    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',

      headers: {
        'Content-Type': 'application/json',
      },

      body: JSON.stringify({
        conversation_id: conversationId.value,

        message: text,
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const data: AgentResponse = await response.json()

    console.log('Agent response:', data)

    // ----------------------------------------
    // Execution Trace
    // ----------------------------------------

    lastRun.value = data

    // ----------------------------------------
    // Agent 回复
    // ----------------------------------------

    if (data.status === 'success') {
      messages.value.push({
        role: 'assistant',

        content: data.output ?? 'Agent 没有返回内容。',
      })

      // ----------------------------------------
      // 首轮消息后刷新 Conversation 标题
      // ----------------------------------------

      if (shouldRefreshTitle) {
        console.log('刷新 Conversation 标题')

        await loadConversations()
      }
    } else {
      messages.value.push({
        role: 'assistant',

        content: `Agent 执行失败：${data.error ?? '未知错误'}`,
      })
    }
  } catch (error) {
    console.error('发送消息失败:', error)

    messages.value.push({
      role: 'assistant',

      content: `请求失败：${error instanceof Error ? error.message : String(error)}`,
    })
  } finally {
    loading.value = false
  }
}

// ========================================
// 甲骨文单字识别
// ========================================

function clearOraclePreview() {
  if (oraclePreviewUrl.value) {
    URL.revokeObjectURL(oraclePreviewUrl.value)
    oraclePreviewUrl.value = null
  }
}

function selectOracleFile(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0] ?? null

  oracleError.value = ''
  oracleResult.value = null
  clearOraclePreview()

  if (!file) {
    oracleFile.value = null
    return
  }

  if (!file.type.startsWith('image/')) {
    oracleFile.value = null
    oracleError.value = '请选择图片文件。'
    target.value = ''
    return
  }

  if (file.size > 10 * 1024 * 1024) {
    oracleFile.value = null
    oracleError.value = '图片不能超过 10 MB。'
    target.value = ''
    return
  }

  oracleFile.value = file
  oraclePreviewUrl.value = URL.createObjectURL(file)
}

async function recognizeOracle() {
  if (!oracleFile.value || oracleLoading.value) {
    return
  }

  oracleLoading.value = true
  oracleError.value = ''
  oracleResult.value = null

  const formData = new FormData()
  formData.append('file', oracleFile.value)

  try {
    const response = await fetch(`${API_BASE_URL}/oracle/recognize`, {
      method: 'POST',
      body: formData,
    })

    const data = await response.json()

    if (!response.ok) {
      throw new Error(
        typeof data.detail === 'string' ? data.detail : `识别失败：HTTP ${response.status}`,
      )
    }

    oracleResult.value = data as OracleRecognitionResponse
  } catch (error) {
    oracleError.value = error instanceof Error ? error.message : String(error)
  } finally {
    oracleLoading.value = false
  }
}

// ========================================
// 页面加载
// ========================================

onMounted(async () => {
  // 页面刷新时只读取历史记录
  // 不自动创建 Conversation
  await loadConversations()
})

onBeforeUnmount(() => {
  clearOraclePreview()
})
</script>

<template>
  <div class="page">
    <div class="workspace">
      <!-- ================================= -->
      <!-- 左侧：历史会话 -->
      <!-- ================================= -->

      <aside class="conversation-sidebar">
        <div class="sidebar-header">
          <div class="sidebar-title-row">
            <h2>历史会话</h2>

            <span class="conversation-count">
              {{ conversations.length }}
            </span>
          </div>

          <button
            class="new-conversation-button"
            :disabled="conversationCreating || loading"
            @click="createConversation"
          >
            {{ conversationCreating ? '创建中...' : '+ 新建对话' }}
          </button>
        </div>

        <!-- Conversation List -->

        <div class="conversation-list">
          <div
            v-for="conversation in conversations"
            :key="conversation.id"
            class="conversation-item"
            :class="{
              active: conversation.id === conversationId,
            }"
            @click="selectConversation(conversation)"
          >
            <div class="conversation-info">
              <div class="conversation-title">
                {{ conversation.title || 'New Conversation' }}
              </div>

              <div class="conversation-id">
                {{ conversation.id.slice(0, 8) }}
              </div>
            </div>

            <!-- 删除按钮 -->

            <button
              class="delete-conversation-button"
              title="删除会话"
              @click.stop="deleteConversation(conversation)"
            >
              ×
            </button>
          </div>

          <div v-if="conversations.length === 0" class="conversation-empty">暂无历史会话</div>
        </div>
      </aside>

      <!-- ================================= -->
      <!-- 中间：聊天区域 -->
      <!-- ================================= -->

      <section class="chat-container">
        <header class="header">
          <div>
            <h1>
              {{ activeMode === 'chat' ? 'Career Agent' : '甲骨文单字识别' }}
            </h1>

            <p>
              {{
                activeMode === 'chat'
                  ? 'Agent · Tool Calling · RAG'
                  : 'YOLO11 Classification · Top-5'
              }}
            </p>
          </div>

          <div class="header-right">
            <div class="mode-switch">
              <button
                :class="{
                  active: activeMode === 'chat',
                }"
                @click="activeMode = 'chat'"
              >
                Agent 对话
              </button>

              <button
                :class="{
                  active: activeMode === 'oracle',
                }"
                @click="activeMode = 'oracle'"
              >
                甲骨文识别
              </button>
            </div>

            <div class="status">
              <span class="status-dot"></span>

              Online
            </div>

            <div class="conversation-status">
              <span v-if="activeMode === 'oracle'"> Single Glyph Mode </span>

              <span v-else-if="conversationCreating"> Creating Session... </span>

              <span v-else-if="conversationId"> Session Ready </span>

              <span v-else> No Session </span>
            </div>
          </div>
        </header>

        <!-- Messages -->

        <main v-if="activeMode === 'chat'" class="messages">
          <div v-if="historyLoading" class="history-loading">正在加载历史消息...</div>

          <div
            v-for="(message, index) in messages"
            :key="index"
            class="message-row"
            :class="message.role"
          >
            <div class="message">
              {{ message.content }}
            </div>
          </div>

          <div v-if="loading" class="message-row assistant">
            <div class="message loading">Agent 正在处理...</div>
          </div>
        </main>

        <main v-else class="oracle-workspace">
          <div class="oracle-intro">
            <span class="oracle-kicker"> Oracle Bone Script </span>

            <h2>上传一张已裁剪的单字图片</h2>

            <p>当前模型支持 39 个类别。请上传只包含一个甲骨文字的 BMP、PNG、JPG 或 WEBP 图片。</p>
          </div>

          <label
            class="oracle-dropzone"
            :class="{
              'has-preview': oraclePreviewUrl,
            }"
          >
            <input
              type="file"
              accept="image/*"
              :disabled="oracleLoading"
              @change="selectOracleFile"
            />

            <img v-if="oraclePreviewUrl" :src="oraclePreviewUrl" alt="待识别甲骨文字预览" />

            <div v-else class="oracle-dropzone-copy">
              <strong>选择单字图片</strong>
              <span>最大 10 MB</span>
            </div>
          </label>

          <div v-if="oracleFile" class="oracle-file-meta">
            <span>{{ oracleFile.name }}</span>
            <span> {{ (oracleFile.size / 1024).toFixed(1) }} KB </span>
          </div>

          <div v-if="oracleError" class="oracle-error">
            {{ oracleError }}
          </div>

          <button class="oracle-submit" :disabled="!canRecognizeOracle" @click="recognizeOracle">
            {{ oracleLoading ? '识别中...' : '开始识别' }}
          </button>

          <p class="oracle-note">模型输出为数据集类别编码；现代汉字和释义映射将在后续阶段补充。</p>
        </main>

        <!-- Input -->

        <footer v-if="activeMode === 'chat'" class="input-area">
          <input
            v-model="input"
            type="text"
            placeholder="输入问题，例如：我有什么大模型应用开发经历？"
            :disabled="loading || historyLoading || conversationCreating || !conversationId"
            @keyup.enter="sendMessage"
          />

          <button :disabled="!canSend" @click="sendMessage">
            {{ loading ? '处理中' : '发送' }}
          </button>
        </footer>
      </section>

      <!-- ================================= -->
      <!-- 右侧：Execution Trace -->
      <!-- ================================= -->

      <aside class="trace-panel">
        <template v-if="activeMode === 'oracle'">
          <h2>Recognition Result</h2>

          <div v-if="oracleLoading" class="oracle-result-empty">
            正在加载模型并识别图片，首次请求可能需要几秒钟。
          </div>

          <div v-else-if="!oracleResult" class="oracle-result-empty">
            上传单字图片后，这里会显示预测类别、置信度和 Top‑5 候选。
          </div>

          <template v-else>
            <div class="oracle-primary-result">
              <span>预测类别</span>
              <strong>
                {{ oracleResult.prediction.class_code }}
              </strong>
              <small> {{ (oracleResult.prediction.confidence * 100).toFixed(2) }}% </small>
            </div>

            <div class="oracle-top5-title">Top‑5 Candidates</div>

            <div
              v-for="candidate in oracleResult.top5"
              :key="candidate.class_id"
              class="oracle-candidate"
            >
              <div class="oracle-candidate-row">
                <code>{{ candidate.class_code }}</code>
                <span> {{ (candidate.confidence * 100).toFixed(2) }}% </span>
              </div>

              <div class="oracle-confidence-track">
                <div
                  class="oracle-confidence-fill"
                  :style="{
                    width: `${candidate.confidence * 100}%`,
                  }"
                ></div>
              </div>
            </div>

            <div class="oracle-model-meta">
              <div>
                <span>Input</span>
                <strong>
                  {{ oracleResult.image.width }} ×
                  {{ oracleResult.image.height }}
                </strong>
              </div>
              <div>
                <span>Classes</span>
                <strong>
                  {{ oracleResult.model.class_count }}
                </strong>
              </div>
              <div>
                <span>Device</span>
                <strong>
                  {{ oracleResult.model.device }}
                </strong>
              </div>
            </div>
          </template>
        </template>

        <template v-else>
          <h2>Execution Trace</h2>

          <!-- Conversation ID -->

          <div class="trace-item">
            <span> Conversation ID </span>

            <code v-if="conversationId">
              {{ conversationId }}
            </code>

            <code v-else> Not Ready </code>
          </div>

          <!-- 暂无 Trace -->

          <div v-if="!lastRun" class="empty-trace">发送一条消息后，这里会显示 Agent 执行信息。</div>

          <!-- Agent Trace -->

          <template v-else>
            <div class="trace-item">
              <span> Run ID </span>

              <code>
                {{ lastRun.run_id }}
              </code>
            </div>

            <div class="trace-item">
              <span> Status </span>

              <strong>
                {{ lastRun.status }}
              </strong>
            </div>

            <div class="trace-item">
              <span> Latency </span>

              <strong>
                {{ typeof lastRun.latency === 'number' ? lastRun.latency.toFixed(2) : '-' }}
                s
              </strong>
            </div>

            <div class="trace-item">
              <span> Tool Calls </span>

              <strong>
                {{ lastRun.tool_call_count ?? 0 }}
              </strong>
            </div>

            <!-- Tool Logs -->

            <div v-for="(log, index) in lastRun.tool_logs ?? []" :key="index" class="tool-log">
              <div class="tool-type">
                {{ log.type }}
              </div>

              <div v-if="log.tool_name">
                <b>Tool:</b>

                {{ log.tool_name }}
              </div>

              <div v-if="log.arguments">
                <b>Arguments:</b>

                <pre>{{ log.arguments }}</pre>
              </div>

              <div v-if="log.output">
                <b>Output:</b>

                <pre>{{ log.output }}</pre>
              </div>
            </div>
          </template>
        </template>
      </aside>
    </div>
  </div>
</template>

<style scoped>
/* ========================================
   Page
======================================== */

.page {
  width: 100%;
  min-height: 100vh;

  padding: 30px;

  box-sizing: border-box;

  background: #f5f6f8;

  display: flex;
  justify-content: center;
  align-items: center;
}

.workspace {
  width: 1480px;
  max-width: 100%;
  height: 720px;

  display: flex;

  gap: 20px;
}

/* ========================================
   Conversation Sidebar
======================================== */

.conversation-sidebar {
  width: 220px;

  flex-shrink: 0;

  background: white;

  border: 1px solid #e5e7eb;

  border-radius: 16px;

  display: flex;
  flex-direction: column;

  overflow: hidden;

  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.05);
}

/* Sidebar Header */

.sidebar-header {
  padding: 16px;

  border-bottom: 1px solid #e5e7eb;

  box-sizing: border-box;
}

.sidebar-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;

  margin-bottom: 12px;
}

.sidebar-header h2 {
  margin: 0;

  font-size: 16px;

  color: #111827;
}

.conversation-count {
  min-width: 24px;
  height: 24px;

  padding: 0 7px;

  display: flex;
  align-items: center;
  justify-content: center;

  background: #f3f4f6;

  border-radius: 12px;

  font-size: 12px;

  color: #6b7280;

  box-sizing: border-box;
}

/* New Conversation */

.new-conversation-button {
  width: 100%;
  height: 38px;

  border: 1px solid #d1d5db;

  border-radius: 9px;

  background: #111827;

  color: white;

  font-size: 13px;

  cursor: pointer;

  transition: background 0.15s ease;
}

.new-conversation-button:hover {
  background: #1f2937;
}

.new-conversation-button:disabled {
  opacity: 0.5;

  cursor: not-allowed;
}

/* Conversation List */

.conversation-list {
  flex: 1;

  padding: 12px;

  overflow-y: auto;
}

/* Conversation Card */

.conversation-item {
  padding: 12px;

  margin-bottom: 8px;

  border-radius: 10px;

  cursor: pointer;

  display: flex;
  flex-direction: row;

  align-items: center;
  justify-content: space-between;

  gap: 8px;

  transition: background 0.15s ease;
}

.conversation-item:hover {
  background: #f3f4f6;
}

.conversation-item.active {
  background: #111827;
}

/* Conversation Text */

.conversation-info {
  flex: 1;

  min-width: 0;
}

.conversation-title {
  font-size: 13px;
  font-weight: 500;

  color: #111827;

  white-space: nowrap;

  overflow: hidden;

  text-overflow: ellipsis;
}

.conversation-item.active .conversation-title {
  color: white;
}

.conversation-id {
  margin-top: 5px;

  font-size: 11px;

  color: #9ca3af;
}

.conversation-item.active .conversation-id {
  color: #d1d5db;
}

/* Delete Button */

.delete-conversation-button {
  width: 28px;
  height: 28px;

  flex-shrink: 0;

  display: flex;
  align-items: center;
  justify-content: center;

  border: none;

  border-radius: 7px;

  background: transparent;

  color: #9ca3af;

  font-size: 20px;

  line-height: 1;

  cursor: pointer;

  opacity: 1;

  transition:
    background 0.15s ease,
    color 0.15s ease;
}

.delete-conversation-button:hover {
  background: #fee2e2;

  color: #dc2626;
}

.conversation-item.active .delete-conversation-button {
  color: #d1d5db;
}

.conversation-item.active .delete-conversation-button:hover {
  background: rgba(255, 255, 255, 0.15);

  color: white;
}

.conversation-empty {
  padding: 20px 8px;

  text-align: center;

  color: #9ca3af;

  font-size: 13px;
}

/* ========================================
   Chat
======================================== */

.chat-container {
  flex: 1;

  min-width: 0;

  background: white;

  border: 1px solid #e5e7eb;

  border-radius: 16px;

  display: flex;
  flex-direction: column;

  overflow: hidden;

  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.06);
}

/* Header */

.header {
  min-height: 90px;

  padding: 20px 28px;

  border-bottom: 1px solid #e5e7eb;

  display: flex;
  justify-content: space-between;
  align-items: center;

  box-sizing: border-box;
}

.header h1 {
  margin: 0;

  font-size: 22px;

  color: #111827;
}

.header p {
  margin: 5px 0 0;

  color: #777;

  font-size: 13px;
}

.header-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;

  gap: 5px;
}

.mode-switch {
  display: flex;

  padding: 3px;

  background: #f3f4f6;

  border-radius: 9px;
}

.mode-switch button {
  padding: 6px 10px;

  border: none;

  border-radius: 7px;

  background: transparent;

  color: #6b7280;

  font-size: 11px;

  cursor: pointer;
}

.mode-switch button.active {
  background: white;

  color: #111827;

  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
}

.status {
  display: flex;
  align-items: center;

  gap: 8px;

  font-size: 13px;

  color: #111827;
}

.status-dot {
  width: 9px;
  height: 9px;

  background: #22c55e;

  border-radius: 50%;
}

.conversation-status {
  font-size: 11px;

  color: #9ca3af;
}

/* ========================================
   Messages
======================================== */

.messages {
  flex: 1;

  padding: 28px;

  overflow-y: auto;

  box-sizing: border-box;
}

.history-loading {
  padding: 20px;

  text-align: center;

  color: #9ca3af;

  font-size: 13px;
}

.message-row {
  display: flex;

  margin-bottom: 18px;
}

.message-row.user {
  justify-content: flex-end;
}

.message-row.assistant {
  justify-content: flex-start;
}

.message {
  max-width: 70%;

  padding: 13px 17px;

  border-radius: 14px;

  line-height: 1.7;

  white-space: pre-wrap;

  word-break: break-word;

  font-size: 14px;
}

.user .message {
  background: #111827;

  color: white;
}

.assistant .message {
  background: #f1f3f5;

  color: #222;
}

.loading {
  color: #777;
}

/* ========================================
   Oracle Recognition
======================================== */

.oracle-workspace {
  flex: 1;

  padding: 36px;

  overflow-y: auto;

  box-sizing: border-box;
}

.oracle-intro {
  max-width: 600px;

  margin-bottom: 24px;
}

.oracle-kicker {
  color: #92400e;

  font-size: 11px;
  font-weight: 700;

  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.oracle-intro h2 {
  margin: 8px 0;

  color: #111827;

  font-size: 22px;
}

.oracle-intro p,
.oracle-note {
  margin: 0;

  color: #6b7280;

  font-size: 13px;
  line-height: 1.7;
}

.oracle-dropzone {
  width: 100%;
  min-height: 250px;

  padding: 20px;

  border: 1px dashed #d1d5db;

  border-radius: 14px;

  background: #fafaf9;

  display: flex;
  align-items: center;
  justify-content: center;

  cursor: pointer;

  box-sizing: border-box;
}

.oracle-dropzone:hover {
  border-color: #a16207;

  background: #fffbeb;
}

.oracle-dropzone input {
  display: none;
}

.oracle-dropzone img {
  max-width: 100%;
  max-height: 290px;

  object-fit: contain;
}

.oracle-dropzone-copy {
  display: flex;
  flex-direction: column;
  align-items: center;

  gap: 7px;

  color: #6b7280;
}

.oracle-dropzone-copy strong {
  color: #111827;

  font-size: 15px;
}

.oracle-dropzone-copy span {
  font-size: 12px;
}

.oracle-file-meta {
  margin-top: 12px;

  display: flex;
  justify-content: space-between;

  color: #6b7280;

  font-size: 12px;
}

.oracle-error {
  margin-top: 14px;

  padding: 10px 12px;

  border-radius: 8px;

  background: #fef2f2;

  color: #b91c1c;

  font-size: 13px;
}

.oracle-submit {
  width: 100%;
  height: 46px;

  margin-top: 18px;

  border: none;

  border-radius: 10px;

  background: #78350f;

  color: white;

  font-size: 14px;
  font-weight: 600;

  cursor: pointer;
}

.oracle-submit:disabled {
  opacity: 0.45;

  cursor: not-allowed;
}

.oracle-note {
  margin-top: 12px;

  text-align: center;
}

.oracle-result-empty {
  color: #888;

  font-size: 14px;
  line-height: 1.7;
}

.oracle-primary-result {
  margin-bottom: 24px;

  padding: 18px;

  border: 1px solid #fde68a;

  border-radius: 12px;

  background: #fffbeb;

  display: flex;
  flex-direction: column;

  gap: 4px;
}

.oracle-primary-result span,
.oracle-primary-result small {
  color: #92400e;

  font-size: 12px;
}

.oracle-primary-result strong {
  color: #451a03;

  font-family: Consolas, monospace;

  font-size: 28px;
}

.oracle-top5-title {
  margin-bottom: 12px;

  color: #374151;

  font-size: 12px;
  font-weight: 700;

  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.oracle-candidate {
  margin-bottom: 14px;
}

.oracle-candidate-row {
  margin-bottom: 6px;

  display: flex;
  justify-content: space-between;

  font-size: 12px;
}

.oracle-candidate-row code {
  color: #111827;

  font-weight: 600;
}

.oracle-candidate-row span {
  color: #6b7280;
}

.oracle-confidence-track {
  height: 6px;

  overflow: hidden;

  border-radius: 3px;

  background: #f3f4f6;
}

.oracle-confidence-fill {
  height: 100%;

  border-radius: 3px;

  background: #b45309;
}

.oracle-model-meta {
  margin-top: 24px;
  padding-top: 18px;

  border-top: 1px solid #e5e7eb;

  display: grid;
  grid-template-columns: repeat(3, 1fr);

  gap: 8px;
}

.oracle-model-meta div {
  display: flex;
  flex-direction: column;

  gap: 4px;
}

.oracle-model-meta span {
  color: #9ca3af;

  font-size: 10px;
}

.oracle-model-meta strong {
  color: #374151;

  font-size: 11px;
}

/* ========================================
   Input
======================================== */

.input-area {
  padding: 20px 24px;

  border-top: 1px solid #e5e7eb;

  display: flex;

  gap: 12px;

  box-sizing: border-box;
}

.input-area input {
  flex: 1;

  min-width: 0;

  height: 46px;

  border: 1px solid #ddd;

  border-radius: 10px;

  padding: 0 14px;

  font-size: 15px;

  outline: none;

  box-sizing: border-box;
}

.input-area input:focus {
  border-color: #888;
}

.input-area input:disabled {
  background: #f9fafb;

  cursor: not-allowed;
}

.input-area button {
  width: 90px;

  border: none;

  border-radius: 10px;

  background: #111827;

  color: white;

  cursor: pointer;
}

.input-area button:disabled {
  opacity: 0.5;

  cursor: not-allowed;
}

/* ========================================
   Execution Trace
======================================== */

.trace-panel {
  width: 330px;

  flex-shrink: 0;

  padding: 22px;

  background: white;

  border: 1px solid #e5e7eb;

  border-radius: 16px;

  overflow-y: auto;

  box-sizing: border-box;

  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.05);
}

.trace-panel h2 {
  margin: 0 0 22px;

  font-size: 18px;

  color: #111827;
}

.empty-trace {
  padding-top: 8px;

  color: #888;

  font-size: 14px;

  line-height: 1.6;
}

.trace-item {
  margin-bottom: 18px;
}

.trace-item span {
  display: block;

  margin-bottom: 5px;

  color: #777;

  font-size: 12px;
}

.trace-item code {
  display: block;

  word-break: break-all;

  font-size: 12px;

  color: #374151;
}

.trace-item strong {
  font-size: 14px;

  color: #111827;
}

/* Tool Trace */

.tool-log {
  margin-top: 16px;

  padding: 14px;

  background: #f8fafc;

  border-radius: 10px;

  font-size: 13px;

  line-height: 1.5;

  overflow: hidden;
}

.tool-type {
  margin-bottom: 10px;

  font-weight: 600;
}

.tool-log pre {
  margin: 7px 0 0;

  white-space: pre-wrap;

  word-break: break-word;

  font-size: 12px;

  font-family: Consolas, monospace;
}

/* ========================================
   Responsive
======================================== */

@media (max-width: 1000px) {
  .workspace {
    height: auto;

    flex-direction: column;
  }

  .conversation-sidebar {
    width: 100%;

    max-height: 300px;
  }

  .chat-container {
    min-height: 650px;
  }

  .trace-panel {
    width: 100%;
  }
}
</style>
