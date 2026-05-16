
# 📚 AI 智能文档助手 (AI Document Agent)

> 💡 **一个带自定义 ReAct Agent 的全栈 RAG 文档分析平台。**  
> 上传文档 → Agent 自主分析 → 问答/摘要/翻译/对比/报告，全自动。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python" />
  <img src="https://img.shields.io/badge/LLM-DeepSeek/GPT/Ollama-green?style=flat" />
  <img src="https://img.shields.io/badge/Vector-ChromaDB-orange?style=flat" />
  <img src="https://img.shields.io/badge/Tests-210%20passed-brightgreen?style=flat" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat" />
</p>

---

## 🤖 AI Agent 架构

```
用户输入 "帮我总结然后翻译成日语"
       │
       ▼
┌─────────────────────────────────────┐
│      AgentSession.stream()          │
│                                     │
│  ┌─ 1. Build Messages ───────────┐  │
│  │  SystemPrompt + 历史 + 新消息  │ │
│  └────────────────────────────────┘ │
│              ▼                      │
│  ┌─ 2. _stream_llm() ───────────┐   │
│  │  LLM.stream() 逐 token 接收  │   │
│  │  首个字符 = ⚙️?              │   │
│  │  ├─ YES → 工具调用，静默收集   │  │
│  │  └─ NO  → 逐字流式输出到前端   │  │
│  └────────────────────────────────┘ │
│              ▼                      │
│  ┌─ 3. 检测工具标记 ────────────┐   │
│  │  ⚙️TOOL:{"name":...}⚙️END   │  │
│  │  正则解析 → 工具名 + 参数     │  │
│  └────────────────────────────────┘ │
│              ▼                      │
│  ┌─ 4. Execute Tool ────────────┐  │
│  │  ask_document / summarize... │  │
│  │  结果塞回 messages           │  │
│  └───────┬──────────────────────┘  │
│          └──── 循环回到 2 ──────┘  │
│                                    │
│  ┌─ 5. 无工具标记 → 输出完毕 ──┐  │
│  │  token 已流式展示，yield done│  │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### 为什么自写 Agent，不用 LangChain create_agent？

| 问题 | 说明 |
|------|------|
| **DeepSeek reasoning bug** | LangGraph 序列化消息时丢失 `reasoning_content`，DeepSeek API 400 报错 |
| **逐字流式** | 框架层不支持 `for char in token` 级别的流式粒度控制 |
| **工具调用兼容** | 文本标记 `⚙️TOOL:` 不依赖模型 function calling 能力，任何 LLM 通用 |

**项目切换 `deepseek-chat`（非 reasoning 模型）即可直接使用主流 Agent 框架。** 自写 loop 是为了展示对 Agent 原理的完整理解。

---

## 🚀 快速体验

```bash
cd ai-doc-assistant
pip install -r requirements.txt
cp .env.example .env     # 填写 API key
streamlit run app.py
```

或 Docker：

```bash
docker-compose up        # 内置 Ollama
```

---

## ✨ 功能矩阵

| 功能 | 说明 | Agent 支持 |
|------|------|:---:|
| 🧠 **AI Agent** | 自然语言驱动，自主判断意图、调用工具、多步推理 | ⭐ |
| 💬 **智能问答** | RAG 检索增强，向量库搜索 + LLM 生成 | ⭐ |
| 📝 **文档摘要** | 5 种摘要类型（short/detailed/bullet/executive/qa） | ⭐ |
| 🏗️ **结构分析** | 正则提取标题 + LLM 语义理解，文档类型/质量评估 | ⭐ |
| 🔍 **实体提取** | 关键词、行动项、主题提取，LLM 验证相关性 | ⭐ |
| 🌍 **文档翻译** | 8 语言互译，自动检测源语言，4 格式下载 | ⭐ |
| 📊 **报告生成** | 一键生成综合 Markdown 分析报告 | ⭐ |
| 🔄 **文档对比** | 相似度计算、差异摘要、逐行高亮对比 | ⭐ |

---

## 🛠️ 技术栈

| 层 | 技术 | 用途 |
|----|------|------|
| 前端 | **Streamlit** | Web UI + 自定义 Arctic Frost 主题 |
| Agent 框架 | **自定义 ReAct** | Think→Act→Observe 循环，文本标记工具调用 |
| RAG | **LangChain + ChromaDB** | 文档分块、向量化、相似度检索 |
| LLM | **DeepSeek / GPT / Ollama** | 多 provider 自动 fallback |
| 文档解析 | **PyPDF2 / python-docx / openpyxl / python-pptx** | PDF/DOCX/XLSX/PPTX/TXT/MD |
| 缓存 | **SemanticCache (Jaccard + LRU)** | QA 语义缓存 |
| 持久化 | **JSON session store** | 会话历史 + 生成内容保存 |
| 日志 | **Rotating file + console** | info/error/detail 三级 |
| 测试 | **pytest（210 用例）** | 单元测试 + mock 覆盖 |

---

## 📁 项目结构

```
ai-doc-assistant/
├── app.py                 # Streamlit 入口
├── src/                   # 业务逻辑层
│   ├── agent.py           # 🤖 AgentSession ReAct 循环（核心）
│   ├── agent_tools.py     # 7 个 LangChain @tool
│   ├── qa_engine.py       # RAG 问答引擎
│   ├── vector_store.py    # ChromaDB 向量库管理
│   ├── summary_engine.py  # 摘要生成
│   ├── structure_analyzer.py  # 文档结构分析
│   ├── keyword_extractor.py   # 信息提取
│   ├── translation_engine.py  # 翻译引擎
│   ├── report_generator.py    # 报告生成
│   ├── document_comparer.py   # 文档对比
│   ├── cache_manager.py    # 语义缓存
│   ├── session_manager.py  # 会话持久化
│   └── ...
├── ui/                    # UI 组件
│   ├── agent_chat.py      # Agent 对话界面
│   ├── sidebar.py         # 侧边栏导航
│   └── *.py               # 各功能 Tab
├── tests/                 # pytest (210 用例)
└── Dockerfile / docker-compose.yml
```

---

## 🧪 测试

```bash
pytest tests/ -v           # 210 用例
pytest tests/ --cov=src    # 覆盖率
```

---

## 🎯 项目亮点

- **ReAct Agent 手写实现**：理解 Agent 本质，能独立实现而不依赖黑盒框架
- **解决 DeepSeek 兼容性问题**：定位 LangGraph 序列化与 reasoning model 的冲突，用文本标记绕开，非回避而是解决
- **生产级工程实践**：依赖注入、模块解耦、日志系统、语义缓存、会话持久化、210 个单测
- **架构决策能力**：对比 LangChain create_agent / CrewAI / 自写方案的 trade-off，能讲清楚选型理由

---

## 📄 License

MIT
