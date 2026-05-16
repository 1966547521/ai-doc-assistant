# 📚 AI 智能文档助手 (AI Document Assistant)

基于 **RAG** 架构的全栈 AI 文档处理平台，支持多格式文档上传、智能问答、摘要生成、结构分析、实体提取、翻译和报告生成。

## 🏗️ 架构

```
用户 → Streamlit UI
         ↓
   ┌─────────────────────────────────────┐
   │  DocumentProcessor                  │  PDF/DOCX/XLSX/PPTX/TXT/MD
   │  VectorStoreManager (ChromaDB)      │  向量化 + 增量去重
   │  QAEngine (LangChain RAG)           │  检索增强生成
   │  SummaryEngine / StructureAnalyzer  │  摘要 / 结构分析
   │  TranslationEngine                  │  8 语言翻译
   │  ReportGenerator                    │  综合分析报告
   │  DocumentComparer                   │  diff 对比
   └─────────────────────────────────────┘
         ↓
   LLM Provider (DeepSeek / DashScope / Ollama)
   Embedding Provider (DashScope / OpenAI / Ollama)
         ↓
   SemanticCache (LRU + Jaccard similarity)
   SessionManager (sessions.json 持久化)
```

## 🚀 快速开始

### 前置条件

- Python 3.10+
- [Ollama](https://ollama.com) (可选，用于本地模型)

### 安装

```bash
cd ai-doc-assistant
pip install -r requirements.txt
```

### 配置

复制 `.env.example` 为 `.env`，填写 API 密钥：

```bash
cp .env.example .env
```

至少配置以下之一：

| 方式 | 需要设置 |
|------|---------|
| DeepSeek API | `DEEPSEEK_API_KEY` |
| DashScope (阿里云) | `DASHSCOPE_API_KEY` |
| Ollama 本地 | 安装并启动 Ollama（无需 API key） |

### 启动

```bash
streamlit run app.py
```

打开 `http://localhost:8501`

### Docker (推荐)

```bash
docker-compose up
```

## ✨ 功能

| 功能 | 说明 |
|------|------|
| 💬 **智能问答** | RAG 检索增强生成，支持多轮对话、参考来源、语义缓存、回答中断 |
| 📝 **文档摘要** | 5 种摘要类型（简明/详细/要点/执行/问答式），流式输出 |
| 🏗️ **结构分析** | 正则提取 + LLM 语义理解，文档类型识别、章节总结、质量评估 |
| 🔍 **实体提取** | 关键词、行动项、主题提取，LLM 验证相关性 |
| 🌍 **文档翻译** | 8 种语言互译，自动检测源语言，TXT/MD/DOCX/PDF 下载 |
| 📊 **报告生成** | 一键生成 Markdown/纯文本综合分析报告 |
| 🔄 **文档对比** | 相似度计算、差异摘要、逐行对比 |
| 📜 **会话管理** | 历史会话保存/恢复，生成内容持久化 |

## 🛠️ 技术栈

| 层 | 技术 |
|----|------|
| 前端 | Streamlit (wide layout, custom CSS theme) |
| RAG 框架 | LangChain (chains, retrievers, splitters) |
| 向量数据库 | ChromaDB (persistent, with dimension detection) |
| LLM | DeepSeek v4 / DashScope qwen-plus / Ollama |
| Embeddings | DashScope text-embedding-v3 / OpenAI / Ollama nomic-embed-text |
| 文档解析 | PyPDF2 / python-docx / openpyxl / python-pptx |
| PDF 生成 | reportlab (CJK TrueType font support) |
| 缓存 | 自定义 SemanticCache (Jaccard 相似度 + LRU) |
| 日志 | 自定义 LoggerManager (文件 + 控制台 + 日切) |
| 测试 | pytest (168 用例) |

## 📁 项目结构

```
ai-doc-assistant/
├── app.py                 # Streamlit 入口
├── requirements.txt       # Python 依赖
├── Dockerfile
├── docker-compose.yml
├── .streamlit/
│   └── config.toml        # Streamlit 主题配置
├── src/                   # 核心业务逻辑
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── qa_engine.py
│   ├── summary_engine.py
│   ├── structure_analyzer.py
│   ├── keyword_extractor.py
│   ├── llm_enhancer.py
│   ├── translation_engine.py
│   ├── report_generator.py
│   ├── document_comparer.py
│   ├── cache_manager.py
│   ├── session_manager.py
│   ├── history_manager.py
│   ├── memory_manager.py
│   ├── prompt_manager.py
│   ├── logger.py
│   └── utils.py
├── ui/                    # Streamlit UI 组件
│   ├── theme.py
│   ├── utils.py
│   ├── sidebar.py
│   ├── qa_tab.py
│   ├── summary_tab.py
│   ├── structure_tab.py
│   ├── entity_tab.py
│   ├── translation_tab.py
│   ├── report_tab.py
│   └── compare_tab.py
├── tests/                 # 单元测试 (pytest)
├── prompts/               # Prompt 模板
└── docs/                  # 文档
```

## 🧪 测试

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html  # 覆盖率报告
```

## 📄 License

MIT
