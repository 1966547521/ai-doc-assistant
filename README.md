# AI 智能文档分析平台

> 手写 ReAct Agent + RAG 的全栈文档分析系统。一句话指令，自动完成多步分析。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LLM](https://img.shields.io/badge/LLM-DeepSeek%2FGPT%2FOllama-green)
![Tests](https://img.shields.io/badge/Tests-210_passed-brightgreen)

---

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 API Key
streamlit run app.py
```

或 Docker：`docker-compose up`

---

## 它能做什么

用户只需输入一句话（如"总结这份文档并用日语导出报告"），系统自动拆解意图、分步调用分析工具，最终交付完整结果。

| 功能     | 说明                                                |
| -------- | --------------------------------------------------- |
| AI Agent | 自然语言驱动，自主判断意图、多步推理、调用工具      |
| 智能问答 | RAG 检索增强，向量搜索 + LLM 生成                   |
| 文档摘要 | 5 种摘要类型（简短 / 详细 / 要点 / 执行摘要 / Q&A） |
| 结构分析 | 标题提取 + 文档类型/质量评估                        |
| 实体提取 | 关键词、行动项、主题提取                            |
| 文档翻译 | 8 种语言互译，4 种格式下载                          |
| 报告生成 | 一键生成综合 Markdown 报告                          |
| 文档对比 | 相似度计算 + 差异摘要 + 逐行高亮                    |

---

## 为什么手写 Agent 而非 LangChain

LangGraph 与 DeepSeek reasoning 模型存在序列化兼容问题（`reasoning_content` 字段丢失导致 API 400）。

本项目用**文本标记 `⚙️TOOL:` + 手写轮询**替代框架调用，不依赖模型的 function calling 能力，可对接任意 LLM。

---

## 项目结构

```
├── app.py               # 入口
├── src/
│   ├── agent.py         # ReAct Agent 循环（核心）
│   ├── agent_tools.py   # 7 个分析工具
│   ├── qa_engine.py     # RAG 问答
│   ├── vector_store.py  # ChromaDB 向量库
│   ├── summary_engine.py
│   ├── translation_engine.py
│   ├── document_comparer.py
│   ├── report_generator.py
│   ├── cache_manager.py # 语义缓存
│   ├── session_manager.py
│   └── ...
├── ui/                  # Streamlit 界面
├── tests/               # 210 个单元测试
└── Dockerfile
```

---

## 技术栈

- **前端**: Streamlit
- **Agent**: 自定义 ReAct（手写轮询）
- **RAG**: LangChain + ChromaDB
- **LLM**: DeepSeek / GPT / Ollama（多 Provider 自动降级）
- **文档解析**: PyPDF2 / python-docx / openpyxl / python-pptx
- **测试**: pytest（210 用例，含 mock 覆盖）
- **部署**: Docker / docker-compose
