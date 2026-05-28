"""Generate sample documents for manual benchmark testing.

Usage:
    cd tests/sample_docs
    python generate_samples.py

Creates .txt, .md files always; also tries .docx / .xlsx / .pptx if libraries available.
"""
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SECTIONS = [
    ("项目背景", "本项目旨在构建一个基于 RAG（检索增强生成）的智能文档问答系统。"
     "系统支持 PDF、DOCX、XLSX、PPTX、TXT、Markdown 六种文档格式的解析与索引。"),
    ("技术架构", "系统采用四层模块化架构：文档处理层(DocumentProcessor)负责多格式解析与分块；"
     "向量存储层(VectorStoreManager)基于 ChromaDB 实现增量索引与 SHA-256 去重；"
     "问答引擎层(QAEngine)实现 RAG 链路，集成语义缓存；"
     "Agent 层(AgentSession)采用手写 ReAct 循环，兼容 DeepSeek 推理模型。"),
    ("核心功能", "1. 六种文档格式自动解析与清洗\n"
     "2. 基于 RecursiveCharacterTextSplitter 的分块（chunk_size=1000, overlap=200）\n"
     "3. 向量检索与 Top-K 召回（top_k=5）\n"
     "4. 语义缓存（TTL+LRU 淘汰策略）\n"
     "5. 多模型嵌入（DashScope → OpenAI → Ollama 三级回退，自动适配维度）\n"
     "6. 手写 ReAct Agent，支持文档问答/摘要/结构分析/信息提取/翻译/报告生成/文档对比"),
    ("Embedding 配置", "系统通过环境变量选择 Embedding 模型：\n"
     "- DASHSCOPE_API_KEY → text-embedding-v3\n"
     "- OPENAI_API_KEY → text-embedding-3-small\n"
     "- 均不可用时 → 回退到 Ollama(nomic-embed-text)\n"
     "三种模型的输出维度不同（1536/1536/768），系统自动检测向量库已有维度并匹配。"),
    ("Agent 工具集", "Agent 通过文本标记 ⚙️TOOL: / ⚙️END 调用工具，避免 "
     "DeepSeek 推理模型在原生 tool calling 中返回 extra reasoning_content 的问题。"
     "单会话最多 8 轮工具调用，支持多步协作（如先总结后翻译）。"),
    ("测试覆盖", "系统包含 210 个单元测试，覆盖 DocumentProcessor、VectorStore、"
     "QAEngine、AgentSession、CacheManager 等全部核心模块。"
     "测试使用 mock 隔离外部依赖，可在无 API Key 环境下运行。"),
]


def write_txt():
    path = os.path.join(OUT_DIR, "sample_zh.txt")
    with open(path, "w", encoding="utf-8") as f:
        for title, body in SECTIONS:
            f.write(f"# {title}\n\n{body}\n\n")
    print(f"  {os.path.basename(path)}  — {os.path.getsize(path)} bytes")


def write_md():
    path = os.path.join(OUT_DIR, "sample_zh.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("---\ntitle: 智能文档助手项目文档\nauthor: Benchmark\n---\n\n")
        for title, body in SECTIONS:
            f.write(f"## {title}\n\n{body}\n\n")
    print(f"  {os.path.basename(path)}  — {os.path.getsize(path)} bytes")


def write_en_txt():
    path = os.path.join(OUT_DIR, "sample_en.txt")
    content = (
        "This is a sample English document for benchmark testing.\n\n"
        + "The quick brown fox jumps over the lazy dog. " * 500
        + "\n\nEnd of document."
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  {os.path.basename(path)}  — {os.path.getsize(path)} bytes")


def try_write_docx():
    try:
        from docx import Document
        doc = Document()
        doc.add_heading("智能文档助手 - 样本文档", 0)
        for title, body in SECTIONS:
            doc.add_heading(title, 1)
            doc.add_paragraph(body)
        path = os.path.join(OUT_DIR, "sample_zh.docx")
        doc.save(path)
        print(f"  {os.path.basename(path)}  — {os.path.getsize(path)} bytes")
    except ImportError:
        print("  [skip] python-docx not installed")


def try_write_xlsx():
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "性能测试"
        headers = ["模块", "功能", "参数", "状态"]
        ws.append(headers)
        rows = [
            ("DocumentProcessor", "多格式解析", "6种格式", "完成"),
            ("VectorStore", "增量索引", "SHA-256去重", "完成"),
            ("QAEngine", "RAG问答", "语义缓存", "完成"),
            ("AgentSession", "ReAct循环", "文本标记调用", "完成"),
        ]
        for row in rows:
            ws.append(row)
        path = os.path.join(OUT_DIR, "sample_zh.xlsx")
        wb.save(path)
        print(f"  {os.path.basename(path)}  — {os.path.getsize(path)} bytes")
    except ImportError:
        print("  [skip] openpyxl not installed")


def try_write_pptx():
    try:
        from pptx import Presentation
        prs = Presentation()
        for title, body in SECTIONS[:4]:
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = title
            slide.placeholders[1].text = body[:200]
        path = os.path.join(OUT_DIR, "sample_zh.pptx")
        prs.save(path)
        print(f"  {os.path.basename(path)}  — {os.path.getsize(path)} bytes")
    except ImportError:
        print("  [skip] python-pptx not installed")


if __name__ == "__main__":
    print("Generating sample documents...")
    write_txt()
    write_md()
    write_en_txt()
    try_write_docx()
    try_write_xlsx()
    try_write_pptx()
    print("Done.")
