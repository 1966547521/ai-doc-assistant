---
title: 智能文档助手项目文档
author: Benchmark
---

## 项目背景

本项目旨在构建一个基于 RAG（检索增强生成）的智能文档问答系统。系统支持 PDF、DOCX、XLSX、PPTX、TXT、Markdown 六种文档格式的解析与索引。

## 技术架构

系统采用四层模块化架构：文档处理层(DocumentProcessor)负责多格式解析与分块；向量存储层(VectorStoreManager)基于 ChromaDB 实现增量索引与 SHA-256 去重；问答引擎层(QAEngine)实现 RAG 链路，集成语义缓存；Agent 层(AgentSession)采用手写 ReAct 循环，兼容 DeepSeek 推理模型。

## 核心功能

1. 六种文档格式自动解析与清洗
2. 基于 RecursiveCharacterTextSplitter 的分块（chunk_size=1000, overlap=200）
3. 向量检索与 Top-K 召回（top_k=5）
4. 语义缓存（TTL+LRU 淘汰策略）
5. 多模型嵌入（DashScope → OpenAI → Ollama 三级回退，自动适配维度）
6. 手写 ReAct Agent，支持文档问答/摘要/结构分析/信息提取/翻译/报告生成/文档对比

## Embedding 配置

系统通过环境变量选择 Embedding 模型：
- DASHSCOPE_API_KEY → text-embedding-v3
- OPENAI_API_KEY → text-embedding-3-small
- 均不可用时 → 回退到 Ollama(nomic-embed-text)
三种模型的输出维度不同（1536/1536/768），系统自动检测向量库已有维度并匹配。

## Agent 工具集

Agent 通过文本标记 ⚙️TOOL: / ⚙️END 调用工具，避免 DeepSeek 推理模型在原生 tool calling 中返回 extra reasoning_content 的问题。单会话最多 8 轮工具调用，支持多步协作（如先总结后翻译）。

## 测试覆盖

系统包含 210 个单元测试，覆盖 DocumentProcessor、VectorStore、QAEngine、AgentSession、CacheManager 等全部核心模块。测试使用 mock 隔离外部依赖，可在无 API Key 环境下运行。

