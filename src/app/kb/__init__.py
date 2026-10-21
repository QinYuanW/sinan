"""司南 · 知识库（L11 · RAG）

RAG = Retrieval-Augmented Generation（检索增强生成）。
一句话：**先查资料，再回答**——让模型基于"我给的证据"说话，而不是凭记忆编。

本包结构（对应 RAG 的标准六步）：

    chunker.py      ① 文档加载  ② 分块
    embeddings.py   ③ 向量化（可插拔 provider）
    retriever.py    ④ 索引存储  ⑤ 检索（稀疏 / 稠密 / 混合）
    tools/exam_kb.py ⑥ 接入生成（注册成 Agent 的工具）

对外只需要三个入口：

    from app.kb import get_kb, build, describe_kb

    kb = get_kb()                       # 首次调用自动建索引
    kb.search("泰勒展开怎么用", top_k=3)  # 返回命中的知识块

为什么这个包值得单独存在：
    在这之前，司南回答"泰勒展开怎么考"只能靠 `tools/exam_search.py` 去
    Tavily 搜外网——**每次都联网、结果不可控、还容易搜到营销号**。
    有了本地知识库，高频问题的答案就在自己手里，联网只作为兜底。
    这就是账本 L11 验收标准「真题库检索替代全靠 Tavily」的含义。
"""

from app.kb.retriever import (
    DEFAULT_TOP_K,
    KnowledgeBase,
    build,
    describe_kb,
    get_kb,
    reset_kb,
    tokenize,
)

__all__ = [
    "KnowledgeBase",
    "get_kb",
    "build",
    "reset_kb",
    "describe_kb",
    "tokenize",
    "DEFAULT_TOP_K",
]
