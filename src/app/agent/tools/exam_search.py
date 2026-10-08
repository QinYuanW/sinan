"""司南 · 检索工具

用 Tavily 查证考点频次、真题、院校信息等外部事实。
设计原则：模型不编造事实，凡是不确定的都先检索。
返回 JSON 字符串而不是网页原文，避免污染上下文。
"""

from __future__ import annotations

import json

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

_tavily: TavilySearch | None = None


def get_tavily() -> TavilySearch:
    """懒加载，避免导入期就要求 TAVILY_API_KEY"""
    global _tavily
    if _tavily is None:
        _tavily = TavilySearch(max_results=5, topic="general", include_images=True)
    return _tavily


@tool
def search_exam_info(query: str) -> str:
    """检索与备考相关的外部信息：考点在真题中的出现频次、真题原文、目标院校与专业信息、
    考试大纲变化、复习方法等。

    当你不确定某个考点的考察频率、或需要引用真题依据时，必须先调用本工具核实，
    不要凭记忆编造考频数据。

    Args:
        query: 检索关键词，例如"数学二 中值定理 历年真题 考察频率"
    """
    try:
        result = get_tavily().invoke(query)
    except Exception as exc:  # noqa: BLE001 — 检索失败要让模型知道，而不是静默返回空
        return json.dumps(
            {"query": query, "error": f"{type(exc).__name__}: {exc}", "results": []},
            ensure_ascii=False,
        )

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (TypeError, ValueError):
            return json.dumps({"query": query, "results": [], "raw": result[:500]}, ensure_ascii=False)

    items = []
    for r in (result or {}).get("results", []):
        item = {
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": (r.get("content") or "")[:300],
        }
        images = r.get("images") or []
        if images:
            item["images"] = images[:3]
        items.append(item)

    return json.dumps({"query": query, "results": items}, ensure_ascii=False)
