"""司南 · 工具：本地考点知识库检索（L11 · RAG 第六步）

把知识库包装成一个工具，交给主模型自主调用——这就是 RAG 里的 "G"（生成）：
模型不再是"凭记忆回答"，而是**先拿到检索到的证据，再组织语言**。

═══════════════════════════════════════════════════════════════════════════
和 `search_exam_info`（Tavily）的关系：优先级不同，不是二选一
═══════════════════════════════════════════════════════════════════════════

    search_exam_kb   （本工具）本地 · 快 · 准 · 不花钱 · 内容自有
    search_exam_info （Tavily）联网 · 慢 · 结果不可控 · 有配额

    策略：**先查本地，查不到再联网。**

为什么这个顺序不能反：
    Tavily 搜"泰勒展开"很容易返回考研机构的营销页，
    而本地库是我自己写的考点精讲——**准确性完全不在一个量级**。
    联网的定位是"兜底"：本地库覆盖不到的院校信息、招生政策、最新大纲。

★ 这也是 L11 账本验收标准「真题库检索替代全靠 Tavily」的落点：
  不是删掉 Tavily，而是让高频问题不再依赖它。
  一个 RAG 系统成熟的标志，不是"不用外部搜索"，
  而是"知道自己什么时候该用外部搜索"。

【作业】当前是"靠工具描述引导模型自己选"。
        更硬的做法是在 prompt 里写死优先级，或者做一个路由工具
        （先 kb 命中数 ≥ 阈值就直接返回，不再给模型选的机会）。
        等你觉得模型选得不稳时再来改。
"""

from __future__ import annotations

from langchain.tools import tool

from app.common.logger import logger

# 返回给模型的最大条数。3 条是"够用又不淹"的经验值——
# 检索质量高时 3 条足够定位；全塞进去会让上下文里 80% 是噪声。
TOP_K = 3


@tool
def search_exam_kb(query: str) -> str:
    """检索**本地**考研考点知识库（考点精讲、常见考法、易错点、解题思路）。

    什么时候调用：
      用户问某个**考点怎么考、有什么易错点、解题思路是什么**时，
      **优先调用本工具**——答案在本地，准确且无需联网。

    和 search_exam_info 的关系：
      本工具覆盖"考点本身的知识"；院校信息、招生政策、最新大纲
      这类本地库没有的内容，再用 search_exam_info 联网搜索。
      顺序是：**先查本地，本地没有才联网。**

    Args:
        query: 要查询的问题或考点名，例如"泰勒展开阶数怎么定"、"二重积分换序"
    """
    from app.kb import get_kb

    try:
        kb = get_kb()
    except Exception as exc:  # noqa: BLE001 — 知识库挂了不该拖垮整轮对话
        logger.error(f"[司南·kb] 知识库加载失败: {type(exc).__name__}: {exc}")
        return "本地知识库暂时不可用，请改用 search_exam_info 联网搜索。"

    hits = kb.search(query, top_k=TOP_K)
    if not hits:
        return (
            f"本地知识库未找到与「{query}」相关的内容。\n"
            f"请改用 search_exam_info 联网搜索，或直接说明你掌握的信息。"
        )

    logger.info(
        f"[司南·kb] 检索 '{query[:20]}' 命中 {len(hits)} 条: "
        + " | ".join(h["heading_path"] for h in hits)
    )

    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        meta_bits = [f"{k}={v}" for k, v in (h.get("meta") or {}).items() if k == "subject"]
        head = f"【{i}】{h['heading_path']}"
        if meta_bits:
            head += f"（{'，'.join(meta_bits)}）"
        blocks.append(f"{head}\n{h['text']}")

    return (
        f"从本地考点知识库检索到 {len(hits)} 条相关内容：\n\n"
        + "\n\n".join(blocks)
        + "\n\n请基于以上材料回答；材料未覆盖的部分如实说明，不要补充臆测内容。"
    )


__all__ = ["search_exam_kb"]
