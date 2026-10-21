"""司南 · 知识库 · 向量化（L11 · 可插拔 Provider）

RAG 第三步：把文本变成向量。

═══════════════════════════════════════════════════════════════════════════
为什么这一层要"可插拔"
═══════════════════════════════════════════════════════════════════════════

embedding 是 RAG 链路里**唯一绑死外部服务**的一环：

    · 模型换版本（embedding-2 → embedding-3），维度变了，索引要重建
    · 余额用完（本项目当前就是这个情况：HTTP 429 / code 1113）
    · 换成自建服务或本地模型

这三件事都**不应该**牵动检索逻辑。所以这里只暴露一个统一接口：

    get_embedder() -> Callable[[list[str]], list[list[float]]] | None

返回 None 表示"没有可用的向量模型"——调用方据此退回纯稀疏检索，
而不是崩掉或拿假向量糊弄。**"优雅降级"比"必须成功"更适合生产。**

支持的后端（由环境变量 `KB_EMBED_PROVIDER` 选择）：

    siliconflow  硅基流动，默认模型 `BAAI/bge-m3`（1024 维，中文表现好）
                 需 SILICONFLOW_API_KEY / SILICONFLOW_BASE_URL
    zhipu        智谱 embedding-3（需 ZHIPUAI_API_KEY）
    openai       OpenAI 或任何 OpenAI 兼容服务（需 OPENAI_API_KEY / OPENAI_BASE_URL）
    none         显式关闭向量检索

★ 这三家**只写了一份实现**（`_openai_compatible`）——
  因为它们的 embeddings 接口协议完全一致，换服务商本质上只换
  「key + base_url + 模型名」三样东西。这就是"OpenAI 兼容"的价值。

不设 `KB_EMBED_PROVIDER` 时按 `_AUTO_ORDER` 探测：siliconflow → zhipu → openai，
都没有就返回 None（退回稀疏检索）。
（顺序不是"谁更好"，是"谁的可用性更可预期"——智谱额度用尽会 429，
 排前面只会让每次启动白撞一次墙。）
"""

from __future__ import annotations

import os
from typing import Any, Callable

import requests

from app.common.logger import logger

EmbedFn = Callable[[list[str]], list[list[float]]]

# 各家模型维度不同，检索前必须知道维度才能建索引/校验
KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-zh-v1.5": 1024,
    "embedding-3": 2048,
    "embedding-2": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def _openai_compatible(
    key_env: str,
    base_env: str,
    default_base: str,
    default_model: str,
    label: str,
) -> EmbedFn | None:
    """通用 OpenAI 兼容 embedding 客户端。

    ★ 为什么三家后端只写一份实现：
        智谱、SiliconFlow、OpenAI 的 embedding 接口**协议完全一致**——
        都是 `POST {base}/embeddings`，body `{model, input}`，
        返回 `{data: [{index, embedding}]}`。
        所以"换服务商"本质上只换三样东西：**key、base_url、model 名**。

      这就是"OpenAI 兼容"这个词的价值：它把 N 个厂商收敛成一种接口。
      项目里另一处同款证据是 `vision.py` 里
      `init_chat_model(model="glm-4v-flash", model_provider="openai")` ——
      用 OpenAI 的协议调智谱的视觉模型。

    收尾时按 `index` 排序：批量请求的返回顺序**不保证**与输入一致，
    不排序会导致 A 段文本配上 B 段向量——**静默错位，比报错危险得多**。
    """
    key = os.getenv(key_env)
    if not key:
        return None
    base = os.getenv(base_env) or default_base
    model = os.getenv("KB_EMBED_MODEL") or default_model
    url = f"{base.rstrip('/')}/embeddings"

    def embed(texts: list[str]) -> list[list[float]]:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "input": texts},
            timeout=90,
        )
        if resp.status_code != 200:
            # 把"余额不足"这类业务错误原样带出来，别吞成"检索失败"
            raise RuntimeError(
                f"{label} embedding 失败 HTTP {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()["data"]
        return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]

    logger.info(f"[司南·kb] embedder = {label}:{model}  ({url})")
    return embed


# 三家都走同一个 OpenAI 兼容实现，只是参数不同
_PROVIDERS: dict[str, Callable[[], EmbedFn | None]] = {
    "siliconflow": lambda: _openai_compatible(
        "SILICONFLOW_API_KEY", "SILICONFLOW_BASE_URL",
        "https://api.siliconflow.cn/v1", "BAAI/bge-m3", "siliconflow",
    ),
    "zhipu": lambda: _openai_compatible(
        "ZHIPUAI_API_KEY", "ZHIPUAI_BASE_URL",
        "https://open.bigmodel.cn/api/paas/v4/", "embedding-3", "zhipu",
    ),
    "openai": lambda: _openai_compatible(
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "https://api.openai.com/v1", "text-embedding-3-small", "openai",
    ),
    "none": lambda: None,
}

# 自动探测顺序。siliconflow 在前是**有意为之**：
# 智谱的 embedding 免费额度用尽后返回 429，把它排在前面会让每次启动
# 都白撞一次墙（探测只看 key 在不在，不看余额）。
# 顺序不是"谁更好"，是"谁的可用性更可预期"。
_AUTO_ORDER = ["siliconflow", "zhipu", "openai"]

_embedder: EmbedFn | None = None
_resolved = False


def get_embedder() -> EmbedFn | None:
    """取当前可用的向量化函数；没有就返回 None（调用方走稀疏检索）。

    只探测一次并缓存结果——探测本身可能要读环境变量、建客户端，
    每次检索都探一遍是浪费。
    """
    global _embedder, _resolved
    if _resolved:
        return _embedder

    name = (os.getenv("KB_EMBED_PROVIDER") or "").strip().lower()
    order = [name] if name in _PROVIDERS else _AUTO_ORDER

    for candidate in order:
        try:
            fn = _PROVIDERS[candidate]()
        except Exception as exc:  # noqa: BLE001 — 探测失败不该阻断启动
            logger.warning(f"[司南·kb] embedder {candidate} 初始化失败: {exc}")
            continue
        if fn is not None:
            _embedder = fn
            break

    _resolved = True
    if _embedder is None:
        logger.info("[司南·kb] 未配置可用的 embedding —— 检索将只走稀疏（BM25）路径")
    return _embedder


def reset_embedder() -> None:
    """清空探测缓存（自检里换环境变量后调用）。"""
    global _embedder, _resolved
    _embedder = None
    _resolved = False


def embed_documents(
    texts: list[str], batch: int = 16, fn: EmbedFn | None = None
) -> list[list[float]]:
    """批量向量化（自动分批，避免单次请求过大）。

    `fn` 可显式传入——这样"建索引"与"查询"能确保用的是同一个 embedder。
    """
    fn = fn or get_embedder()
    if fn is None:
        raise RuntimeError("没有可用的 embedding provider")
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out.extend(fn(texts[i: i + batch]))
    return out


def describe_embedder() -> dict[str, Any]:
    fn = get_embedder()
    return {
        "available": fn is not None,
        "provider": (os.getenv("KB_EMBED_PROVIDER") or "auto").strip() or "auto",
        "model": os.getenv("KB_EMBED_MODEL") or "(默认)",
        "known_dims": KNOWN_DIMS.get(os.getenv("KB_EMBED_MODEL") or "", None),
    }


__all__ = [
    "EmbedFn",
    "get_embedder",
    "reset_embedder",
    "embed_documents",
    "describe_embedder",
    "KNOWN_DIMS",
]
