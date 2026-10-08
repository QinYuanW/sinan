"""司南 · 主 Agent

双模型编排：
    glm-4v-flash  -> 看图（通过 recognize_homework 工具）
    deepseek-chat -> 推理与规划

为什么这样拆：
    推理模型是纯文本的，不具备多模态能力。
    把视觉识别收敛成一个工具，既恢复了多模态链路，
    也让模型在需要时才调用视觉能力（按需付费、按需延迟）。

Agent 缓存策略：
    按 (role, user_id, 日期) 缓存。日期进入缓存键，保证每天刷新一次
    系统提示词里的动态数据（剩余天数、打卡连续天数），
    既保证新鲜度，又不用每次都重建 Agent。
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import date as _date
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agent.core.planner import build_context
from app.agent.roles import get_pack
from app.agent.tools.exam_search import search_exam_info
from app.agent.tools.mastery import build_mastery_tools
from app.agent.tools.vision import recognize_photo
from app.common.logger import logger
from app.db.models import DB_PATH
from app.db.repository import SQLiteRepository

load_dotenv()

# 会话记忆库与业务库分开：两者写入频率与生命周期不同，分开避免锁竞争
CHECKPOINT_DB = Path(DB_PATH).parent / "checkpoints.db"

# 全局仓储（FastAPI 单进程内共享）
_repo: SQLiteRepository | None = None
_agent_cache: dict[tuple[str, str, str], object] = {}
_checkpointer: SqliteSaver | None = None


def get_repo() -> SQLiteRepository:
    global _repo
    if _repo is None:
        _repo = SQLiteRepository()
    return _repo


def get_checkpointer() -> SqliteSaver:
    """单例 checkpointer。

    SqliteSaver 持有数据库连接，每次重建都会新占一个连接导致泄漏；
    会话隔离由 thread_id 承担，不需要多个 checkpointer 实例。
    """
    global _checkpointer
    if _checkpointer is None:
        connection = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
        _checkpointer = SqliteSaver(connection)
        _checkpointer.setup()
    return _checkpointer


# ======================================================================
# 工具：学习照片识别
# ======================================================================
@tool
def recognize_homework(photo_url: str) -> str:
    """识别学习照片（习题、试卷、笔记）中的内容。

    返回：题目摘要、涉及的知识点、作答情况（全对/部分错误/空白）、错误原因。

    用户发来图片时**必须先调用本工具**获取图片内容，
    绝不允许凭想象描述图片或直接假设用户做了什么题。

    Args:
        photo_url: 图片地址，支持 http(s) URL、data URL 或本地文件路径
    """
    return recognize_photo(photo_url)


# ======================================================================
# Agent 工厂
# ======================================================================
def _build_agent(user_id: str, role: str):
    pack = get_pack(role)
    repo = get_repo()
    ctx = build_context(repo, user_id, role, pack)

    tools = [search_exam_info, recognize_homework, *build_mastery_tools(repo, user_id, role)]

    middleware = [
        SummarizationMiddleware(
            model="deepseek-chat",
            trigger=("messages", 20),  # 长会话压缩，避免上下文无限膨胀
            keep=("messages", 6),
        )
    ]

    return create_agent(
        model=init_chat_model(model="deepseek-chat"),
        tools=tools,
        system_prompt=pack.system_prompt(ctx),
        checkpointer=get_checkpointer(),
        middleware=middleware,
    )


def get_agent(user_id: str, role: str):
    """按 (user_id, role, 日期) 缓存 Agent 实例"""
    key = (user_id, role, _date.today().isoformat())
    if key not in _agent_cache:
        logger.info(f"[司南] 构建 Agent: user={user_id} role={role}")
        _agent_cache[key] = _build_agent(user_id, role)
    return _agent_cache[key]


def clear_agent_cache() -> None:
    """目标或掌握度发生大变化后可主动调用，强制下次重建"""
    _agent_cache.clear()


# ======================================================================
# 流式对话
# ======================================================================
async def stream_chat(
    prompt: str,
    image: str | None,
    thread_id: str,
    user_id: str = "default",
    role: str = "kaoyan",
):
    """以异步生成器逐块产出回答文本。

    返回的是纯文本片段，SSE 的 `data:` 封装由 API 层负责——
    职责分离，避免业务代码关心传输格式。
    """
    logger.info(f"[司南] user={user_id} role={role} thread={thread_id} prompt={prompt[:60]!r} image={bool(image)}")

    try:
        if image:
            message = HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": prompt or "请识别这张图片中的学习内容。"},
                ]
            )
        else:
            message = HumanMessage(content=prompt)

        agent = get_agent(user_id, role)
        config = {"configurable": {"thread_id": f"{user_id}:{role}:{thread_id}"}}

        # agent.stream 是同步生成器，放进线程池执行，避免阻塞事件循环
        iterator = await asyncio.to_thread(
            lambda: agent.stream(
                {"messages": [message]}, config, stream_mode="messages"
            )
        )

        def _next():
            try:
                return next(iterator)
            except StopIteration:
                return None

        while True:
            item = await asyncio.to_thread(_next)
            if item is None:
                break
            chunk, _metadata = item
            if isinstance(chunk, AIMessage) and getattr(chunk, "content", None):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            yield str(block.get("text", ""))

    except Exception as exc:  # noqa: BLE001 — 对流式接口，异常必须以文本形式交代
        logger.error(f"[司南] 对话失败: {type(exc).__name__}: {exc}")
        yield f"\n[出错] {type(exc).__name__}: {exc}\n请检查模型配置与网络后重试。"


# ======================================================================
# 会话历史
# ======================================================================
def _get_checkpointer() -> SqliteSaver:
    return get_checkpointer()


def get_messages(thread_id: str, user_id: str = "default", role: str = "kaoyan") -> list[dict]:
    """读取会话历史"""
    checkpointer = _get_checkpointer()
    config = {"configurable": {"thread_id": f"{user_id}:{role}:{thread_id}"}}

    try:
        checkpoint = checkpointer.get(config)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[司南] 读取历史失败: {exc}")
        return []

    if not checkpoint:
        return []

    channel_values = (checkpoint or {}).get("channel_values") or {}
    messages = channel_values.get("messages") or []

    result: list[dict] = []
    for msg in messages:
        content = getattr(msg, "content", None)
        if not content:
            continue
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": content})
    return result


def clear_messages(thread_id: str, user_id: str = "default", role: str = "kaoyan") -> None:
    """清空会话历史（只清会话记忆，不影响掌握度画像）"""
    checkpointer = _get_checkpointer()
    checkpointer.delete_thread(f"{user_id}:{role}:{thread_id}")
    logger.info(f"[司南] 已清空会话 user={user_id} role={role} thread={thread_id}")


# ======================================================================
# 惰性导出：支持 langgraph.json 的 `sinan_agent.py:agent` 引用
# ======================================================================
def __getattr__(name: str):
    """Agent 需要按 (user_id, role) 动态构建，因此做模块级惰性导出。

    只有真正访问 `sinan_agent.agent` 时才构建默认身份（考研）的实例，
    避免导入期就要求数据库与模型配置全部就绪。
    """
    if name == "agent":
        return get_agent("default", "kaoyan")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
