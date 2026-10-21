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
from app.agent.tools.exam_kb import search_exam_kb
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

# 摘要模型单例（懒加载，见 B10）——与主 Agent 分开持有，互不干扰
_SUMMARIZER = None


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


def get_summarizer_model():
    """懒加载摘要模型（B10 懒加载单例 / B3 多模型编排的简化版）。

    只在第一次真正用到 summarize_notes 时才创建客户端——
    这样即使模型 key 没配好，模块照样能正常导入，报错推迟到使用时。
    """
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = init_chat_model(model="deepseek-chat")
    return _SUMMARIZER


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
# 工具：笔记摘要（B2 · @tool 内部再调一次模型）
# ======================================================================
@tool
def summarize_notes(text: str) -> str:
    """把一段学习笔记压缩成 3 条要点（固定格式，便于写进档案）。

    用户粘贴大段笔记、错题描述，或说"帮我总结 / 提炼一下"时**必须调用本工具**，
    不要自己总结——本工具使用独立的摘要模型，输出统一格式。

    Args:
        text: 原始笔记文本，可以包含多段、错题描述或知识点摘录
    """
    result = get_summarizer_model().invoke(
        "把下面这段学习笔记压缩成 3 条要点。要求："
        "每条不超过 30 字；保留具体的知识点名称与数字；"
        "不要寒暄、不要评论、不要加小标题，直接输出 3 条并用 1. 2. 3. 编号。\n\n"
        f"{text}"
    )
    return result.content


# ======================================================================
# Agent 工厂
# ======================================================================
def _build_agent(user_id: str, role: str):
    pack = get_pack(role)
    repo = get_repo()
    ctx = build_context(repo, user_id, role, pack)

    tools = [
        # ★ L11：本地知识库排在联网搜索**前面**。
        #   工具列表的顺序不决定调用顺序（模型自己选），但它是给读代码的人看的
        #   "优先级声明"——先查自有知识，再联网兜底。
        search_exam_kb,
        search_exam_info,
        recognize_homework,
        summarize_notes,
        *build_mastery_tools(repo, user_id, role),
    ]

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
def _chunk_to_events(chunk, seen_tools: set[str]):
    """把模型输出的一帧翻译成语义事件（异步版与同步版共用）。

    这是两个版本之间【唯一共享】的逻辑——它们真正的差异只在于
    「怎么拿到下一帧」：一个是阻塞迭代，一个是丢进线程池 await。
    """
    # ① 工具调用信号：模型决定调工具的那一帧会带上 tool_call_chunks
    for tc in (getattr(chunk, "tool_call_chunks", None) or []):
        name = tc.get("name")
        if name and name not in seen_tools:
            seen_tools.add(name)
            yield ("tool_call", {"name": name})

    # ② 正文片段
    if isinstance(chunk, AIMessage) and getattr(chunk, "content", None):
        content = chunk.content
        if isinstance(content, str) and content:
            yield ("text", content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    yield ("text", str(block.get("text", "")))


async def stream_chat(
    prompt: str,
    image: str | None,
    thread_id: str,
    user_id: str = "default",
    role: str = "kaoyan",
):
    """以异步生成器逐块产出**结构化事件**。

    产出 (kind, payload) 二元组：
        ("text",      "正文片段")           —— 模型输出的正文
        ("tool_call", {"name": "工具名"})    —— 模型决定调用工具（供前端提示进度）

    注意：本函数只产出语义事件，不碰 `data:` / `event:` 这类传输格式——
    封装仍由 API 层负责，保持职责分离。
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

        seen_tools: set[str] = set()  # 同一个工具在本轮只提示一次

        while True:
            item = await asyncio.to_thread(_next)
            if item is None:
                break
            chunk, _metadata = item
            # 注意：async 函数里【不能】写 `yield from`（PEP 525 明确禁止），
            # 只能显式循环逐条 yield。同步版（stream_chat_sync）才可以用。
            for evt in _chunk_to_events(chunk, seen_tools):
                yield evt

    except Exception as exc:  # noqa: BLE001 — 对流式接口，异常必须以文本形式交代
        logger.error(f"[司南] 对话失败: {type(exc).__name__}: {exc}")
        yield ("text", f"\n[出错] {type(exc).__name__}: {exc}\n请检查模型配置与网络后重试。")


def stream_chat_sync(
    prompt: str,
    image: str | None,
    thread_id: str,
    user_id: str = "default",
    role: str = "kaoyan",
):
    """★ 对照版：与 stream_chat 逐行对齐，但**全程同步**。

    唯一的差别就是下面那行 `for chunk in agent.stream(...)`——
    异步版必须把每一步都塞进 `asyncio.to_thread`，这里不用。

    为什么要保留它：
      1. CLI / 脚本 / Jupyter 里直接迭代，省掉事件循环；
      2. 它是**反面教材**：把这个函数塞进 FastAPI 的 async 路由，
         整个事件循环会被它按住不动——量化实测见
         `examples/b07.1_async_vs_sync.py`。

    产出的事件格式与 stream_chat 完全一致，两者可直接互换。
    """
    logger.info(
        f"[司南·sync] user={user_id} role={role} thread={thread_id} prompt={prompt[:60]!r} image={bool(image)}"
    )

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

        seen_tools: set[str] = set()

        # ★ 就是这一行，两个版本的真正分野
        for chunk, _metadata in agent.stream(
            {"messages": [message]}, config, stream_mode="messages"
        ):
            yield from _chunk_to_events(chunk, seen_tools)

    except Exception as exc:  # noqa: BLE001
        logger.error(f"[司南·sync] 对话失败: {type(exc).__name__}: {exc}")
        yield ("text", f"\n[出错] {type(exc).__name__}: {exc}\n请检查模型配置与网络后重试。")


# ======================================================================
# 会话历史
# ======================================================================
def _get_checkpointer() -> SqliteSaver:
    """获取检查点保存器。"""
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
