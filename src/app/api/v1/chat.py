"""司南 · 对话接口

统一入口：一个聊天框承载所有交互。
后端先做意图路由，命中业务链路（打卡/汇报/规划）就直接执行并返回结构化回执，
未命中才交给主 Agent 自主决策。

SSE 说明（此处修复了原项目的缺陷）：
    原实现声明 media_type="text/event-stream" 却直接 yield 裸文本，
    没有 `data:` 前缀，导致 EventSource 收到的事件不含任何内容。
    这里按规范封装：每行加 data: 前缀，消息之间用空行分隔，结束发 [DONE]。

═══════════════════════════════════════════════════════════════════════════
L7 说明：两条并行的编排路径
═══════════════════════════════════════════════════════════════════════════

本文件里现在有**两套**业务编排实现：

    路径 A（原有）：`_event_stream_legacy()` —— 一串 if-elif，写在函数体里
    路径 B（L7 新增）：`_event_stream_graph()`  —— 交给 `core/graph.py` 的图

由环境变量 `SINAN_USE_GRAPH` 切换（见下方 `USE_GRAPH`）：

    SINAN_USE_GRAPH=1  → 走图
    未设置 / 其它值     → 走原 if-elif（默认，保证线上行为不变）

为什么默认仍是 A：
    学习型项目的安全做法——原链路一行没删，随时可以对拍。
    等你把 L8（interrupt 追问）做完、图的语义真正超过 if-elif 时，
    再把这个默认值翻过来，然后删掉 `_event_stream_legacy`。

    【作业】把 `USE_GRAPH` 默认改成 True，跑一遍所有场景，确认无回归后再删旧路径。
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agent.core import service
from app.agent.core.router import Intent, route, strip_at_mentions
from app.agent.sinan_agent import clear_messages, get_messages, stream_chat
from app.common.logger import logger
from app.models.schemas import ChatRequest

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 关闭 Nginx 类代理的缓冲，保证逐块下发
}

# ⚠️ 环境变量在**模块导入时**读取一次，进程内固定。
#    这样避免了每个请求都去读环境变量，代价是改完必须重启服务。
#    （想支持热切换就放进函数里读，但没必要——这是学习开关，不是产品功能。）
#
# ⚠️ L8 起两条路**行为不同**，这是刻意的：
#       legacy —— 长期规划生成后**直接落库**（老行为）
#       graph  —— 长期规划生成后**挂起等你确认**，你点头了才落库
#    想体验人工介入，必须用 SINAN_USE_GRAPH=1 启动。
#    这不是 bug，是开关的意义所在：新旧行为可以并排对照。
USE_GRAPH = os.getenv("SINAN_USE_GRAPH", "0").strip() in {"1", "true", "True", "yes"}


def sse(payload: Any, event: str | None = None) -> str:
    """按 SSE 规范封装一条消息。

    payload 可以是 dict（会被 JSON 序列化）或字符串；
    多行内容必须逐行加 data: 前缀，否则接收端会截断。

    event 非空时额外输出一行 `event: <名称>`——接收端据此区分事件类型。
    （浏览器 EventSource 用 addEventListener 监听具名事件；
      像本项目这样手写解析的客户端，需要自己识别 `event:` 行。）
    """
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    lines.extend(f"data: {line}" for line in text.split("\n"))
    return "\n".join(lines) + "\n\n"


@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """流式对话（含意图路由）

    这是一个**薄壳**：只负责选路径、包 SSE、兜异常。
    真正的业务编排在 `_event_stream_legacy` 或 `_event_stream_graph` 里。
    """
    stream_impl = _event_stream_graph if USE_GRAPH else _event_stream_legacy

    async def event_stream():
        try:
            async for line in stream_impl(request):
                yield line
        except Exception as exc:  # noqa: BLE001 — 流式接口必须把错误也变成一条消息
            logger.error(f"[司南] 流式处理失败: {type(exc).__name__}: {exc}")
            yield sse({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
        yield sse("[DONE]")

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=SSE_HEADERS
    )


# ======================================================================
# 路径 B · L7 图编排（新）
# ======================================================================
async def _event_stream_graph(request: ChatRequest):
    """走 `core/graph.py` 的业务图。

    与 legacy 版的差别一目了然：
      · 这里**没有 if-elif**。分流由图的 `decide_branch` 条件边承担。
      · 这里**不 import service**。图自己去调。
      · 本函数的全部工作就是"把图产出的事件翻译成 SSE 帧"。

    一句话：legacy 版把"编排"和"传输"揉在一起，图把两者分开了。

    ★ L8 新增：图可能**跑不完**（interrupt 挂起等人）。
        所以拿到 GraphRun 之后要做两件事：
          1. 把已经产出的事件发出去
          2. 如果 interrupted，把"图在等什么"渲染成一个问题抛给用户

        ⚠️ 注意 `run.resumed` 意味着"用户刚才那句话是对上一次确认的答复"。
           这时图会重新执行被中断的节点，产出"已保存 / 已取消"这类结果，
           走的是和普通事件完全一样的通道——api 层不需要为它写特例。
    """
    from app.agent.core.graph import run_graph  # 延迟导入，避免导入期就编译图

    run = await run_graph(
        message=request.message,
        thread_id=request.thread_id,
        user_id=request.user_id,
        role=request.role,
        image_url=request.image_url,
    )

    logger.info(
        f"[司南·graph] 意图={run.intent} role={request.role} user={request.user_id} "
        f"事件数={len(run.events)} 是答复={run.resumed} 又挂起={run.interrupted}"
    )

    # ---------- ① 先发本跑产出的事件 ----------
    for kind, payload in run.events:
        if kind == "tool_call":
            # 具名事件：前端据此提示"正在调用工具…"
            yield sse(payload, event="tool_call")
        elif kind == "meta":
            # 图产出的 meta 事件语义与 legacy 的 {"type": "meta"} 一致
            yield sse({"type": "meta", **(payload if isinstance(payload, dict) else {})})
        elif kind == "text":
            yield sse({"type": "text", "text": payload})
        elif kind == "status":
            yield sse({"type": "status", **(payload if isinstance(payload, dict) else {"text": payload})})
        elif kind in {"checkin", "plan", "replan", "followup", "confirm_result"}:
            yield sse({"type": kind, **(payload if isinstance(payload, dict) else {})})
        else:
            # 未知事件类型不吞掉：原样透传，便于以后扩事件时不需要改这里
            yield sse({"type": kind, "data": payload})

    # ---------- ② 图挂起了就把"在等什么"问出来 ----------
    if run.interrupted:
        ip = run.interrupt or {}
        display = ip.get("display") or ""
        question = ip.get("question") or "需要你确认一下才能继续。"

        # display 是给人看的正文（这里是渲染好的规划 Markdown）
        if display:
            yield sse({"type": "text", "text": display})
        # 再发一遍问题，保证它一定出现在对话框最末
        yield sse({"type": "text", "text": f"**{question}**"})
        # 结构化事件：将来前端可以据此渲染"确认 / 取消"按钮，
        # 现在不认识也没关系，下面的 text 已经保证了可读性
        yield sse(
            {
                "type": "confirm",
                "kind": ip.get("kind", "confirm"),
                "question": question,
                "plan": ip.get("plan"),
            }
        )


# ======================================================================
# 路径 A · 原 if-elif 编排（保留，供对拍）
# ======================================================================
async def _event_stream_legacy(request: ChatRequest):
    """原实现：业务分流写在函数体里的一串 if-elif。

    ⚠️ 这份代码**没有被删除**，是刻意的：
      它是图版的对照组。当图版行为可疑时，切回这里跑同样的输入，
      差异立刻暴露。等你确信图版更好了，再来删它。
    """
    intent, _cleaned = route(request.message)
    photos = [request.image_url] if request.image_url else []
    logger.info(
        f"[司南] 意图={intent.value} role={request.role} user={request.user_id}"
    )

    # ---------- 打卡 ----------
    if intent is Intent.CHECKIN:
        result = await service.handle_checkin(request.user_id, request.role)
        yield sse({"type": "meta", "intent": "checkin"})
        yield sse({"type": "text", "text": result["message"] + "\n\n" + result["next_action"]})
        yield sse({"type": "checkin", "streak": result["streak"], "date": result["date"]})

    # ---------- 汇报 ----------
    elif intent is Intent.REPORT:
        yield sse({"type": "meta", "intent": "report"})
        yield sse({"type": "status", "text": "正在解析汇报内容…"})
        result = await service.handle_report(
            request.user_id,
            request.role,
            text=strip_at_mentions(request.message) or None,
            photos=photos,
        )
        if result["status"] == "need_followup":
            yield sse({"type": "text", "text": "需要你补充一下：\n\n" + result["question"]})
            yield sse({"type": "followup", "question": result["question"]})
        else:
            yield sse({"type": "text", "text": result["summary"]})
            yield sse({
                "type": "replan",
                "data": {
                    "changed": result["plan_changed"],
                    "mastery_delta": result["mastery_delta"],
                    "reason": result["replan_reason"],
                },
            })
            if result.get("next_plan"):
                yield sse({"type": "plan", "plan": result["next_plan"]})

    # ---------- 今日计划 ----------
    elif intent is Intent.PLAN_DAILY:
        yield sse({"type": "meta", "intent": "plan_daily"})
        result = await service.handle_daily_plan(request.user_id, request.role)
        yield sse({"type": "text", "text": _render_daily_plan(result["plan"])})
        yield sse({"type": "plan", "plan": result["plan"]})

    # ---------- 长期规划 ----------
    elif intent is Intent.PLAN_LONG:
        yield sse({"type": "meta", "intent": "plan_long"})
        result = await service.handle_long_plan(request.user_id, request.role)
        yield sse({"type": "text", "text": _render_long_plan(result["plan"])})

    # ---------- 其他：交给主 Agent ----------
    else:
        yield sse({"type": "meta", "intent": "chat"})
        async for kind, payload in stream_chat(
            strip_at_mentions(request.message),
            request.image_url,
            request.thread_id,
            request.user_id,
            request.role,
        ):
            if kind == "tool_call":
                # 具名事件：前端收到后可以提示"正在调用工具…"
                yield sse(payload, event="tool_call")
            else:
                yield sse({"type": "text", "text": payload})


# ======================================================================
# 会话历史
# ======================================================================
@router.get("/chat/pending")
async def get_pending_confirmation(
    thread_id: str, user_id: str = "default", role: str = "kaoyan"
):
    """查询这个会话是否正在等待人工确认（L8）。

    用途：页面刷新后恢复"待确认"状态。
    因为挂起状态是持久化在 checkpoints 表里的，**隔天打开也依然查得到**——
    这正是 interrupt 相比"前端弹个窗等人点"的本质区别：
    等待这件事本身被存下来了，不依赖页面活着。
    """
    from app.agent.core.graph import get_pending_interrupt  # 延迟导入

    # ⚠️ 别漏 await：它是 async 函数（图用的是异步 checkpointer）。
    #    漏了不会报语法错，只会在运行时抛
    #    `AttributeError: 'coroutine' object has no attribute 'get'` ——
    #    实测踩过一次。改函数签名时**务必全局搜一遍调用点**。
    pending = await get_pending_interrupt(thread_id, user_id, role)
    if not pending:
        return {"pending": False}

    ip = pending.get("interrupt") or {}
    return {
        "pending": True,
        "next": pending["next"],          # 等着被唤醒的节点
        "kind": ip.get("kind"),           # 中断类型，如 plan_confirm
        "question": ip.get("question"),   # 要问用户的话
    }


@router.get("/chat/messages")
async def get_chat_messages(
    thread_id: str, user_id: str = "default", role: str = "kaoyan"
):
    return {"messages": get_messages(thread_id, user_id, role)}


@router.delete("/chat/messages")
async def clear_chat_messages(
    thread_id: str, user_id: str = "default", role: str = "kaoyan"
):
    clear_messages(thread_id, user_id, role)
    return {"success": True}


# ======================================================================
# 渲染辅助：把结构化计划渲染成 Markdown，供聊天框直接展示
# ======================================================================
def _render_daily_plan(plan: dict[str, Any]) -> str:
    lines = [f"## 今日作战清单（{plan.get('date', '')}）"]

    if plan.get("carry_over"):
        lines.append("\n### 昨日顺延")
        for t in plan["carry_over"]:
            lines.append(f"- **{t['kp']}**（{t['est_hours']}h）")

    lines.append("\n### 今日任务")
    for i, t in enumerate(plan.get("tasks", []), start=1):
        lines.append(
            f"\n**{i}. {t['kp']}** ｜ {t['subject']} ｜ 预计 {t['est_hours']} 小时 ｜ 优先级 {t['priority']:.2f}"
        )
        lines.append(f"- 为什么先做它：{t['reason']}")
        lines.append(f"- 验收标准：{t['acceptance']}")

    if not plan.get("tasks"):
        lines.append("\n当前没有需要安排的知识点，请先设置目标或补充考点。")

    lines.append(f"\n### 节奏提醒\n{plan.get('coach_note', '')}")
    lines.append(f"\n总预计耗时：{plan.get('total_hours', 0)} 小时")
    return "\n".join(lines)


def _render_long_plan(plan: dict[str, Any]) -> str:
    lines = [f"## 长期阶段规划（剩余 {plan.get('days_left', '?')} 天）"]
    lines.append(f"\n{plan.get('coverage_note', '')}")

    for ms in plan.get("milestones", []):
        lines.append(f"\n### {ms['name']}（{ms['start_date']} ~ {ms['end_date']}）")
        lines.append(f"- 阶段说明：{ms['note']}")
        kps = ms.get("target_kps", [])
        preview = "、".join(kps[:12]) + ("…" if len(kps) > 12 else "")
        lines.append(f"- 覆盖考点 {len(kps)} 个：{preview}")
    return "\n".join(lines)
