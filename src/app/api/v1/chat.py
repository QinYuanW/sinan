"""司南 · 对话接口

统一入口：一个聊天框承载所有交互。
后端先做意图路由，命中业务链路（打卡/汇报/规划）就直接执行并返回结构化回执，
未命中才交给主 Agent 自主决策。

SSE 说明（此处修复了原项目的缺陷）：
    原实现声明 media_type="text/event-stream" 却直接 yield 裸文本，
    没有 `data:` 前缀，导致 EventSource 收到的事件不含任何内容。
    这里按规范封装：每行加 data: 前缀，消息之间用空行分隔，结束发 [DONE]。
"""

from __future__ import annotations

import json
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


def sse(payload: Any) -> str:
    """按 SSE 规范封装一条消息。

    payload 可以是 dict（会被 JSON 序列化）或字符串；
    多行内容必须逐行加 data: 前缀，否则接收端会截断。
    """
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return "".join(f"data: {line}\n" for line in text.split("\n")) + "\n"


@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """流式对话（含意图路由）"""

    async def event_stream():
        intent, _cleaned = route(request.message)
        photos = [request.image_url] if request.image_url else []
        logger.info(
            f"[司南] 意图={intent.value} role={request.role} user={request.user_id}"
        )

        try:
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
                async for piece in stream_chat(
                    strip_at_mentions(request.message),
                    request.image_url,
                    request.thread_id,
                    request.user_id,
                    request.role,
                ):
                    yield sse({"type": "text", "text": piece})

        except Exception as exc:  # noqa: BLE001 — 流式接口必须把错误也变成一条消息
            logger.error(f"[司南] 流式处理失败: {type(exc).__name__}: {exc}")
            yield sse({"type": "error", "text": f"{type(exc).__name__}: {exc}"})

        yield sse("[DONE]")

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=SSE_HEADERS
    )


# ======================================================================
# 会话历史
# ======================================================================
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
