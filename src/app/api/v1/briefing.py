"""司南 · 每日早报接口（L13）

一个端点，两种用法：

    GET /api/v1/briefing/daily?user_id=u1&role=kaoyan
        读今天的早报（没有就生成）

    GET /api/v1/briefing/daily?user_id=u1&role=kaoyan&refresh=true
        强制重算（会重新搜资讯、重新花 token）

    GET /api/v1/briefing/daily?user_id=u1&role=kaoyan&with_news=false
        跳过资讯（纯复盘，秒回）——前端首屏用这个，避免等 3~5 秒

★ 为什么资讯要能跳过：
    早报的两半成本完全不同——复盘是查库（毫秒级），资讯是搜+总结（秒级且花钱）。
    把"要不要资讯"做成参数，前端就能先秒出复盘、资讯异步补上。
    **一个接口塞两种延迟的东西，是最容易做砸的体验细节。**
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.agent.briefing import build_briefing, describe_briefing
from app.agent.sinan_agent import get_repo
from app.common.logger import logger

router = APIRouter()


@router.get("/briefing/daily")
async def get_daily_briefing(
    user_id: str = "default",
    role: str = "kaoyan",
    date: str | None = None,
    refresh: bool = False,
    with_news: bool = True,
):
    """取（或生成）某天的早报。"""
    repo = get_repo()
    try:
        # build_briefing 里有网络调用（资讯），必须丢线程池，不能阻塞事件循环
        briefing = await asyncio.to_thread(
            build_briefing, repo, user_id, role, date, with_news, refresh
        )
    except Exception as exc:  # noqa: BLE001 — 早报失败不该让前端白屏
        logger.error(f"[司南·briefing] 生成失败: {type(exc).__name__}: {exc}")
        return {
            "status": "error",
            "message": f"早报生成失败：{type(exc).__name__}",
            "briefing": None,
        }

    return {"status": "ok", "briefing": briefing}


@router.get("/briefing/meta")
async def briefing_meta():
    """早报的组成说明（前端调试/答辩演示用）。"""
    return {"status": "ok", **describe_briefing()}
