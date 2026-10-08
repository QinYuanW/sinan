"""司南 · 计划接口"""

from __future__ import annotations

from fastapi import APIRouter

from app.agent.core import service
from app.agent.sinan_agent import get_repo
from app.models.schemas import PlanRequest

router = APIRouter()


@router.post("/plan/daily")
async def daily_plan(req: PlanRequest):
    """生成（或读取）今日作战清单。

    默认幂等：当天已有计划则直接返回，不重算。
    需要强制重排时传 force_refresh=true。
    """
    return await service.handle_daily_plan(
        req.user_id, req.role, req.date, req.force_refresh
    )


@router.get("/plan/long")
async def long_plan(user_id: str = "default", role: str = "kaoyan"):
    """生成长期阶段规划（基础 / 强化 / 冲刺三阶段里程碑）"""
    return await service.handle_long_plan(user_id, role)


@router.get("/plan/saved")
def saved_plans(
    user_id: str = "default",
    role: str = "kaoyan",
    ptype: str = "daily",
    limit: int = 10,
):
    """查询已保存的计划历史"""
    repo = get_repo()
    return {
        "status": "ok",
        "plans": repo.list_plans(user_id, role, ptype, limit=max(1, min(limit, 50))),
    }
