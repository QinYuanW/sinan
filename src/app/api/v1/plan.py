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


@router.post("/plan/task")
async def update_task_state(req: dict):
    """跳过 / 延期 / 恢复清单里的某个任务（L16）。

    body:
        {"user_id": "u1", "role": "kaoyan", "date": "2026-10-21",
         "kp": "矩阵的秩", "action": "skip" | "defer" | "restore"}

    三种动作的语义（这是本功能最容易做错的地方）：

        skip    今天不做了，**明天也不硬性顺延** → 放回算法池
        defer   今天做不完，**明天排在清单首位**（carry_over 会 +0.15 优先级）
        restore 撤销，恢复正常

    ★ 为什么 skip 和 defer 要分开：
        "跳过"如果还顺延，等于"明天继续烦你"，那用户按它就没意义了。
        "延期"才是"我想做但现在没空"。
        **两个动词对应两种用户意图，不能合成一个"删除"。**
    """
    from datetime import date as _d

    user_id = req.get("user_id") or "default"
    role = req.get("role") or "kaoyan"
    date = req.get("date") or _d.today().isoformat()
    kp = req.get("kp")
    action = (req.get("action") or "").strip().lower()

    if not kp:
        return {"status": "error", "message": "缺少 kp（考点名）"}

    patches = {
        "skip": {"skipped": True, "deferred": False},
        "defer": {"deferred": True, "skipped": False},
        "restore": {"skipped": False, "deferred": False},
    }
    if action not in patches:
        return {
            "status": "error",
            "message": f"未知 action '{action}'，可选: skip / defer / restore",
        }

    repo = get_repo()
    task = repo.update_plan_task(user_id, role, date, kp, patches[action])
    if task is None:
        return {"status": "error", "message": f"清单里没有这个任务: {kp}"}

    return {"status": "ok", "action": action, "task": task}


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
