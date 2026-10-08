"""司南 · 打卡与汇报接口

两个接口，一张表，两个入口：
    POST /log/checkin  -> 打卡，只写 checked_in_at，不触发重规划
    POST /log/report   -> 汇报，写 report_*，触发掌握度更新与重规划
"""

from __future__ import annotations

from fastapi import APIRouter

from app.agent.core import service
from app.models.schemas import CheckInIn, ReportIn

router = APIRouter()


@router.post("/log/checkin")
async def checkin(req: CheckInIn):
    """打卡：记录出勤，不调整计划。

    计划稳定性保护的第一层——打卡信息量不足以支撑重排，
    若打卡就重排，用户会看到计划天天变，信任度会崩塌。
    """
    return await service.handle_checkin(req.user_id, req.role, req.date)


@router.post("/log/report")
async def report(req: ReportIn):
    """学习汇报：文字 + 照片双通道。

    流程：照片识别 -> 结构化解析 -> 掌握度画像更新 -> 按需重规划。
    信息不足以判断掌握度时返回 need_followup，要求补充说明而不是猜测。
    """
    return await service.handle_report(
        req.user_id, req.role, text=req.text, photos=req.photos, date=req.date
    )


@router.get("/log/history")
def history(user_id: str = "default", role: str = "kaoyan", days: int = 14):
    """最近的打卡与汇报记录 + 连续打卡天数"""
    return service.handle_history(user_id, role, days=max(1, min(days, 90)))


@router.get("/log/today")
def today_log(user_id: str = "default", role: str = "kaoyan"):
    """今天的打卡 / 汇报状态"""
    repo = service._repo()  # noqa: SLF001 — 同包内部使用
    from datetime import date as _date

    today = _date.today().isoformat()
    log = repo.get_daily_log(user_id, role, today)
    return {
        "status": "ok",
        "date": today,
        "checked_in": bool(log and log.get("checked_in_at")),
        "reported": bool(log and log.get("report_raw")),
        "log": log,
        "streak": repo.check_in_streak(user_id, role),
    }
