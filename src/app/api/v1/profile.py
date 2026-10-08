"""司南 · 身份与画像接口"""

from __future__ import annotations

from fastapi import APIRouter

from app.agent.core import service
from app.models.schemas import ProfileIn

router = APIRouter()


@router.get("/roles")
def list_roles():
    """列出平台支持的身份包（考研 / 专升本 / 教师 …）"""
    return service.handle_roles()


@router.get("/profile")
def get_profile(user_id: str = "default", role: str = "kaoyan"):
    """读取用户目标与画像概览"""
    return service.handle_profile_get(user_id, role)


@router.put("/profile")
def set_profile(req: ProfileIn):
    """设置 / 更新目标。

    goal_payload 会用该身份包的 goal_schema 校验，
    不同身份的目标字段不同（考研要考试日期，教师要教学周期），因此这里是动态校验。
    """
    return service.handle_profile_set(
        req.user_id,
        req.role,
        name=req.name,
        goal_payload=req.goal_payload,
        daily_hours=req.daily_hours,
        subjects=req.subjects,
    )


@router.get("/mastery")
def mastery(user_id: str = "default", role: str = "kaoyan"):
    """掌握度画像：平均分、最薄弱与最强的知识点"""
    return service.handle_mastery(user_id, role)


@router.get("/kp")
def list_kp(
    user_id: str = "default",
    role: str = "kaoyan",
    subject: str | None = None,
    chapter: str | None = None,
):
    """查看考点树"""
    from app.agent.sinan_agent import get_repo

    repo = get_repo()
    return {
        "status": "ok",
        "kps": repo.list_kp(user_id, role, subject=subject, chapter=chapter),
    }
