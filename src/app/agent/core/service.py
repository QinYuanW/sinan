"""司南 · 业务流程编排

把「引擎层」的原语编排成可被 API 直接调用的业务流程。
API 层只做 HTTP，业务规则全部集中在这里。

四条链路：
    checkin  -> 打卡（轻量，不触发重规划）
    report   -> 汇报（解析 -> 更新画像 -> 重规划）
    daily    -> 今日作战清单
    long     -> 长期阶段规划
"""

from __future__ import annotations

import asyncio
from datetime import date as _date, timedelta
from typing import Any

from app.agent.core.planner import (
    generate_daily_plan,
    generate_long_plan,
    replan_after_report,
)
from app.agent.core.reporter import parse_report
from app.agent.roles import get_pack, list_packs
from app.common.logger import logger
from app.db.repository import SQLiteRepository


def _today() -> str:
    return _date.today().isoformat()


def _repo() -> SQLiteRepository:
    # 延迟导入避免循环依赖（sinan_agent 也持有仓储单例）
    from app.agent.sinan_agent import get_repo

    return get_repo()


# ======================================================================
# 打卡
# ======================================================================
async def handle_checkin(
    user_id: str, role: str, date: str | None = None
) -> dict[str, Any]:
    """打卡：只记录"今天来过了"，不触发任何计划调整。

    为什么不触发重规划：打卡的信息量只是"做没做"，
    不足以支撑任务排序变化。若打卡就重排，用户会觉得计划天天在变，信任度会崩。
    """
    repo = _repo()
    log = repo.check_in(user_id, role, date or _today())
    streak = repo.check_in_streak(user_id, role)

    return {
        "status": "ok",
        "date": log.get("date"),
        "checked_in_at": log.get("checked_in_at"),
        "streak": streak,
        "message": f"打卡成功，已连续打卡 {streak} 天。",
        "next_action": "打卡只记录出勤。要汇报具体学习情况（做题正确率、遇到的问题），请发送 @汇报 + 文字或照片。",
    }


# ======================================================================
# 汇报
# ======================================================================
async def handle_report(
    user_id: str,
    role: str,
    text: str | None = None,
    photos: list[str] | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    """汇报：解析 -> 更新掌握度画像 -> 按需重规划（含稳定性保护）"""
    repo = _repo()
    target_date = date or _today()

    # ---- 1. 解析（视觉识别 + 结构化抽取，都是阻塞调用，放线程池）----
    parse = await asyncio.to_thread(parse_report, text, photos, target_date, role)

    # ---- 2. 无论是否需要追问，先落库原始材料 ----
    repo.save_report(
        user_id,
        role,
        target_date,
        raw=text,
        photos=photos or [],
        parsed=parse.model_dump(),
    )

    # ---- 3. 信息不足 -> 追问，不猜数据 ----
    if parse.needs_followup or not parse.accuracy:
        logger.info(f"[司南] 汇报信息不足，需要追问: {parse.followup_question[:60]}")
        return {
            "status": "need_followup",
            "date": target_date,
            "question": parse.followup_question or "请补充：今天做了哪些知识点，正确率大概多少？",
            "recognized": parse.photos_recognized,
            "problems": parse.problems,
            "message": "信息还不足以判断掌握程度，需要你先补充说明。",
        }

    # ---- 4. 更新画像 + 重规划 ----
    result = await asyncio.to_thread(
        replan_after_report,
        repo,
        user_id,
        role,
        target_date,
        parse.accuracy,
        parse.completed_kps,
        parse.problems,
    )

    # ---- 5. 回填掌握度变化 ----
    repo.save_report(user_id, role, target_date, mastery_delta=result["mastery_delta"])

    logger.info(
        f"[司南] 汇报处理完成 user={user_id} role={role} "
        f"更新{len(result['mastery_delta'])}个知识点 计划变更={result['changed']}"
    )

    return {
        "status": "ok",
        "date": target_date,
        "parsed": parse.model_dump(),
        "mastery_delta": result["mastery_delta"],
        "problems": result["problems"],
        "replan_reason": result["replan_reason"],
        "plan_changed": result["changed"],
        "next_plan": result["next_plan"],
        "summary": _report_summary(parse, result),
    }


def _report_summary(parse, result: dict[str, Any]) -> str:
    """给出一段人类可读的汇报回执"""
    lines = [f"## 汇报回执（{parse.date}）"]

    if parse.completed_kps:
        lines.append("**完成情况**：" + "、".join(parse.completed_kps))

    if parse.accuracy:
        parts = [f"{kp} {acc:.0%}" for kp, acc in parse.accuracy.items()]
        lines.append("**作答证据**：" + "、".join(parts))

    if result.get("mastery_delta"):
        parts = [
            f"{kp} {delta:+.2f}" for kp, delta in result["mastery_delta"].items()
        ]
        lines.append("**画像更新**：" + "、".join(parts))

    if parse.problems:
        lines.append("**暴露的问题**：")
        lines.extend(f"- {p}" for p in parse.problems)

    lines.append(f"**计划处理**：{result.get('replan_reason', '')}")
    return "\n".join(lines)


# ======================================================================
# 计划
# ======================================================================
async def handle_daily_plan(
    user_id: str,
    role: str,
    date: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """今日作战清单。

    默认幂等：当天已有计划就直接返回，不重算——
    这是「只改未来不改今天」的实现层保证。
    """
    repo = _repo()
    plan = await asyncio.to_thread(
        generate_daily_plan, repo, user_id, role, date, force_refresh
    )
    return {"status": "ok", "plan": plan}


async def handle_long_plan(user_id: str, role: str) -> dict[str, Any]:
    repo = _repo()
    plan = await asyncio.to_thread(generate_long_plan, repo, user_id, role)
    return {"status": "ok", "plan": plan}


# ======================================================================
# 身份 / 画像
# ======================================================================
def handle_roles() -> dict[str, Any]:
    return {"status": "ok", "roles": list_packs()}


def handle_profile_get(user_id: str, role: str) -> dict[str, Any]:
    repo = _repo()
    pack = get_pack(role)
    profile = repo.get_profile(user_id, role)
    return {
        "status": "ok",
        "role": role,
        "role_name": pack.name,
        "profile": profile,
        "goal_summary": pack.goal_summary(profile or {}),
        "kp_count": repo.count_kp(user_id, role),
        "streak": repo.check_in_streak(user_id, role),
    }


def handle_profile_set(
    user_id: str,
    role: str,
    name: str | None = None,
    goal_payload: dict[str, Any] | None = None,
    daily_hours: float | None = None,
    subjects: list[str] | None = None,
) -> dict[str, Any]:
    repo = _repo()
    pack = get_pack(role)

    # 用身份包的 goal_schema 校验目标字段，脏数据在入口就被拦住
    if goal_payload:
        validated = pack.goal_schema(**goal_payload)
        goal_payload = validated.model_dump()

    repo.upsert_profile(
        user_id,
        role,
        name=name,
        goal_payload=goal_payload,
        daily_hours=daily_hours,
        subjects=subjects,
    )

    # 目标变化后重建考点树（首次）与刷新 Agent
    from app.agent.core.planner import ensure_kp_tree, generate_daily_plan
    from app.agent.sinan_agent import clear_agent_cache

    ensure_kp_tree(repo, user_id, role, pack)
    clear_agent_cache()

    # 目标变更会让当日计划的隐含假设失效（每日时长、考试日期都可能变了），
    # 因此强制重算一次。这是"只改未来不改今天"的例外：
    # 用户改了目标，就等于明确要求推翻今天的清单。
    today = _today()
    regenerated = False
    if repo.get_plan(user_id, role, "daily", today):
        generate_daily_plan(repo, user_id, role, date=today, force_refresh=True)
        regenerated = True

    profile = repo.get_profile(user_id, role)
    return {
        "status": "ok",
        "profile": profile,
        "goal_summary": pack.goal_summary(profile or {}),
        "plan_regenerated": regenerated,
        "message": "目标已保存，今日清单已按新目标重算。"
        if regenerated
        else "目标已保存，可以生成长期规划了。",
    }


def handle_mastery(user_id: str, role: str) -> dict[str, Any]:
    repo = _repo()
    return {"status": "ok", "summary": repo.mastery_summary(user_id, role)}


def handle_history(user_id: str, role: str, days: int = 14) -> dict[str, Any]:
    repo = _repo()
    logs = repo.list_recent_logs(user_id, role, limit=days)
    return {
        "status": "ok",
        "streak": repo.check_in_streak(user_id, role),
        "logs": logs,
    }
