"""司南 · 规划器（与身份无关）

职责：
    1. 组装 PlanningContext（把数据库里的画像/掌握度/打卡记录拼成引擎输入）
    2. 生成长期计划（阶段里程碑）与每日计划（作战清单）
    3. 处理汇报后的重规划，并执行「计划稳定性保护」

计划稳定性保护（防止计划来回震荡，这是用户信任度的关键）：
    ① 只改未来，不改今天：汇报影响的对象是明天的计划，今天已发出的清单不推翻
    ② 小偏差不重排：完成率偏差 ≤ 10% 时只写提醒，不调整任务顺序
    ③ 顺延而非丢弃：未完成的任务进入 carry_over，排在次日清单首位
"""

from __future__ import annotations

from datetime import date as _date, timedelta
from typing import Any

from app.agent.core.priority import build_daily_plan, rank_kps
from app.agent.roles import PlanningContext, RolePack, get_pack
from app.agent.roles.base import get_pack as _get_pack  # noqa: F401  (保持导出兼容)
from app.db.repository import PRIOR_MASTERY, SQLiteRepository

# 完成率偏差低于该阈值时不触发重排
REPLAN_DEVIATION_THRESHOLD = 0.10


def _today() -> str:
    return _date.today().isoformat()


def _parse_date(value: str | None) -> _date:
    if not value:
        return _date.today()
    try:
        return _date.fromisoformat(value)
    except ValueError:
        return _date.today()


# ======================================================================
# 上下文组装
# ======================================================================
def ensure_kp_tree(
    repo: SQLiteRepository, user_id: str, role: str, pack: RolePack
) -> int:
    """首次使用时把身份包的考点树导入数据库。幂等。"""
    if repo.count_kp(user_id, role) > 0:
        return 0
    kps = pack.build_kp_tree()
    return repo.bulk_upsert_kp(user_id, role, kps)


def build_context(
    repo: SQLiteRepository, user_id: str, role: str, pack: RolePack | None = None
) -> PlanningContext:
    """把数据库状态组装成引擎输入"""
    pack = pack or get_pack(role)
    profile = repo.get_profile(user_id, role) or {}
    goal = profile.get("goal_payload") or {}

    exam_date = goal.get("exam_date") or goal.get("term_end") or None
    days_left = None
    if exam_date:
        days_left = (_parse_date(exam_date) - _date.today()).days

    # 上次复习时间走仓储方法（L10：原先这里是裸 SQL，切 Store 后端会漏网）
    last_review: dict[str, str] = repo.get_last_review_map(user_id, role)

    return PlanningContext(
        user_id=user_id,
        role=role,
        kp_list=repo.list_kp(user_id, role),
        mastery_map=repo.get_mastery_map(user_id, role),
        last_review=last_review,
        daily_hours=float(profile.get("daily_hours") or pack.default_daily_hours),
        today=_today(),
        exam_date=exam_date,
        days_left=days_left,
        recent_logs=repo.list_recent_logs(user_id, role, limit=14),
        streak=repo.check_in_streak(user_id, role),
        extra={"goal": goal, **goal},
    )


# ======================================================================
# 每日计划
# ======================================================================
def generate_daily_plan(
    repo: SQLiteRepository,
    user_id: str,
    role: str,
    date: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """生成（或读取）某日的作战清单。

    默认幂等：当天已有 active 计划则直接返回，不重新计算——
    这是「只改未来不改今天」在实现层的落点。
    """
    pack = get_pack(role)
    ensure_kp_tree(repo, user_id, role, pack)
    target_date = date or _today()

    if not force_refresh:
        existing = repo.get_plan(user_id, role, "daily", target_date)
        if existing and existing.get("status") == "active":
            payload = dict(existing["payload"])
            payload["cached"] = True
            return payload

    ctx = build_context(repo, user_id, role, pack)

    # 顺延项：昨天计划里未完成的任务
    carry_over = _collect_carry_over(repo, user_id, role, ctx, target_date)
    plan = build_daily_plan(ctx, pack, target_date, carry_over=carry_over)

    # 入库与返回使用同一份结构，避免"存进去的"和"给出去的"字段不一致
    cleaned = _serialize(plan)
    repo.save_plan(user_id, role, "daily", target_date, cleaned)
    return cleaned


def _collect_carry_over(
    repo: SQLiteRepository,
    user_id: str,
    role: str,
    ctx: PlanningContext,
    target_date: str,
) -> list[dict[str, Any]]:
    """昨日未完成的任务顺延到今日清单首位"""
    prev_date = (_parse_date(target_date) - timedelta(days=1)).isoformat()
    prev_plan = repo.get_plan(user_id, role, "daily", prev_date)
    prev_log = repo.get_daily_log(user_id, role, prev_date)
    if not prev_plan or not prev_log:
        return []

    parsed = prev_log.get("report_parsed") or {}
    done = set(parsed.get("completed_kps") or [])

    carry: list[dict[str, Any]] = []
    for task in (prev_plan.get("payload") or {}).get("tasks", []):
        kp_name = task.get("kp") or task.get("name")
        if not kp_name or kp_name in done:
            continue
        if ctx.mastery_of(kp_name) >= 0.85:
            continue
        item = dict(task)
        item["priority"] = round(min(1.0, float(item.get("priority", 0.5)) + 0.15), 4)
        item["reason"] = "昨日顺延：" + str(item.get("reason", ""))
        item.setdefault("planned_hours", item.get("est_hours", 2.0))
        item.setdefault("partial", False)
        carry.append(item)
    return carry


def _serialize(plan: dict[str, Any]) -> dict[str, Any]:
    """把内部结构序列化为可存库的纯 JSON（剔除 weights 等冗余字段）

    统一字段名为 kp：内部排序结果是原始考点字典（字段名 name），
    入库与对外返回一律用 kp，避免"存进去和给出去的字段不一样"。
    """
    def _clean(task: dict[str, Any]) -> dict[str, Any]:
        kp_name = task.get("kp") or task.get("name") or ""
        est = round(float(task.get("est_hours", 2.0)), 2)
        return {
            "kp": kp_name,
            "subject": task.get("subject", ""),
            "chapter": task.get("chapter", ""),
            "exam_freq": round(float(task.get("exam_freq", 0.5)), 4),
            "mastery": round(float(task.get("mastery", 0.3)), 4),
            "est_hours": est,
            "planned_hours": round(float(task.get("planned_hours", est)), 2),
            "partial": bool(task.get("partial", False)),
            "priority": round(float(task.get("priority", 0.5)), 4),
            "reason": task.get("reason", ""),
            "acceptance": task.get("acceptance", ""),
        }

    return {
        "date": plan["date"],
        "role": plan["role"],
        "total_hours": plan["total_hours"],
        "budget_hours": plan.get("budget_hours"),
        "tasks": [_clean(t) for t in plan.get("tasks", [])],
        "carry_over": [_clean(t) for t in plan.get("carry_over", [])],
        "coach_note": plan.get("coach_note", ""),
    }


# ======================================================================
# 长期计划
# ======================================================================
PHASES = [
    ("基础阶段", 0.00, 0.40, "系统过一遍全部考点，优先建立计算能力"),
    ("强化阶段", 0.40, 0.75, "主攻高频考点与综合题型，补齐薄弱环节"),
    ("冲刺阶段", 0.75, 1.00, "真题实战与错题回炉，只保高频高性价比内容"),
]

def generate_long_plan(
    repo: SQLiteRepository,
    user_id: str,
    role: str,
    persist: bool = True,
) -> dict[str, Any]:
    """按剩余天数把考点分配到三个阶段里程碑。

    L8 改动：新增 `persist` 开关，把「算」与「存」拆开。

    为什么必须拆：
        `interrupt` 会让**被中断的节点从头重跑**（官方语义：resumes from the
        start of the node, re-executing all logic）。如果"算"和"问确认"挤在
        同一个节点里，用户每确认一次，这张规划就被重算一次——纯浪费。
        拆开之后，恢复时只重跑"问"那个节点，"算"的结果安稳躺在 state 里。

    `persist=False` 时只返回计划、不落库——这正是「确认才落库」的前提：
        先算给你看，你点头了才写进 plan 表。

    默认 True 保持原有调用方（legacy 路径）行为不变。
    """
    pack = get_pack(role)
    ensure_kp_tree(repo, user_id, role, pack)
    ctx = build_context(repo, user_id, role, pack)

    today = _date.today()
    if ctx.exam_date and ctx.days_left and ctx.days_left > 0:
        exam = _parse_date(ctx.exam_date)
    else:
        exam = today + timedelta(days=180)  # 未设置考试日期时的兜底（当前产品暂不支持自由学习者）

    total_days = max((exam - today).days, 1)

    # 考点分配：基础阶段放高频且低难度的，强化阶段放高难度/高耗时的，冲刺阶段只做复习
    ranked = rank_kps(ctx, pack)
    by_difficulty = sorted(ranked, key=lambda x: (float(x.get("difficulty", 0.5)), -x["priority"]))

    n = len(by_difficulty)
    buckets = [
        by_difficulty[: int(n * 0.45)],
        by_difficulty[int(n * 0.45): int(n * 0.85)],
        [],
    ]
    # 冲刺阶段不再引入新考点，只回炉高频项
    buckets[2] = [
        t for t in by_difficulty[int(n * 0.85):]
    ] + [t for t in by_difficulty[: int(n * 0.1)]]

    milestones = []
    for (name, lo, hi, note), bucket in zip(PHASES, buckets):
        start = today + timedelta(days=int(total_days * lo))
        end = today + timedelta(days=max(int(total_days * hi) - 1, int(total_days * lo)))
        milestones.append(
            {
                "name": name,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "target_kps": [t["name"] for t in bucket],
                "note": note,
            }
        )

    coverage_note = (
        f"共 {n} 个考点待覆盖，剩余 {total_days} 天，"
        f"日均需消化 {n / total_days:.1f} 个考点"
        + (f"（考试日期 {ctx.exam_date}）" if ctx.exam_date else "（未设置考试日期，按 180 天估算）")
    )

    plan = {
        "role": pack.key,
        "exam_date": ctx.exam_date,
        "days_left": ctx.days_left,
        "milestones": milestones,
        "coverage_note": coverage_note,
    }

    if persist:
        save_long_plan(repo, user_id, role, plan, date=today.isoformat())
    return plan


def save_long_plan(
    repo: SQLiteRepository,
    user_id: str,
    role: str,
    plan: dict[str, Any],
    date: str | None = None,
    status: str = "active",
) -> None:
    """把长期规划落库（L8 从 generate_long_plan 里抽出来）。

    单独成函数的意义：让「什么时候写库」变成一个**显式的调用点**。
    在图里，这个调用点被放在 interrupt 之后——
    于是「确认才落库」不是靠注释保证的，而是靠代码结构保证的：
    不走到这一行，数据库里就没有这份规划。

    status 参数留给后续做「待确认」草稿态（plan 表已有该字段，无需改表）。
    """
    repo.save_plan(
        user_id,
        role,
        "long",
        date or _today(),
        plan,
        status=status,
    )


# ======================================================================
# 重规划（汇报驱动）
# ======================================================================
def replan_after_report(
    repo: SQLiteRepository,
    user_id: str,
    role: str,
    report_date: str,
    accuracy: dict[str, float],
    completed_kps: list[str],
    problems: list[str],
    planned_hours: float | None = None,
    actual_hours: float | None = None,
) -> dict[str, Any]:
    """汇报后的重规划。

    步骤：
        1. 按证据更新掌握度画像（指数平滑）
        2. 计算完成率偏差
        3. 偏差小 -> 不改计划，只写提醒；偏差大 -> 生成次日计划并顺延未完成项
    """
    pack = get_pack(role)

    # ---- 1. 更新掌握度，并记录变化 ----
    mastery_delta: dict[str, float] = {}
    for kp_name, acc in (accuracy or {}).items():
        before = repo.get_mastery(user_id, role, kp_name)
        after = repo.update_mastery(user_id, role, kp_name, acc)
        # 首次出现的知识点以先验值为基准计算变化，这样用户能看到"从 0.30 到 0.39"的过程
        baseline = PRIOR_MASTERY if before is None else before
        mastery_delta[kp_name] = round(after - baseline, 4)

    # ---- 2. 计算偏差 ----
    old_plan = repo.get_plan(user_id, role, "daily", report_date)
    planned_kps: list[str] = []
    if old_plan:
        planned_kps = [t.get("kp", "") for t in (old_plan.get("payload") or {}).get("tasks", [])]

    reason_parts: list[str] = []
    changed = False

    if planned_kps:
        done = len(set(planned_kps) & set(completed_kps or []))
        deviation = 1.0 - done / len(planned_kps)
        if actual_hours is not None and planned_hours:
            hours_dev = abs(actual_hours - planned_hours) / max(planned_hours, 0.1)
            deviation = max(deviation, min(hours_dev, 1.0))

        if deviation <= REPLAN_DEVIATION_THRESHOLD:
            reason_parts.append(
                f"计划完成率 {1 - deviation:.0%}，偏差在 {REPLAN_DEVIATION_THRESHOLD:.0%} 以内，"
                f"判定为正常波动，不调整任务顺序（避免计划来回震荡）。"
            )
        else:
            reason_parts.append(
                f"计划完成率 {1 - deviation:.0%}，偏差超过阈值，需要重新排布后续任务。"
            )
            changed = True
    else:
        reason_parts.append("今日无已发布计划，按最新掌握度直接生成次日安排。")
        changed = True

    # ---- 3. 生成次日计划（只改未来，不改今天）----
    next_plan = None
    if changed:
        next_date = (_parse_date(report_date) + timedelta(days=1)).isoformat()
        next_plan = generate_daily_plan(
            repo, user_id, role, date=next_date, force_refresh=True
        )
        reason_parts.append(f"已按最新掌握度生成 {next_date} 的清单，未完成项顺延至清单首位。")

    if problems:
        reason_parts.append("本次暴露的问题：" + "；".join(problems[:3]))

    return {
        "date": report_date,
        "mastery_delta": mastery_delta,
        "problems": problems or [],
        "replan_reason": " ".join(reason_parts),
        "changed": changed,
        "next_plan": next_plan,
    }
