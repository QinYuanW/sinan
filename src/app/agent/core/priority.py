"""司南 · 优先度引擎（与身份无关）

核心公式：

    priority = Σ  wᵢ · fᵢ(ctx, kp)

其中因子集合 fᵢ 与权重 wᵢ 全部由身份包提供，本模块不认识任何具体身份。
权重允许为负（表达"成本惩罚"这类负向因子）。

考研身份下的展开形式：
    priority = 0.40·考频 + 0.35·掌握度缺口 + 0.15·遗忘紧急度 - 0.10·时间成本
且当剩余天数 ≤ 60 时由身份包的 adjust_weights 钩子动态调权。
"""

from __future__ import annotations

from typing import Any

from app.agent.roles.base import PlanningContext, RolePack

# 掌握度高于该阈值视为已过关，不再进入每日清单
MASTERED_THRESHOLD = 0.85
# 优先级低于该值的知识点不值得占用当日时间
MIN_PRIORITY = 0.15

def score_kp(
    ctx: PlanningContext,
    kp: dict[str, Any],
    pack: RolePack,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """计算单个知识点的原始得分与因子明细。

    Returns:
        (raw_score, {factor_key: factor_value})
    """
    w = weights if weights is not None else pack.adjust_weights(ctx)
    detail: dict[str, float] = {}
    total = 0.0
    for factor in pack.factors:
        value = factor.compute(ctx, kp)
        detail[factor.key] = round(value, 4)
        total += w.get(factor.key, 0.0) * value
    return round(total, 4), detail


def rank_kps(
    ctx: PlanningContext,
    pack: RolePack,
    limit: int | None = None,
    kp_pool: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """给全部知识点打分并排序。

    返回的每一项在原始 kp 基础上附加：
        raw_score / priority / factors / reason / acceptance
    其中 priority 已归一化到 0-1，便于跨身份横向比较与展示。
    """
    weights = pack.adjust_weights(ctx)
    pool = kp_pool if kp_pool is not None else ctx.kp_list

    scored: list[dict[str, Any]] = []
    for kp in pool:
        mastery = ctx.mastery_of(kp["name"])
        if mastery >= MASTERED_THRESHOLD:
            continue  # 已过关，不占用时间
        raw, detail = score_kp(ctx, kp, pack, weights)
        scored.append(
            {
                **kp,
                "mastery": round(mastery, 4),
                "raw_score": raw,
                "factors": detail,
                "weights": weights,
            }
        )

    if not scored:
        return []

    # 归一化到 0-1（保留相对次序，便于展示与跨身份比较）
    scores = [item["raw_score"] for item in scored]
    lo, hi = min(scores), max(scores)
    for item in scored:
        if hi - lo < 1e-9:
            item["priority"] = 0.5
        else:
            item["priority"] = round((item["raw_score"] - lo) / (hi - lo), 4)
        item["reason"] = explain(item, ctx)
        item["acceptance"] = acceptance_for(item)

    scored.sort(key=lambda x: x["priority"], reverse=True)

    if limit is not None:
        scored = scored[:limit]
    return scored

def explain(item: dict[str, Any], ctx: PlanningContext) -> str:
    """把因子明细翻译成人话。推荐理由必须可解释，这是本项目的核心主张。"""
    parts: list[str] = []
    factors = item.get("factors", {})

    freq = factors.get("exam_freq")
    if freq is not None:
        if freq >= 0.8:
            level = "高频考点"
        elif freq >= 0.6:
            level = "常考考点"
        else:
            level = "低频考点"
        parts.append(f"考频 {freq:.2f}（{level}）")

    gap = factors.get("mastery_gap")
    if gap is not None:
        parts.append(f"掌握度 {item['mastery']:.2f}，缺口 {gap:.2f}")

    days = ctx.days_since_review(item["name"])
    if days is None:
        parts.append("尚未开始学习")
    elif days >= 14:
        parts.append(f"已 {days} 天未复习，遗忘风险高")
    elif days >= 7:
        parts.append(f"距上次复习 {days} 天")

    if "student_signal" in factors:  # 教师身份
        signal = factors["student_signal"]
        if signal >= 0.7:
            parts.append(f"学情反馈偏差明显（{signal:.2f}）")

    if "schedule_urgency" in factors:  # 教师身份
        urg = factors["schedule_urgency"]
        if urg >= 0.8:
            parts.append("教学进度临近，需优先安排")

    reason = "；".join(parts) if parts else "按当前整体优先级排序"
    return f"{reason}。预计投入 {float(item.get('est_hours', 2.0)):.1f} 小时。"

def acceptance_for(item: dict[str, Any]) -> str:
    """验收标准：让"今天学会了"变成一个可判断的事实，而不是感觉。"""
    difficulty = float(item.get("difficulty", 0.5))
    if difficulty >= 0.75:
        return "能独立完成 3 道同类真题（含 1 道综合题），并说清每一步的依据；错题当场归因。"
    if difficulty >= 0.5:
        return "能独立完成 5 道同类真题且全对，并能默写关键公式与适用条件。"
    return "能独立完成 5 道基础题且全对，概念题不再出错。"

def build_daily_plan(
    ctx: PlanningContext,
    pack: RolePack,
    date: str,
    carry_over: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """按每日可用时长把排序结果装箱成当日清单。

    装箱策略：按优先级降序装入，直到预算耗尽。允许**部分推进**——
    一个考点完整掌握可能要 6-8 小时，而用户每天只有 4 小时，
    这时标记 partial=True 并给出"今日推进多少小时"，而不是
    要么整天只放一条、要么硬塞到超时长。
    """
    budget = float(ctx.daily_hours or pack.default_daily_hours)
    ranked = rank_kps(ctx, pack)

    tasks: list[dict[str, Any]] = []
    used = 0.0
    for item in ranked:
        if item["priority"] < MIN_PRIORITY:
            continue

        remaining = budget - used
        if remaining <= 0.25:
            break

        full_hours = float(item.get("est_hours", 2.0))
        entry = dict(item)

        if full_hours <= remaining:
            entry["planned_hours"] = round(full_hours, 2)
            entry["partial"] = False
            used += full_hours
        else:
            entry["planned_hours"] = round(remaining, 2)
            entry["partial"] = True
            entry["acceptance"] = (
                f"今日目标：推进该知识点的前 {remaining:.1f} 小时内容"
                f"（完整掌握预计 {full_hours:.1f} 小时），"
                f"结束时能独立做出该部分的基础题。"
            )
            used += remaining

        tasks.append(entry)
        if used >= budget - 1e-9:
            break

    return {
        "date": date,
        "role": pack.key,
        "total_hours": round(used, 2),
        "budget_hours": round(budget, 2),
        "tasks": tasks,
        "carry_over": carry_over or [],
        "coach_note": coach_note(ctx, tasks, budget),
    }


def coach_note(
    ctx: PlanningContext, tasks: list[dict[str, Any]], budget: float
) -> str:
    """节奏提醒：只说风险与事实，不说废话鼓励。"""
    notes: list[str] = []
    used = sum(float(t.get("est_hours", 2.0)) for t in tasks)

    if ctx.days_left is not None:
        total_left_hours = ctx.days_left * budget
        total_need = sum(
            float(k.get("est_hours", 2.0)) * (1 - ctx.mastery_of(k["name"]))
            for k in ctx.kp_list
        )
        if total_left_hours > 0:
            ratio = total_need / total_left_hours
            if ratio > 1.2:
                notes.append(
                    f"按当前进度，剩余 {ctx.days_left} 天的可用时长约为 {total_left_hours:.0f} 小时，"
                    f"而待补内容预计需要 {total_need:.0f} 小时，缺口约 {ratio:.1f} 倍。"
                    f"建议优先保证高频考点，低频考点只做真题级别的最低掌握。"
                )
            elif ratio < 0.7:
                notes.append(
                    f"剩余时间相对充裕（待补 {total_need:.0f} 小时 / 可用 {total_left_hours:.0f} 小时），"
                    f"可以在高频考点上追加综合题训练。"
                )

    if used < budget * 0.6 and tasks:
        notes.append(f"今日清单只排了 {used:.1f} 小时，低于可用 {budget:.1f} 小时，可考虑追加低频考点。")

    if not tasks:
        notes.append("当前没有需要安排的知识点，可能是掌握度均已达标，请补充考点或调整目标。")

    if ctx.streak >= 3:
        notes.append(f"已连续打卡 {ctx.streak} 天。")

    return " ".join(notes)
