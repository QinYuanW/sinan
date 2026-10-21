"""司南 · 周报 / 月报（L16）

早报回答"昨天怎么样"，周报回答"这一段怎么样"。
两者的数据底座是同一个（daily_log + plan + mastery），但视角完全不同：

    早报：昨天的事 → 今天怎么调
    周报：七天的趋势 → 方向对不对

═══════════════════════════════════════════════════════════════════════════
★ 一个绕不开的坑：掌握度没有历史快照
═══════════════════════════════════════════════════════════════════════════

mastery 表只存**当前值**（指数平滑后的结果），没有"上周三那天是多少"。

    ❌ 不能直接查历史 —— 数据不存在
    ❌ 也不能从当前值反解 —— 指数平滑是单向的，`new = old + α(acc − old)` 不可逆

本模块的解法：**用每日的 mastery_delta 累加，倒推周期初的值**

    周期初均值 ≈ 当前均值 − Σ(周期内所有 delta)
    第 k 天的值  = 周期初均值 + Σ(前 k 天 delta)

这是个**近似**，但足够支撑一条趋势线，因为 delta 本身就是每次更新的真实增量。

【作业】想要精确曲线，加一张 `mastery_snapshot(user, role, date, kp, mastery)` 表，
在每次 `update_mastery` 后写一条。代价是写放大（69 考点 × 每天 = 69 行/天）。
**先问自己：真的需要精确到小数点后四位的历史吗？趋势看的是方向，不是精度。**
"""

from __future__ import annotations

from datetime import date as _date, timedelta
from typing import Any

from app.common.logger import logger

# 周期长度
PERIOD_DAYS = {"week": 7, "month": 30}

# 一周里打卡低于这个比例，判定为"节奏断了"
LOW_ADHERENCE = 0.5


def _parse(d: str) -> _date:
    return _date.fromisoformat(d)


def build_period_report(
    repo,
    user_id: str,
    role: str,
    kind: str = "week",
    anchor: str | None = None,
) -> dict[str, Any]:
    """生成周报 / 月报。

    kind: "week"（近 7 天，含今天）| "month"（近 30 天）
    anchor: 结束日期，默认今天
    """
    if kind not in PERIOD_DAYS:
        raise ValueError(f"未知周期 '{kind}'，可选: week / month")

    end = _parse(anchor) if anchor else _date.today()
    span = PERIOD_DAYS[kind]
    start = end - timedelta(days=span - 1)
    days = [(start + timedelta(days=i)).isoformat() for i in range(span)]

    # ---------------- 逐日扫描 ----------------
    series: list[dict[str, Any]] = []
    total_planned = 0
    total_done = 0
    checked_days = 0
    reported_days = 0
    delta_acc = 0.0
    per_kp_delta: dict[str, float] = {}

    for d in days:
        log = repo.get_daily_log(user_id, role, d)
        plan = repo.get_plan(user_id, role, "daily", d)

        checked = bool((log or {}).get("checked_in_at"))
        parsed = (log or {}).get("report_parsed") or {}
        tasks = ((plan or {}).get("payload") or {}).get("tasks") or []

        planned = len(tasks)
        done = sum(1 for t in tasks if (t.get("kp") in set(parsed.get("completed_kps") or [])))
        rate = (done / planned) if planned else None

        delta = (log or {}).get("mastery_delta") or {}
        day_sum = sum(float(v) for v in delta.values() if isinstance(v, (int, float)))
        for k, v in delta.items():
            if isinstance(v, (int, float)):
                per_kp_delta[k] = per_kp_delta.get(k, 0.0) + float(v)
        delta_acc += day_sum

        if checked:
            checked_days += 1
        if parsed:
            reported_days += 1
        total_planned += planned
        total_done += done

        series.append({
            "date": d,
            "checked_in": checked,
            "has_report": bool(parsed),
            "planned": planned,
            "done": done,
            "completion_rate": round(rate, 3) if rate is not None else None,
            # 累计净增量（供前端画掌握度趋势）
            "mastery_delta_cum": round(delta_acc, 4),
        })

    # ---------------- 掌握度趋势 ----------------
    # ⚠️ 这里踩过一次坑，值得记：
    #    最初写的是 `baseline = current_avg - Σdelta`，结果算出 **负数**。
    #    因为 Σdelta 是"5 个考点各 +0.18 = 0.92"，而 current_avg 是**均值**。
    #    量纲不一致 —— 金额减金额、比例减比例，不能混。
    #
    #    正确做法：不反推绝对均值，而是画**相对周期初的累计涨幅（按考点平均）**：
    #        gain_cum(t) = Σ(截至 t 的所有 delta) / 涉及考点数
    #    这样量纲一致（都是"每个考点涨了多少"），且永远不会有负数。
    #    代价是画不出"掌握度绝对值曲线"——但那本来就需要历史快照，
    #    用 delta 反推的绝对值从一开始就是假的。
    n_kp = max(len(per_kp_delta), 1)
    current_summary = repo.mastery_summary(user_id, role)
    current_avg = current_summary.get("average")

    for row in series:
        row["mastery_gain_avg"] = round(row["mastery_delta_cum"] / n_kp, 4)

    # ---------------- 汇总 ----------------
    adherence = round(checked_days / span, 3)
    overall_rate = round(total_done / total_planned, 3) if total_planned else None

    report = {
        "kind": kind,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "span": span,
        "series": series,
        "summary": {
            "checked_days": checked_days,
            "reported_days": reported_days,
            "adherence": adherence,
            "planned_total": total_planned,
            "done_total": total_done,
            "completion_rate": overall_rate,
            "mastery_avg_now": current_avg,
            # 周期内的净值：total 是所有考点增量之和，avg 是"平均每个考点涨了多少"
            # （后者才是能和"均值"放在一起看的数字）
            "mastery_gain_total": round(delta_acc, 4),
            "mastery_gain_avg": round(delta_acc / n_kp, 4),
            "kps_touched": len(per_kp_delta),
            "top_gain_kps": _top_delta(per_kp_delta, reverse=True),
            "stalled_kps": _top_delta(per_kp_delta, reverse=False),
            "weakest_now": [
                {"kp": w.get("kp_name"), "mastery": round(float(w.get("mastery", 0)), 3)}
                for w in current_summary.get("weakest", [])[:3]
            ],
        },
        "verdict": "",
        "markdown": "",
    }
    report["verdict"] = _verdict(report)
    report["markdown"] = render_period_report(report)
    logger.info(
        f"[司南·period] {kind} 报告 user={user_id} "
        f"打卡 {checked_days}/{span} 完成率 {overall_rate}"
    )
    return report


def _top_delta(per_kp: dict[str, float], reverse: bool, n: int = 3) -> list[dict[str, Any]]:
    items = [(k, round(v, 4)) for k, v in per_kp.items()]
    items.sort(key=lambda x: x[1], reverse=reverse)
    if reverse:
        items = [x for x in items if x[1] > 0][:n]
    else:
        # "停滞" = 增量为 0 或负的
        items = [x for x in items if x[1] <= 0][:n]
    return [{"kp": k, "delta": v} for k, v in items]


def _verdict(r: dict[str, Any]) -> str:
    """一句话结论 —— 周报最该有的东西。

    ★ 为什么这句话必须由规则生成：
        周报的价值是**判断方向**，不是罗列数据。
        "你打卡 5/7 天，完成率 68%" 是数据；
        "节奏稳住了，但完成率在掉，说明任务排多了" 才是判断。
        规则能给出判断，模型常常给出正确的废话。**判断归算法，润色归模型。**
    """
    s = r["summary"]
    if s["checked_days"] == 0:
        return "这一段没有记录。先把打卡恢复起来，数据才有意义。"

    parts: list[str] = []
    parts.append(f"打卡 {s['checked_days']}/{r['span']} 天")

    if s["completion_rate"] is not None:
        rate = s["completion_rate"]
        if rate >= 0.85:
            parts.append(f"完成率 {rate:.0%}，任务量排得合适")
        elif rate >= 0.6:
            parts.append(f"完成率 {rate:.0%}，略有欠账")
        else:
            parts.append(f"完成率只有 {rate:.0%}，**任务排多了**——建议把每日时长调低")

    gain = s["mastery_gain_avg"]
    if gain > 0.12:
        parts.append(f"画像平均涨 {gain:+.2f}，推进明显")
    elif gain > 0.03:
        parts.append(f"画像平均涨 {gain:+.2f}，稳步前进")
    elif gain > -0.03:
        parts.append("画像基本原地踏步")
    else:
        parts.append(f"画像平均 **{gain:+.2f}，在下滑**——是不是复习间隔拉太长了？")

    if s["adherence"] < LOW_ADHERENCE:
        parts.append("**打卡率偏低，先解决'每天都在'比'每天做多少'更重要**")

    return "；".join(parts) + "。"


def render_period_report(r: dict[str, Any]) -> str:
    s = r["summary"]
    label = "周报" if r["kind"] == "week" else "月报"
    lines = [
        f"# {label} · {r['start']} ~ {r['end']}",
        "",
        f"**{r['verdict']}**",
        "",
        "## 关键数字",
        f"- 打卡：{s['checked_days']}/{r['span']} 天（{s['adherence']:.0%}）",
        f"- 汇报：{s['reported_days']} 天",
    ]
    if s["completion_rate"] is not None:
        lines.append(
            f"- 完成任务：{s['done_total']}/{s['planned_total']}（{s['completion_rate']:.0%}）"
        )
    if s["mastery_avg_now"] is not None:
        lines.append(
            f"- 掌握度均值：{s['mastery_avg_now']}"
            f"（本{'周' if r['kind'] == 'week' else '月'}平均每个考点 "
            f"{s['mastery_gain_avg']:+.2f}，共涉及 {s['kps_touched']} 个考点）"
        )

    if s["top_gain_kps"]:
        lines += ["", "## 进步最大的"]
        lines.extend(f"- **{i['kp']}** {i['delta']:+.2f}" for i in s["top_gain_kps"])

    if s["stalled_kps"]:
        lines += ["", "## 停滞不前的"]
        lines.extend(f"- **{i['kp']}** {i['delta']:+.2f}" for i in s["stalled_kps"])

    if s["weakest_now"]:
        lines += ["", "## 当前最薄弱"]
        lines.extend(f"- {w['kp']}（{w['mastery']}）" for w in s["weakest_now"])

    # ---- 每日明细（表格）----
    lines += ["", "## 每日明细", "", "| 日期 | 打卡 | 计划 | 完成 | 完成率 |", "|---|---|---|---|---|"]
    for row in r["series"]:
        rate = f"{row['completion_rate']:.0%}" if row["completion_rate"] is not None else "—"
        lines.append(
            f"| {row['date'][5:]} | {'✅' if row['checked_in'] else '·'} | "
            f"{row['planned']} | {row['done']} | {rate} |"
        )

    lines += ["", "---", "_掌握度曲线由每日增量反推，是估算值；趋势方向可信，小数点不可信。_"]
    return "\n".join(lines)


__all__ = ["build_period_report", "render_period_report", "PERIOD_DAYS"]
