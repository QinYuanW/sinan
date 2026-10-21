"""司南 · 每日早报（L13）

═══════════════════════════════════════════════════════════════════════════
早报回答三个问题
═══════════════════════════════════════════════════════════════════════════

    ① 昨天到底做了什么       —— 不是"打卡了"，而是"完成了哪几个点、正确率多少"
    ② 现在最该补什么         —— 薄弱点 + 昨日暴露的问题 + 昨日的漏项
    ③ 外面发生了什么         —— 政策 / 院校 / 备考方法，自动搜 + 总结

───────────────────────────────────────────────────────────────────────────
★ 本模块最重要的一条原则：**事实由算法给，措辞由模型润色**
───────────────────────────────────────────────────────────────────────────

    "计划 3 项完成 2 项（67%）"      → 必须从 daily_log + plan 算出来
    "你昨天表现不错，继续保持"       → 模型可以这么说，但**不能改事实**

为什么较真：
    如果把"昨天完成得怎么样"整个丢给模型去总结，它会写出
    "你昨天完成了极限部分的练习，效果不错"——而实际上用户只打卡没汇报。
    用户第二天看到这种早报，会立刻失去信任：**它在编我的生活。**

所以这里的结构是：
    _recap / _weak_points / _suggestions  —— 纯算法，零模型
    _news                                  —— 模型只负责"压缩 + 措辞"
    两者拼装时，事实段落和润色段落**物理分离**，谁也别越界。

───────────────────────────────────────────────────────────────────────────
为什么早报要落库（briefing 表）
───────────────────────────────────────────────────────────────────────────

    ① 资讯要调 Tavily + 模型，每天只该搜一次（否则刷新一次花一次钱）
    ② 早报是"当天已发出的事实"，事后回看不能被重算覆盖
    ③ 前端每次刷新都现算会让页面卡 3~5 秒

    → 所以：**首次生成时落库，之后读库；显式 `force=True` 才重算。**
"""

from __future__ import annotations

import asyncio
import json
from datetime import date as _date, timedelta
from typing import Any

from app.common.logger import logger

# 早报里的资讯条数（多了没人看，少了不够）
NEWS_TOP_K = 4


# ======================================================================
# ① 昨日复盘（纯算法）
# ======================================================================
def _recap(repo, user_id: str, role: str, yesterday: str) -> dict[str, Any]:
    """把昨天的事实摊开：打卡了没、计划几项、完成几项、正确率、暴露了什么问题。

    ⚠️ 这里每个数字都能被追溯到某张表的某个字段。
       不许出现"大概""估计""看起来"——那是模型该干的活，不是算法的。
    """
    log = repo.get_daily_log(user_id, role, yesterday)
    plan = repo.get_plan(user_id, role, "daily", yesterday)

    parsed = (log or {}).get("report_parsed") or {}
    planned = [
        t.get("kp", "")
        for t in ((plan or {}).get("payload") or {}).get("tasks", [])
        if t.get("kp")
    ]
    completed = list(parsed.get("completed_kps") or [])

    done_in_plan = [k for k in planned if k in set(completed)]
    missed = [k for k in planned if k not in set(completed)]
    rate = (len(done_in_plan) / len(planned)) if planned else None

    return {
        "date": yesterday,
        "checked_in": bool((log or {}).get("checked_in_at")),
        "has_report": bool(parsed),
        "planned_count": len(planned),
        "planned": planned,
        "completed": completed,
        "completed_in_plan": done_in_plan,
        "missed": missed,
        "completion_rate": round(rate, 2) if rate is not None else None,
        "accuracy": parsed.get("accuracy") or {},
        "problems": list(parsed.get("problems") or []),
        "mastery_delta": (log or {}).get("mastery_delta") or {},
        "actual_hours": (log or {}).get("actual_hours"),
    }


# ======================================================================
# ② 薄弱点（纯算法）
# ======================================================================
def _weak_points(
    repo, user_id: str, role: str, recap: dict[str, Any], limit: int = 5
) -> list[dict[str, Any]]:
    """合并三个来源，按优先级排：

        1. 昨日暴露的问题   —— 最新鲜，最有行动价值
        2. 昨日没做完的     —— 已经欠着的
        3. 掌握度最低的     —— 长期薄弱的

    为什么"昨日问题"排第一：它是**刚发生的**，此刻修复成本最低。
    掌握度低的点可能已经低了两周，今天再补也不差这一天。
    """
    seen: dict[str, dict[str, Any]] = {}

    def add(kp: str, reason: str, source: str, mastery: float | None = None) -> None:
        kp = (kp or "").strip()
        if not kp or kp in seen:
            return
        seen[kp] = {
            "kp": kp,
            "reason": reason,
            "source": source,
            "mastery": mastery,
        }

    # 1. 昨日暴露的问题（problems 是自由文本，尽量提取考点名）
    for p in recap.get("problems", []):
        text = str(p).strip()
        if not text:
            continue
        # problems 通常是"泰勒展开：阶数选取方法不会"这种"考点：问题"格式
        kp = text.split("：")[0].split(":")[0].strip()
        add(kp or text, f"昨日暴露：{text}", "problem")

    # 2. 昨日没做完的
    for k in recap.get("missed", []):
        add(k, "昨日计划内未完成", "carry_over")

    # 3. 掌握度最低的
    summary = repo.mastery_summary(user_id, role)
    for it in summary.get("weakest", []):
        add(
            it.get("kp_name", ""),
            f"掌握度 {it.get('mastery', 0):.2f}（最低档）",
            "mastery",
            it.get("mastery"),
        )

    return list(seen.values())[:limit]


# ======================================================================
# ③ 今日建议（纯算法）
# ======================================================================
def _suggestions(recap: dict[str, Any], weak: list[dict[str, Any]], streak: int) -> list[str]:
    """基于规则的行动建议。

    ★ 为什么不用模型写建议：
        建议的价值在于**能执行**。"今天先补泰勒展开的阶数判断"是可执行的；
        "建议加强薄弱环节"是废话。规则能给出前者，模型常常给后者。
        模型擅长的是措辞，不是决策——这里只需要决策。
    """
    out: list[str] = []
    rate = recap.get("completion_rate")

    if not recap["checked_in"] and not recap["has_report"]:
        out.append("昨天没有打卡也没有汇报。今天先花 10 分钟把昨天做的事补记一下——"
                   "**画像断一天，后面的排序就会失真**。")
    elif rate is not None and rate < 0.5:
        out.append(
            f"昨天计划 {recap['planned_count']} 项只完成 {len(recap['completed_in_plan'])} 项"
            f"（{rate:.0%}）。今天**先减量**：把昨天没做的补完，再考虑新内容，"
            f"不要一边欠账一边加新任务。"
        )
    elif rate is not None and rate >= 0.9:
        out.append(
            f"昨天完成率 {rate:.0%}，节奏很稳。今天可以**适度加量**，"
            f"或者把时间投到更难的考点上。"
        )

    if recap.get("problems"):
        out.append(
            "昨天暴露的问题今天优先处理：" +
            "；".join(str(p)[:40] for p in recap["problems"][:3]) +
            "。**同类问题连错三次，就是方法错了，不是粗心。**"
        )

    if streak >= 3:
        out.append(f"已连续打卡 {streak} 天，保持住。连续性是这套系统里最值钱的变量。")

    if weak and not out:
        out.append(f"当前最该补的是「{weak[0]['kp']}」（{weak[0]['reason']}）。")

    if not out:
        out.append("数据还太少，今天先正常推进。做几道题、汇报一次，明天的早报就会有内容。")

    return out


# ======================================================================
# ④ 今日清单（纯读库 + 幂等生成）
# ======================================================================
def _today_tasks(repo, user_id: str, role: str, today: str) -> list[dict[str, Any]]:
    """取今日清单的前几项，让早报能直接回答"现在该干什么"。

    ★ 为什么早报必须带这个：
        只告诉用户"你哪里弱"，等于把决策又推回给他——
        他还得打开网页、看清单、再决定先做哪个。
        **早报的价值在于"看完就能动手"，不是"看完知道哪里差"。**

    生成策略：今天的计划还没排过，就顺手排一份（`generate_daily_plan` 本身幂等，
    不会因为早报和前端各调一次就产生两份不一样的清单）。
    """
    try:
        from app.agent.core.planner import generate_daily_plan

        # ⚠️ 两个函数返回的**形状不一样**（踩过的坑）：
        #     repo.get_plan()          -> {"payload": {"tasks": [...]}}   ← 带包装
        #     generate_daily_plan()    -> {"tasks": [...]}                ← 就是 payload 本身
        #   按包装去取后者会得到空列表，且**不报错**——
        #   表现是"老用户能看到清单，新用户第一次永远看不到"。
        existing = repo.get_plan(user_id, role, "daily", today)
        if existing:
            tasks = ((existing.get("payload") or {}).get("tasks")) or []
        else:
            generated = generate_daily_plan(repo, user_id, role, date=today) or {}
            tasks = generated.get("tasks") or []

        return [
            {
                "kp": t.get("kp", ""),
                "hours": t.get("planned_hours") or t.get("est_hours"),
                "priority": t.get("priority"),
                "reason": (t.get("reason") or "")[:80],
            }
            for t in tasks[:3]
        ]
    except Exception as exc:  # noqa: BLE001 — 清单生成失败不该拖垮早报
        logger.warning(f"[司南·briefing] 今日清单获取失败: {type(exc).__name__}: {exc}")
        return []


# ======================================================================
# ⑤ 资讯（模型只做压缩）
# ======================================================================
def _fetch_news(role: str) -> list[dict[str, str]]:
    """搜 + 总结当日资讯。

    ★ 两级降级（外部服务不可用不该让早报整体失败）：
        Tavily 挂了   → news 返回空，早报其余部分照常
        模型挂了      → 直接把搜索结果的标题/摘要截断返回，不做二次压缩
    """
    queries = [
        "2027 考研 数学二 考试大纲 政策 最新变化",
        "考研数学二 备考方法 高频考点 复习建议",
    ]
    raw: list[str] = []
    try:
        from app.agent.tools.exam_search import search_exam_info

        for q in queries:
            try:
                text = search_exam_info.invoke({"query": q})
                if text:
                    raw.append(str(text)[:2500])
            except Exception as exc:  # noqa: BLE001 — 单条查询失败不影响另一条
                logger.warning(f"[司南·briefing] 资讯查询失败 '{q[:20]}': {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[司南·briefing] 搜索工具不可用: {exc}")

    if not raw:
        return []

    material = "\n\n---\n\n".join(raw)

    # 用模型压缩成条目；失败则退回"原始摘要截断"
    try:
        from langchain.chat_models import init_chat_model

        model = init_chat_model(model="deepseek-chat")
        prompt = (
            "下面是我从网上搜到的考研资讯材料。请压缩成 3~4 条要点，"
            "每条一句话，必须包含：**发生了什么 + 对考生意味着什么**。"
            "只保留与考研数学二备考真正相关的信息；"
            "如果材料里全是营销广告或没有新信息，就只回一句「今天没有值得关注的更新」。"
            "不要编造材料里没有的内容。输出用「- 」开头的列表，不要标题。\n\n"
            f"{material}"
        )
        result = model.invoke(prompt)
        text = str(getattr(result, "content", result)).strip()
        items: list[dict[str, str]] = []
        for line in text.splitlines():
            line = line.strip().lstrip("-•· ").strip()
            if line and len(line) > 6:
                items.append({"title": line[:120], "source": "search+summary"})
        return items[:NEWS_TOP_K]
    except Exception as exc:  # noqa: BLE001 — 模型不可用就退回原文
        logger.warning(f"[司南·briefing] 资讯总结失败，退回原始摘要: {exc}")
        return [{"title": material[:400], "source": "search_raw"}]


# ======================================================================
# ⑤ 生成入口
# ======================================================================
def build_briefing(
    repo,
    user_id: str,
    role: str,
    date: str | None = None,
    with_news: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """生成（或读取）某天的早报。

    默认幂等：当天已生成过就直接返回，不重算——
    和 `generate_daily_plan` 的幂等是同一个思路：
    **"今天已经发出去的东西"不该因为刷新而变。**
    """
    today = date or _date.today().isoformat()
    yesterday = (_date.fromisoformat(today) - timedelta(days=1)).isoformat()

    if not force:
        cached = repo.get_briefing(user_id, role, today)
        if cached and cached.get("payload"):
            payload = dict(cached["payload"])
            payload["cached"] = True
            return payload

    recap = _recap(repo, user_id, role, yesterday)
    weak = _weak_points(repo, user_id, role, recap)
    streak = repo.check_in_streak(user_id, role)
    suggestions = _suggestions(recap, weak, streak)
    today_tasks = _today_tasks(repo, user_id, role, today)
    news = _fetch_news(role) if with_news else []

    briefing = {
        "date": today,
        "yesterday": yesterday,
        "role": role,
        "recap": recap,
        "weak_points": weak,
        "suggestions": suggestions,
        "today_tasks": today_tasks,
        "news": news,
        "streak": streak,
        "headline": _headline(recap, weak, streak),
        "cached": False,
    }
    briefing["markdown"] = render_briefing(briefing)

    repo.save_briefing(user_id, role, today, briefing)
    logger.info(
        f"[司南·briefing] 已生成 user={user_id} role={role} date={today} "
        f"完成率={recap.get('completion_rate')} 薄弱点={len(weak)} 资讯={len(news)}"
    )
    return briefing


def _headline(recap: dict[str, Any], weak: list[dict[str, Any]], streak: int) -> str:
    """一句话标题 —— 早报的第一眼。"""
    if not recap["checked_in"] and not recap["has_report"]:
        return "昨天没有记录，今天先补一笔。"
    rate = recap.get("completion_rate")
    if rate is None:
        return f"昨天打卡了，但没汇报细节。连续 {streak} 天。"
    if rate >= 0.9:
        return f"昨天完成率 {rate:.0%}，很稳。今天可以往前推一点。"
    if rate >= 0.5:
        return f"昨天完成 {rate:.0%}，今天先把剩下那部分收尾。"
    return f"昨天只完成 {rate:.0%}，今天先减量、补缺口。"


# ======================================================================
# ⑥ 渲染
# ======================================================================
def render_briefing(b: dict[str, Any]) -> str:
    """渲染成 Markdown（给前端/邮件/终端共用）。"""
    r = b["recap"]
    lines = [f"# 早报 · {b['date']}", "", f"**{b['headline']}**", ""]

    # ---- 昨天 ----
    lines.append(f"## 昨天（{b['yesterday']}）")
    if not r["checked_in"] and not r["has_report"]:
        lines.append("- 没有打卡，也没有汇报。")
    else:
        lines.append(f"- 打卡：{'✅ 已打卡' if r['checked_in'] else '❌ 未打卡'}")
        if r["planned_count"]:
            rate = r["completion_rate"]
            lines.append(
                f"- 计划 {r['planned_count']} 项，完成 {len(r['completed_in_plan'])} 项"
                + (f"（{rate:.0%}）" if rate is not None else "")
            )
            if r["missed"]:
                lines.append(f"- 未完成：{'、'.join(r['missed'][:5])}")
        if r["accuracy"]:
            acc = "、".join(f"{k} {v:.0%}" for k, v in list(r["accuracy"].items())[:4])
            lines.append(f"- 作答证据：{acc}")
        if r["mastery_delta"]:
            delta = "、".join(f"{k} {v:+.2f}" for k, v in list(r["mastery_delta"].items())[:4])
            lines.append(f"- 画像变化：{delta}")
        if r["problems"]:
            lines.append("- 暴露的问题：")
            lines.extend(f"  - {p}" for p in r["problems"][:4])

    # ---- 薄弱点 ----
    if b["weak_points"]:
        lines += ["", "## 今天最该补的"]
        for i, w in enumerate(b["weak_points"], 1):
            lines.append(f"{i}. **{w['kp']}** — {w['reason']}")

    # ---- 建议 ----
    if b["suggestions"]:
        lines += ["", "## 今天怎么走"]
        lines.extend(f"- {s}" for s in b["suggestions"])

    # ---- 今日清单（看完就能动手）----
    if b.get("today_tasks"):
        lines += ["", "## 今天的清单（已排好）"]
        for i, t in enumerate(b["today_tasks"], 1):
            hours = f" · 约 {t['hours']}h" if t.get("hours") else ""
            prio = f" · 优先度 {t['priority']:.2f}" if t.get("priority") is not None else ""
            lines.append(f"{i}. **{t['kp']}**{hours}{prio}")
            if t.get("reason"):
                lines.append(f"   {t['reason']}")
        lines.append("")
        lines.append("_完整清单在网页「今日作战」面板里；做完记得 `@汇报`。_")

    # ---- 资讯 ----
    lines += ["", "## 外面发生了什么"]
    if b["news"]:
        lines.extend(f"- {n['title']}" for n in b["news"])
    else:
        lines.append("- 今天没有抓到值得关注的更新。")

    lines += ["", "---", f"连续打卡 {b['streak']} 天"]
    return "\n".join(lines)


def describe_briefing() -> dict[str, Any]:
    return {
        "sections": ["recap", "weak_points", "suggestions", "news"],
        "fact_source": "recap/weak/suggestions 纯算法；news 模型压缩",
        "news_queries": 2,
        "cache": "briefing 表，按 (user, role, date) 唯一",
    }


__all__ = [
    "build_briefing",
    "render_briefing",
    "describe_briefing",
    "NEWS_TOP_K",
]
