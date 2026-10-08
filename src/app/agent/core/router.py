"""司南 · 意图路由

判断用户这句话到底想干什么。路由结果决定调用哪条业务链路。

v1 用规则优先（零成本、可预测、可测试），不认识的才落到闲聊/自主决策。
规则法的好处是行为完全可解释——答辩时可以逐条讲清判定依据。

为什么需要这层：
    前端 `@` 提及组件受限于 static 目录只有构建产物、没有源码，
    因此 v1 把「@汇报助手」的语义放在后端识别，
    前端补上组件后，这里可以无缝保留为兜底路由。
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    PROFILE = "profile"        # 设置目标（院校/考试日期/时长）
    PLAN_LONG = "plan_long"    # 要整体规划
    PLAN_DAILY = "plan_daily"  # 要今天的计划
    CHECKIN = "checkin"        # 打卡
    REPORT = "report"          # 学习汇报
    CHAT = "chat"              # 其他，交给主 Agent 自主决策


# 显式前缀（前端补上 @ 组件后主要靠这些）
_PREFIX_MAP: list[tuple[tuple[str, ...], Intent]] = [
    (("@汇报", "@report", "/report", "汇报助手"), Intent.REPORT),
    (("@打卡", "@checkin", "/checkin"), Intent.CHECKIN),
    (("@规划", "@plan", "@整体规划"), Intent.PLAN_LONG),
    (("@日报", "@today", "@今日计划"), Intent.PLAN_DAILY),
    (("@目标", "@profile", "@设置目标"), Intent.PROFILE),
]

# 自然语言关键词（无前缀时的兜底）
# 注意顺序：先匹配更具体的意图，避免"整体规划"被"计划"抢走
_KEYWORD_MAP: list[tuple[tuple[str, ...], Intent]] = [
    (("我的计划", "今天做什么", "今日任务", "今天学什么", "今日清单", "日报",
      "今天的计划", "今日计划", "生成计划", "安排今天", "今天安排", "安排一下今天",
      "今天怎么学", "今日安排"), Intent.PLAN_DAILY),
    (("整体规划", "长线规划", "长期规划", "阶段规划", "全程规划", "规划一下",
      "做个规划", "总体规划", "备考规划"), Intent.PLAN_LONG),
    (("打卡",), Intent.CHECKIN),
    (("我今天", "今日完成", "汇报", "做完", "刷了", "做了", "错了几道", "正确率",
      "完成了", "学了"), Intent.REPORT),
    (("考试日期", "目标院校", "我要考", "每天能学", "每日时长", "设定目标", "设置目标"), Intent.PROFILE),
]


def route(message: str) -> tuple[Intent, str]:
    """返回 (意图, 去掉前缀后的正文)"""
    text = (message or "").strip()
    if not text:
        return Intent.CHAT, text

    # ---- 1. 前缀优先 ----
    for prefixes, intent in _PREFIX_MAP:
        for prefix in prefixes:
            if text.startswith(prefix):
                cleaned = text[len(prefix):].lstrip("：: ，,")
                return intent, cleaned

    # ---- 2. 关键词兜底 ----
    # 汇报类关键词容易与闲聊混淆，因此需要满足"出现频次"或"长度"门槛
    for keywords, intent in _KEYWORD_MAP:
        for kw in keywords:
            if kw in text:
                if intent is Intent.REPORT and len(text) < 6:
                    continue
                return intent, text

    # ---- 3. 兜底 ----
    return Intent.CHAT, text


def strip_at_mentions(message: str) -> str:
    """移除残留的 @提及，避免污染模型输入"""
    return re.sub(r"@[\w\u4e00-\u9fa5]+", "", message or "").strip()
