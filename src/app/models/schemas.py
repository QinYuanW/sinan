"""司南 · 数据模型

分三类：
    1. 请求模型（API 入参）
    2. 领域模型（引擎产出，也是结构化输出的目标）
    3. 汇报模型（解析结果）
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ======================================================================
# 1. 请求模型
# ======================================================================
class ChatRequest(BaseModel):
    message: str
    image_url: Optional[str] = None
    thread_id: str
    user_id: str = "default"
    role: str = "kaoyan"


class ProfileIn(BaseModel):
    """设置 / 更新用户目标"""

    user_id: str = "default"
    role: str = "kaoyan"
    name: Optional[str] = None
    goal_payload: dict[str, Any] = Field(default_factory=dict)
    daily_hours: float = 4.0
    subjects: list[str] = Field(default_factory=list)


class CheckInIn(BaseModel):
    """打卡入参：只有这些，没有内容 —— 打卡不触发重规划"""

    user_id: str = "default"
    role: str = "kaoyan"
    date: Optional[str] = None


class ReportIn(BaseModel):
    """汇报入参：文字 + 照片双通道"""

    user_id: str = "default"
    role: str = "kaoyan"
    date: Optional[str] = None
    text: Optional[str] = None
    photos: list[str] = Field(default_factory=list)


class PlanRequest(BaseModel):
    user_id: str = "default"
    role: str = "kaoyan"
    date: Optional[str] = None
    force_refresh: bool = False


# ======================================================================
# 2. 领域模型
# ======================================================================
class TaskItem(BaseModel):
    """一条作战任务。每个字段都要让用户看懂"为什么是它"。"""

    kp: str = Field(description="知识点名称")
    subject: str = Field(description="所属科目")
    chapter: str = Field(default="", description="所属章节")
    exam_freq: float = Field(description="考频权重 0-1")
    mastery: float = Field(description="当前掌握度 0-1")
    est_hours: float = Field(description="预计投入小时数")
    priority: float = Field(description="综合优先级 0-1")
    reason: str = Field(description="为什么先做它，必须用具体数字说明")
    acceptance: str = Field(description="验收标准：怎么算这个知识点过关了")


class DailyPlan(BaseModel):
    date: str
    role: str
    total_hours: float = Field(description="清单总预计耗时")
    tasks: list[TaskItem] = Field(default_factory=list)
    carry_over: list[TaskItem] = Field(default_factory=list, description="顺延项")
    coach_note: str = Field(default="", description="节奏提醒")


class Milestone(BaseModel):
    name: str
    start_date: str
    end_date: str
    target_kps: list[str] = Field(default_factory=list)
    note: str = ""


class LongPlan(BaseModel):
    role: str
    exam_date: Optional[str] = None
    days_left: Optional[int] = None
    milestones: list[Milestone] = Field(default_factory=list)
    coverage_note: str = ""


# ======================================================================
# 3. 汇报模型
# ======================================================================
class ReportParse(BaseModel):
    """汇报解析结果。accuracy 用 0-1 表示该知识点的实际掌握证据强度。"""

    date: str = Field(description="汇报日期 YYYY-MM-DD")
    completed_kps: list[str] = Field(default_factory=list, description="实际完成的知识点")
    accuracy: dict[str, float] = Field(
        default_factory=dict, description="知识点 -> 正确率/掌握证据 0-1"
    )
    photos_recognized: list[str] = Field(default_factory=list, description="照片中识别出的知识点")
    problems: list[str] = Field(
        default_factory=list, description="具体问题，要指向知识点与失误类型"
    )
    needs_followup: bool = Field(default=False, description="信息是否不足以判断掌握度")
    followup_question: str = Field(default="", description="需要追问的问题")


class ReplanResult(BaseModel):
    """重规划结果。包含"决定不改"的情况，因为不改也是一种决策。"""

    date: str
    mastery_delta: dict[str, float] = Field(default_factory=dict)
    problems: list[str] = Field(default_factory=list)
    replan_reason: str = ""
    changed: bool = Field(default=False, description="计划是否真的被改动")
    next_plan: Optional[DailyPlan] = None


# ======================================================================
# 4. 通用响应
# ======================================================================
class OkResponse(BaseModel):
    success: bool = True
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
