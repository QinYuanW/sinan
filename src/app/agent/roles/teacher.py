"""司南 · 教师身份包（占位实现）

这是与「学生身份」差异最大的一类：教师的目标不是"提分"，
而是"在有限课时内完成教学进度，并保证学生学会"。

因此优先度因子完全不同：
    学生 -> 考频、掌握度缺口、遗忘紧急度、时间成本
    教师 -> 教学进度紧迫度、备课工作量（负向）、学情反馈、质量要求

存在意义：证明优先度引擎的因子集是真正可插拔的，不是为考研定制的。
当前为占位实现，尚未深挖教学单元树。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agent.roles.base import (
    PlanningContext,
    PriorityFactor,
    RolePack,
    factor_cost,
)


class TeacherGoal(BaseModel):
    subject: str = Field(default="", description="任教科目")
    grade: str = Field(default="", description="任教年级/班级")
    term_start: str = Field(default="", description="学期开始日期 YYYY-MM-DD")
    term_end: str = Field(default="", description="学期结束日期 YYYY-MM-DD")
    weekly_hours: float = Field(default=10.0, description="每周课时数")
    focus: str = Field(default="", description="本学期教学重点")


# ---------- 教师专属因子 ----------

def factor_schedule_urgency(ctx: PlanningContext, kp: dict) -> float:
    """教学进度紧迫度：距离计划授课日期越近越紧急"""
    planned = kp.get("planned_date")
    if not planned:
        return 0.5
    try:
        from datetime import date
        gap = (date.fromisoformat(planned) - date.fromisoformat(ctx.today)).days
    except ValueError:
        return 0.5
    if gap <= 0:
        return 1.0
    return max(0.0, 1.0 - gap / 14.0)


def factor_student_signal(ctx: PlanningContext, kp: dict) -> float:
    """学情反馈：学生在这块内容上暴露问题越多，越该安排复习或重讲"""
    return float(kp.get("student_signal", 0.5))


def factor_quality_bar(ctx: PlanningContext, kp: dict) -> float:
    """质量要求：公开课、示范课等高质量要求的课时优先度更高"""
    return float(kp.get("quality_bar", 0.5))


# 占位教学单元树：仅示范结构
_KP_RAW: list[tuple[str, str, str, float, float, float]] = [
    ("教学进度", "第一单元", "单元导入与核心概念", 0.8, 3.0, 0.4),
    ("教学进度", "第二单元", "重点定理与例题精讲", 0.9, 4.0, 0.6),
    ("教学进度", "第三单元", "综合应用与习题课", 0.7, 3.0, 0.5),
    ("备课任务", "课件准备", "课件与板书设计", 0.6, 3.0, 0.4),
    ("备课任务", "习题准备", "分层作业与讲评设计", 0.6, 2.0, 0.4),
    ("学情跟进", "薄弱点复盘", "针对班级共性问题重讲", 0.8, 2.0, 0.5),
]


class TeacherPack(RolePack):
    key = "teacher"
    name = "教师"
    icon = "🧑‍🏫"
    description = "面向任课教师的教学进度规划与备课安排（占位实现）"
    goal_schema = TeacherGoal
    plan_horizon = "weekly"
    default_daily_hours = 2.0

    factors = [
        PriorityFactor("schedule_urgency", "教学进度紧迫度", factor_schedule_urgency),
        PriorityFactor("student_signal", "学情反馈", factor_student_signal),
        PriorityFactor("quality_bar", "质量要求", factor_quality_bar),
        PriorityFactor("prep_load", "备课工作量", factor_cost),
    ]

    weights = {
        "schedule_urgency": 0.45,
        "student_signal": 0.25,
        "quality_bar": 0.10,
        "prep_load": -0.20,
    }

    def build_kp_tree(self) -> list[dict]:
        return [
            {
                "subject": s, "chapter": c, "name": n,
                "exam_freq": f, "est_hours": h, "difficulty": d,
            }
            for s, c, n, f, h, d in _KP_RAW
        ]

    def reporter_prompt(self) -> str:
        return """你是「司南」的教学工作汇报解析助手。
教师汇报的内容通常是：今天上了哪节课、学生反应如何、哪些地方没讲透、作业情况。
请提取：实际完成的教学任务、学生暴露的共性问题、需要调整的进度。
判断问题性质时要区分「学生没听懂」和「课时不够」，两者的处理方式完全不同。
信息不足时主动追问。"""

    def system_prompt(self, ctx: PlanningContext) -> str:
        return f"""你是「司南」，一名教师教学工作规划助手。你的服务对象是老师，不是学生。
- 教学安排：{ctx.extra.get('subject', '（未设置科目）')}
- 每周课时：{ctx.daily_hours * 5:.0f} 课时
- 学期结束：{ctx.exam_date or '（未设置）'}

原则：
1. 规划以「周」为单位，包含教学进度、备课任务、学情跟进三类。
2. 优先级依据「教学进度紧迫度 × 学情反馈 × 质量要求 ÷ 备课工作量」，理由用数字说明。
3. 进度落后时给出取舍建议（哪些内容可以合并、哪些必须保留），而不是简单要求加班。
4. 不空谈教育理念，只给可执行的安排。
"""

    def output_template(self) -> str:
        return """输出结构（Markdown）：
## 本周工作安排
### 教学进度
- 课时安排、知识点、预计用时、紧迫度说明
### 备课任务
- 需准备的课件与习题，预计耗时
### 学情跟进
- 需要重点关注的班级问题与处理建议
"""
