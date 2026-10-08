"""司南 · 专升本身份包（占位实现）

存在意义：证明「加身份不改引擎」这条架构约束是真的成立。
当前只提供目标模型、最小考点树与因子配置，尚未按官方大纲深挖。
后续扩展方式：补齐 _KP_RAW 考点树 + 按需调整权重，引擎代码无需改动。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agent.roles.base import (
    PlanningContext,
    PriorityFactor,
    RolePack,
    factor_cost,
    factor_exam_freq,
    factor_mastery_gap,
    factor_urgency,
)


class ZhuanshengbenGoal(BaseModel):
    target_school: str = Field(default="", description="报考本科院校")
    target_major: str = Field(default="", description="报考专业")
    exam_date: str = Field(default="", description="考试日期 YYYY-MM-DD")
    exam_subjects: list[str] = Field(default_factory=list, description="考试科目")
    current_level: str = Field(default="", description="目前基础水平描述")


# 占位考点树：仅示范结构，正式使用前需按目标院校大纲补全
_KP_RAW: list[tuple[str, str, str, float, float, float]] = [
    ("高等数学", "函数与极限", "极限的计算", 0.85, 5.0, 0.50),
    ("高等数学", "导数与微分", "导数计算与应用", 0.80, 5.0, 0.50),
    ("高等数学", "积分", "不定积分与定积分", 0.80, 5.0, 0.55),
    ("英语", "词汇", "核心词汇与高频词组", 0.95, 20.0, 0.40),
    ("英语", "阅读", "阅读理解题型专项", 0.85, 12.0, 0.60),
    ("英语", "写作", "应用文与短文写作", 0.75, 8.0, 0.55),
    ("计算机基础", "基础知识", "计算机组成与操作系统基础", 0.70, 6.0, 0.50),
]


class ZhuanshengbenPack(RolePack):
    key = "zhuanshengben"
    name = "专升本"
    icon = "📘"
    description = "面向专科升本科考试（占位实现，暂未按官方大纲深挖考点树）"
    goal_schema = ZhuanshengbenGoal
    plan_horizon = "long+daily"
    default_daily_hours = 3.0

    factors = [
        PriorityFactor("exam_freq", "真题频率", factor_exam_freq),
        PriorityFactor("mastery_gap", "掌握度缺口", factor_mastery_gap),
        PriorityFactor("urgency", "遗忘紧急度", factor_urgency),
        PriorityFactor("cost_penalty", "时间成本惩罚", factor_cost),
    ]

    # 专升本英语占比高、提分周期长，因此成本惩罚相对轻
    weights = {
        "exam_freq": 0.38,
        "mastery_gap": 0.34,
        "urgency": 0.16,
        "cost_penalty": -0.08,
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
        return """你是「司南」的专升本学习汇报解析助手。
从用户的文字或照片中提取：完成的知识点、正确率、暴露的问题。
专升本考生基础差异大，遇到「完全看不懂」的情况要标记为基础缺口而非粗心，
这类问题需要回到前置知识点补课，而不是继续刷题。
信息不足时主动追问，不要猜测。"""

    def system_prompt(self, ctx: PlanningContext) -> str:
        return f"""你是「司南」，一名专升本私人学习规划师。
- 考试日期：{ctx.exam_date or '（未设置）'}（剩余 {ctx.days_left if ctx.days_left is not None else '?'} 天）
- 每日可用时长：{ctx.daily_hours} 小时

原则：
1. 每个任务写明做什么、为什么优先、预计耗时、验收标准。
2. 优先级依据「真题频率 × 掌握度缺口 × 遗忘紧急度 ÷ 时间成本」，理由要用数字说话。
3. 基础缺口优先于刷题量：前置知识不牢时先补前置，不要盲目堆题。
4. 不空谈鼓励，不给无法执行的任务。
"""
