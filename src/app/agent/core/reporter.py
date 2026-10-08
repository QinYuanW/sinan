"""司南 · 汇报助手

职责：把用户"今天学了什么"的汇报（文字 + 照片）变成结构化结果。

为什么汇报走对话而不是表单：
    汇报本质是对话。信息不够时必须能追问"错了 3 道，是没思路还是算错了？"，
    表单只能一次性提交，做不到多轮澄清。

实现说明（v1）：
    这里是函数式的两段解析——先视觉识别照片，再用推理模型结构化输出。
    后续学完 LangGraph 子图后，可升级为真正的子 Agent（带 handoff 与多轮追问）。
"""

from __future__ import annotations

import os
from datetime import date as _date

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.roles import get_pack
from app.agent.tools.vision import recognize_photos
from app.models.schemas import ReportParse

load_dotenv()

_PARSER = None


def _get_parser():
    """懒加载结构化输出解析器。

    使用 function_calling 而非 json_mode：
    json_mode 在实战中曾出现"Failed to parse from completion"的失败，
    function_calling 走 tool call 通道，稳定性更好。
    """
    global _PARSER
    if _PARSER is None:
        model = init_chat_model(model="deepseek-chat")
        _PARSER = model.with_structured_output(ReportParse, method="function_calling")
    return _PARSER


def build_report_text(text: str | None, photos: list[str] | None, role: str) -> str:
    """把文字汇报与照片识别结果合并成一份待解析的汇报材料"""
    sections: list[str] = []

    if text and text.strip():
        sections.append("【用户文字汇报】\n" + text.strip())

    if photos:
        notes = recognize_photos(photos)
        blocks = [f"照片 {i}：{note}" for i, note in enumerate(notes, start=1)]
        sections.append("【照片识别结果】\n" + "\n".join(blocks))

    if not sections:
        return "用户提交了空汇报：既没有文字也没有照片。"

    return "\n\n".join(sections)


def parse_report(
    text: str | None = None,
    photos: list[str] | None = None,
    report_date: str | None = None,
    role: str = "kaoyan",
) -> ReportParse:
    """解析汇报，返回结构化结果。

    解析失败时降级为 needs_followup=True，让上层去追问用户，
    而不是抛异常或者编造数据。
    """
    date_str = report_date or _date.today().isoformat()
    pack = get_pack(role)

    try:
        material = build_report_text(text, photos, role)
    except Exception as exc:  # noqa: BLE001 — 视觉链路失败时降级
        return ReportParse(
            date=date_str,
            completed_kps=[],
            accuracy={},
            photos_recognized=[],
            problems=[],
            needs_followup=True,
            followup_question=f"照片识别失败（{type(exc).__name__}），请用文字描述今天做了哪些题、正确率如何？",
        )

    system = SystemMessage(content=pack.reporter_prompt())
    user = HumanMessage(
        content=(
            f"汇报日期：{date_str}\n\n"
            f"{material}\n\n"
            "请按结构化格式输出。注意：只有在你确实能从材料中判断掌握程度时才填写 accuracy；"
            "信息不足时把 needs_followup 设为 true，并在 followup_question 里写出你要问的具体问题。"
        )
    )

    try:
        result: ReportParse = _get_parser().invoke([system, user])
    except Exception as exc:  # noqa: BLE001 — 模型或解析失败
        return ReportParse(
            date=date_str,
            completed_kps=[],
            accuracy={},
            photos_recognized=[],
            problems=[],
            needs_followup=True,
            followup_question=(
                f"自动解析失败（{type(exc).__name__}），请直接说明今天完成了哪些知识点、"
                f"正确率大概多少？"
            ),
        )

    # 保证日期由服务端决定，避免模型自由发挥
    result.date = date_str
    # 防御：accuracy 值必须在 0-1
    result.accuracy = {
        k: max(0.0, min(1.0, float(v)))
        for k, v in (result.accuracy or {}).items()
        if v is not None
    }
    return result
