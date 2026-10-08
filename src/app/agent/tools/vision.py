"""司南 · 视觉识别工具

用 glm-4v-flash 识别学习照片中的题目与知识点。
设计上是「双模型编排」的一半：视觉模型只看图，推理模型负责规划与解析。

为什么不直接让主模型看图：
    主模型（deepseek 系列）是纯文本模型，不具备多模态能力。
    把视觉与推理拆开，既恢复了被丢失的多模态链路，
    也让两边都能用各自最擅长的模型。
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

# 学习照片的默认识别指令：只描述事实，不给建议（建议由规划器负责）
DEFAULT_PROMPT = """识别这张图片中的学习内容，按以下四项输出：
1. 题目内容：题干摘要（不要抄录全文，抓关键条件）
2. 涉及知识点：用数学二考点名称表述（如"极限的基本计算""矩阵的秩"）
3. 作答情况：全对 / 部分错误 / 空白未做（看不清就说看不清）
4. 错误原因：若有批改痕迹，说明错在概念、计算还是方法

只输出识别到的客观事实，不要给学习建议，不要鼓励。看不清的部分明确说"看不清"。"""

_VISION_MODEL = None


def get_vision_model():
    """懒加载视觉模型（避免导入期就要求环境变量齐全）"""
    global _VISION_MODEL
    if _VISION_MODEL is None:
        api_key = os.getenv("ZHIPUAI_API_KEY")
        base_url = os.getenv("ZHIPUAI_BASE_URL")
        if not api_key:
            raise RuntimeError("缺少环境变量 ZHIPUAI_API_KEY，无法调用视觉模型")
        _VISION_MODEL = init_chat_model(
            model="glm-4v-flash",
            model_provider="openai",  # 关键：走 OpenAI 兼容协议
            api_key=api_key,
            base_url=base_url,
        )
    return _VISION_MODEL


def to_data_url(path: str | Path) -> str:
    """把本地图片文件转成 data URL（前端直传 OSS 失败时的兜底通道）"""
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _normalize_image(image: str) -> str:
    """接受 http(s) URL、data URL 或本地文件路径"""
    if image.startswith(("http://", "https://", "data:")):
        return image
    if Path(image).exists():
        return to_data_url(image)
    raise ValueError(f"无法解析的图片地址：{image[:60]}")


def recognize_photo(image: str, prompt: str = DEFAULT_PROMPT) -> str:
    """识别单张照片，返回文字描述。

    失败时抛出异常，由调用方决定是中断还是降级，
    这里不吞异常——静默失败会让规划建立在信错误息上。
    """
    model = get_vision_model()
    message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": _normalize_image(image)}},
            {"type": "text", "text": prompt},
        ]
    )
    response = model.invoke([message])

    content = response.content
    if isinstance(content, str):
        return content.strip()
    # 某些实现会返回 content block 列表
    if isinstance(content, list):
        chunks = []
        for block in content:
            if isinstance(block, dict):
                chunks.append(str(block.get("text", "")))
            else:
                chunks.append(str(block))
        return "".join(chunks).strip()
    return str(content)


def recognize_photos(images: list[str], prompt: str = DEFAULT_PROMPT) -> list[str]:
    """批量识别。单张失败不阻断其余照片，失败信息以标记形式返回。"""
    results: list[str] = []
    for idx, image in enumerate(images or [], start=1):
        try:
            results.append(recognize_photo(image, prompt))
        except Exception as exc:  # noqa: BLE001 — 单张失败不应影响整批
            results.append(f"[第 {idx} 张照片识别失败：{type(exc).__name__}: {exc}]")
    return results
