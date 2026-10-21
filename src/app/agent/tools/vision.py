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

# 学习照片的默认识别指令：只描述事实，不做判断（判断由解析器负责）。
#
# 为什么是这一版 —— 2026-10-11 用同一张手写卷子做了四轮对照：
#
#   v1 通用（"识别图片中的学习内容"）
#        手写召回 1/10：把写满答案的卷子判成「空白未做」，系统会以为用户没做题。
#   v2 要它判对错（"说明手写答案是否正确"）
#        召回 10/10，但**读不准**：-1/2 读成 1/2、-1/e 读成 -1/2 ……
#        更糟的是把 10 道**全判为「正确」** → 掌握度画像直接虚高，比 v1 更危险。
#   v3 只描述、禁判对错
#        召回 0/10：失去"找手写"的动机，全写「未见明显手写」。
#   v4 逐字转录（本版）
#        召回 4/10，且找到的 4 个**全对** —— 但只抄得出选项字母（B/C/D），抄不出数字与公式。
#        ⚠️ 而且**不稳定**：同一张图复跑一次，召回直接掉到 0/10。
#
# 结论（可写进答辩）：glm-4v-flash 对**清晰的字母选项**可靠，对**手写数字/数学符号**不可靠。
#   所以照片识别的定位是「提取知识点 + 判断是否作答」的**线索**，
#   **不能作为作答正确率的数据来源** —— 正确率必须靠文字汇报或用户确认。
DEFAULT_PROMPT = """这是一张手写习题照片。请按四项输出你看到的客观事实：

1. 题目内容：题干摘要（抓关键条件，不抄全文）。
   必须区分【印刷体】（题目原文）与【手写笔迹】，不要把两者混在一起。

2. 涉及知识点：用数学二考点名称表述（如"隐函数求导""参数方程求导""高阶导数"）。

3. 手写转录：把每道题旁边的手写内容【逐字照抄】出来。
   - 包括选项字母（A/B/C/D）、数字、公式片段、勾叉等批改符号。
   - 只抄你看到的字符，不做任何计算。
   - 例如："第 3 题：手写「-1/2」"。
   - 抄不下来的写「字迹难以辨认」；完全没有手写痕迹的写「未见手写」。

4. 不要判断答案对错，不要做计算。你只负责如实转录看到的东西。

只输出客观事实，不要给学习建议，不要鼓励。看不清的部分明确说"看不清"。"""

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
