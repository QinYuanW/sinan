"""B3 · 多模型编排（视觉 / 推理分工）

学什么
    1) 智谱 glm-4v 通过 OpenAI 兼容协议接入 —— 关键是 model_provider="openai"
    2) 为什么不能直接让 deepseek 看图（它不是多模态模型）
    3) 把"看图"收敛成一个工具，交给专门的视觉模型 —— 这就是编排

对应司南
    src/app/agent/tools/vision.py:37-51   get_vision_model()
    src/app/agent/sinan_agent.py:77       recognize_homework

怎么跑
    python examples/b03_multi_model.py                     # 只看配置
    python examples/b03_multi_model.py path/to/photo.jpg   # 真的识别一张图
    （需要 .env 里的 ZHIPUAI_API_KEY / ZHIPUAI_BASE_URL；
      PART 3 还需要 DEEPSEEK_API_KEY）

观察点
    PART 1：一个 base_url 就把智谱模型接进来了。
    PART 2：deepseek 收到图片会报错 —— 证明它真的看不了图。
    PART 3：glm-4v 能看图，并且被包装成"工具"的形状。

TODO（学完自己改）
    1) 改 DEFAULT_PROMPT，专治手写体识别（提示词里要求"看不清就明说"）
    #完成
    2) 拿同一张图，对比"glm-4v 直接识别" vs "deepseek 调用工具后转述"，
       看看信息在传递过程中丢了多少
"""

from __future__ import annotations

import base64
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

DEFAULT_PROMPT ="""这是一张手写作业/习题照片，请识别其中的内容。

【手写体识别规则】
手写体可能潦草、连笔、涂改、模糊。遇到任何不确定的字符、符号或公式：
- 不猜测、不脑补、不用相似字符代替
- 直接在对应位置写"看不清"
- 整段无法辨认时，明确写"此处无法识别"

按以下四项输出：
1. 题目内容：题干摘要（抓关键条件，不要照抄全文）
2. 涉及知识点：用考点名称表述（如"极限的基本计算""矩阵的秩"）
3. 作答情况：全对 / 部分错误 / 空白未做（看不清就写"看不清"）
4. 错误原因：若有批改痕迹，说明错在概念、计算还是方法

只输出识别到的客观事实，不要给建议，不要鼓励。
任何模糊或不确定的地方，务必明说"看不清"。"""


# ======================================================================
# PART 1 的主角：把智谱模型接进来
# ======================================================================
def get_vision_model():
    api_key = os.getenv("ZHIPUAI_API_KEY")
    base_url = os.getenv("ZHIPUAI_BASE_URL")

    if not api_key:
        raise RuntimeError("缺少 ZHIPUAI_API_KEY，无法调用视觉模型")

    return init_chat_model(
        model="glm-4v-flash",
        model_provider="openai",   # <<< 全篇最重要的一行
        api_key=api_key,
        base_url=base_url,
    )


def to_data_url(path: str | Path) -> str:
    """本地图片 -> data URL。前端直传 OSS 失败时的兜底通道。"""
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _normalize_image(image: str) -> str:
    """接受 http(s) URL、data URL 或本地路径，统一成一个能喂给模型的字符串。"""
    if image.startswith(("http://", "https://", "data:")):
        return image
    if Path(image).exists():
        return to_data_url(image)
    raise ValueError(f"无法解析的图片地址：{image[:60]}")


def recognize_photo(image: str, prompt: str = DEFAULT_PROMPT) -> str:
    """识别单张照片。失败就抛异常，不吞 —— 静默失败会让规划建立在错误信息上。"""
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
    if isinstance(content, list):
        return "".join(
            str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content
        ).strip()
    return str(content)


# ======================================================================
# 主流程
# ======================================================================
def main() -> None:
    print("=" * 64)
    print("PART 1 · 一个 base_url，把智谱模型接进 LangChain")
    print("=" * 64)
    print("""
    init_chat_model(
        model="glm-4v-flash",
        model_provider="openai",      # <- 关键
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        base_url=os.getenv("ZHIPUAI_BASE_URL"),
    )

    为什么写 openai？因为智谱提供了 OpenAI 兼容接口。
    LangChain 的 openai provider 只需要换个 base_url 就能对接，
    协议层的东西（消息格式、tool call、流式）它全都帮你处理了。
    """)

    print("=" * 64)
    print("PART 2 · 证明 deepseek 真的看不了图")
    print("=" * 64)
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("    跳过（缺 DEEPSEEK_API_KEY）")
    else:
        text_model = init_chat_model(model="deepseek-chat")
        try:
            text_model.invoke(
                [
                    HumanMessage(
                        content=[
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                            },
                            {"type": "text", "text": "这张图里写了什么？"},
                        ]
                    )
                ]
            )
            print("    没报错？那它多半把图片块忽略了 —— 反正它给不出图里的内容。")
        except Exception as exc:  # noqa: BLE001
            print(f"    ✓ 如预期报错了：{type(exc).__name__}")
            print(f"      {str(exc)[:160]}")
            print()
            print("    这就是为什么不能直接把图片塞给主模型。")
            print("    必须让专职的视觉模型去看，再把文字结果带回来。")

    print()
    print("=" * 64)
    print("PART 3 · 真的识别一张图")
    print("=" * 64)
    if len(sys.argv) < 2:
        print("    用法：python examples/b03_multi_model.py <图片路径>")
        print("    （找一张你自己的手写习题照，看它识得怎么样）")
        return

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"    找不到文件：{image_path}")
        return

    print(f"    图片：{image_path}")
    print("    识别中……")
    try:
        result = recognize_photo(image_path)
    except Exception as exc:  # noqa: BLE001
        print(f"    ✗ 识别失败：{type(exc).__name__}: {exc}")
        return

    print()
    print("    ---- 识别结果 ----")
    print("   ", result.replace("\n", "\n    "))

    print()
    print("-" * 64)
    print("现在把它和司南对上：")
    print("    recognize_photo()  -> 被 @tool 包一层 -> recognize_homework")
    print("    主模型(deepseek) 需要看图时，调用这个工具")
    print("    于是：推理归推理，看图归看图，各用最合适的模型")
    print()
    print("顺带一提：这个'工具'对主模型来说是不透明的 ——")
    print("它只知道'调用后能拿到图片里的文字描述'，")
    print("完全不知道背后是 glm-4v。这就是编排的解耦价值。")


if __name__ == "__main__":
    main()
