"""B3.1 · 视觉提示词 A/B：手写习题识别到底能准到什么程度

学什么
    同一张手写习题照片、同一个视觉模型，只改提示词策略，结果能差多远。
    这一课的价值是**摸清能力边界** —— 知道什么能做、什么做不到，
    比"调出一份好看的输出"重要得多。

对应司南
    src/app/agent/tools/vision.py    DEFAULT_PROMPT（当前为 v4「逐字转录」版）
    src/app/agent/core/reporter.py   消费识别结果，转成 ReportParse

怎么跑
    python examples/b03.1_vision_prompt_ab.py <手写习题照片路径>
    （需要 .env 里的 ZHIPUAI_API_KEY；只调用免费的 glm-4v-flash）

观察点
    v1 通用      —— "识别图片中的学习内容"
    v4 逐字转录  —— "把每道题旁边的手写内容逐字照抄"
    重点看两者对【手写内容】的召回差异。

实测结论（2026-10-11，一张 10 题的微积分手写卷，四轮对照）
    完整记录见 vision.py 顶部注释。核心两条：
      1) 提示词只能改变模型的「注意力方向」，改变不了「识别精度」——
         要求它判对错，它会努力去找但读不准（-1/2 读成 1/2），
         甚至把 10 道题全判成"正确"（比漏报更危险）；
         禁止它判对错，它又会失去找手写的动机（全报"未见手写"）。
      2) glm-4v-flash 能读对**字母选项**（B/C/D），但读不出**手写数字与数学公式**；
         而且**不稳定** —— 同一张图同样的提示词，复跑一次召回可能从 4/10 掉到 0/10。

    所以：照片识别的定位是「提取知识点 + 判断有没有作答」的线索，
    **不能作为作答正确率的数据来源** —— 正确率必须靠文字汇报或用户确认。
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BASE / ".env")

from app.agent.tools.vision import DEFAULT_PROMPT, recognize_photo  # noqa: E402

V1_GENERIC = """识别这张图片中的学习内容，按以下四项输出：
1. 题目内容：题干摘要（不要抄录全文，抓关键条件）
2. 涉及知识点：用数学二考点名称表述（如"极限的基本计算""矩阵的秩"）
3. 作答情况：全对 / 部分错误 / 空白未做（看不清就说看不清）
4. 错误原因：若有批改痕迹，说明错在概念、计算还是方法

只输出识别到的客观事实，不要给学习建议，不要鼓励。看不清的部分明确说"看不清"。"""

CASES = [
    ("v1 通用", V1_GENERIC),
    ("v4 逐字转录（当前生产版）", DEFAULT_PROMPT),
]


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python examples/b03.1_vision_prompt_ab.py <手写习题照片路径>")
        return 1

    img = sys.argv[1]
    if not Path(img).exists():
        print(f"图片不存在: {img}")
        return 1

    for label, prompt in CASES:
        print("=" * 74)
        print(label)
        print("=" * 74)
        try:
            print(recognize_photo(img, prompt))
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {type(exc).__name__}: {exc}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
