"""B4 · with_structured_output：json_mode vs function_calling

学什么
    你实战 .ipynb 里用的是 method="json_mode"，司南里换成了 function_calling。
    两者不是"哪个新用哪个"，而是走两条完全不同的底层通道：
      · json_mode        —— 要求模型直接吐 JSON 文本，再解析成对象
      · function_calling —— 把 schema 伪装成一个 tool，走 tool call 通道
    tool call 是模型的"母语"，所以 function_calling 通常更稳。

对应司南
    src/app/agent/core/reporter.py:32-43   _get_parser()

怎么跑
    python examples/b04_structured_compare.py
    （需要 DEEPSEEK_API_KEY，会跑 10 次模型调用，约 30-60 秒）

观察点
    重点不是"谁成功率更高"，而是你自己跑出来一张数据表 ——
    这就是答辩时能拿出来的一张图：同一份 schema，两种 method 的稳定性对比。

TODO（学完自己改）
    1) 把 FUZZY_INPUTS 换成你自己的模糊汇报，看结论是否一致
    2) 加第三个 method="json_schema"（如果模型支持），三方对比
    3) 想想：为什么司南的 reporter 在解析失败时要降级成 needs_followup，
       而不是抛异常或硬凑一个结果？
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

load_dotenv()


# ======================================================================
# 结构化目标：和司南里的 ReportParse 同构（简化版）
# ======================================================================
class ReportParse(BaseModel):
    """学习汇报的结构化解析结果。"""

    completed_kps: list[str] = Field(
        default_factory=list, description="今天完成的知识点，用考点名称表述"
    )
    accuracy: float | None = Field(
        None, description="本次作答的正确率（0-1）。无法从材料判断时必须留空，不要猜。"
    )
    needs_followup: bool = Field(
        False, description="材料信息不足、需要向用户追问时置为 true"
    )
    followup_question: str | None = Field(
        None, description="需要追问的具体问题，一次只问一个"
    )


SYSTEM = """你是一个学习汇报解析助手。
从用户的汇报中抽取完成的知识点与正确率。
严格要求：
- 只有明确说了数量（如"15 道错了 4 道"）才能填 accuracy
- 只能判断"做了"但判断不出对错的，必须把 needs_followup 设为 true
- 不要编造知识点名称，用材料里出现的表述"""

# 故意模糊的输入 —— 越模糊越容易触发解析失败
FUZZY_INPUTS = [
    "今天做了点题",
    "还行吧",
    "错了几道",
    "学了一下午，感觉效率不高",
    "把极限那章过了一遍，大概对了一半多",
]

METHODS = ["json_mode", "function_calling"]


def run_one(parser, text: str) -> tuple[bool, str, float]:
    """跑一次解析，返回 (是否成功, 摘要或错误, 耗时秒)"""
    t0 = time.perf_counter()
    try:
        result: ReportParse = parser.invoke(
            [SystemMessage(content=SYSTEM), HumanMessage(content=text)]
        )
        elapsed = time.perf_counter() - t0
        if result.needs_followup:
            note = f"需追问: {str(result.followup_question)[:24]}"
        elif result.accuracy is not None:
            note = f"accuracy={result.accuracy}"
        else:
            note = f"kps={len(result.completed_kps)}"
        return True, note, elapsed
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        return False, f"{type(exc).__name__}", elapsed


def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("跳过：未检测到 DEEPSEEK_API_KEY，请先在 .env 里配置。")
        return

    base_model = init_chat_model(model="deepseek-chat")
    parsers = {
        m: base_model.with_structured_output(ReportParse, method=m) for m in METHODS
    }

    stats = {m: {"ok": 0, "fail": 0, "time": 0.0, "errors": []} for m in METHODS}
    rows: list[tuple[str, dict[str, tuple[bool, str, float]]]] = []

    print("=" * 78)
    print("同一份 schema，两种 method，各跑 5 条模糊输入")
    print("=" * 78)

    for text in FUZZY_INPUTS:
        print(f"\n输入：{text}")
        row: dict[str, tuple[bool, str, float]] = {}
        for method in METHODS:
            ok, note, elapsed = run_one(parsers[method], text)
            row[method] = (ok, note, elapsed)
            s = stats[method]
            s["ok" if ok else "fail"] += 1
            s["time"] += elapsed
            if not ok:
                s["errors"].append(note)
            flag = "OK  " if ok else "FAIL"
            print(f"    {method:18s} [{flag}] {elapsed:5.2f}s  {note}")
        rows.append((text, row))

    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"{'method':20s}{'成功':>6s}{'失败':>6s}{'成功率':>10s}{'平均耗时':>12s}")
    print("-" * 78)
    for m in METHODS:
        s = stats[m]
        total = s["ok"] + s["fail"]
        rate = s["ok"] / total * 100 if total else 0
        avg = s["time"] / total if total else 0
        print(f"{m:20s}{s['ok']:>6d}{s['fail']:>6d}{rate:>9.0f}%{avg:>11.2f}s")

    print()
    for m in METHODS:
        if stats[m]["errors"]:
            print(f"  {m} 的失败类型：{set(stats[m]['errors'])}")

    print()
    print("-" * 78)
    print("怎么读这张表：")
    print("  · 如果两边都 100% —— 说明 deepseek 对这两种 method 支持都很好，")
    print("    那就该比别的：延迟、token 消耗、对复杂 schema 的表现。")
    print("  · 如果 json_mode 出现 'OutputParserException' 或")
    print("    'Failed to parse' —— 那就是你实战里踩过的坑，原样复现了。")
    print()
    print("司南选择了 function_calling，理由写在 reporter.py 的注释里：")
    print("    json_mode 在实战中曾出现 Failed to parse，")
    print("    function_calling 走 tool call 通道，稳定性更好。")
    print()
    print("⚠️ 但别把这句话当结论 —— 你现在有数据了，用数据说话。")

    # TODO: 换成你自己的模糊汇报，看看结论稳不稳


if __name__ == "__main__":
    main()
