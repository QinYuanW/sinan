"""B4.1 · function_calling vs json_mode：结构化输出的失败率实测

学什么
    with_structured_output 有两种 method，走完全不同的底层通道：
      json_mode        —— 要求模型"直接吐 JSON 文本"，再由框架解析
      function_calling —— 把 schema 伪装成一个 tool，走 tool call 通道

    这一课把两者的失败率【实测】出来 —— 这是一张能写进答辩的数据。

对应司南
    src/app/agent/core/reporter.py:32  _get_parser() 用的是 function_calling，
    docstring 里写着"json_mode 曾出现 Failed to parse"。本脚本就是给它取证。

怎么跑
    python examples/b04.1_structured_method_compare.py          # 每组 10 次
    python examples/b04.1_structured_method_compare.py --quick  # 每组 3 次（省额度）
    （需要 .env 里的 DEEPSEEK_API_KEY）

观察点
    1) 三组对照：function_calling / json_mode（裸） / json_mode（+JSON 提示）
    2) 失败原因分布 —— 重点看模型输出的 JSON 字段名长什么样

实测结果（2026-10-11，deepseek-chat，每组 10 次模糊输入）
    function_calling      成功 10/10   失败率   0%
    json_mode（裸）        成功  0/10   失败率 100%   全部 400: Prompt must contain the word 'json'
    json_mode（+提示）     成功  3/10   失败率  70%   全部 Failed to parse

    模型自己编的字段名（典型）：
        {"date": ..., "completed": [...]}            <- 正确字段是 completed_kps
        {"completed_points": [...]}
        {"report_date": ..., "completed_items": [...]}
        {"summary": ...}

结论
    json_mode 下模型【看不到字段名】，只能靠猜 —— 所以字段名自由发挥。
    function_calling 把 schema 变成 tool 定义，字段名是模型"必须照着填"的参数，
    因此零失败。这就是"tool call 是模型的母语"的量化证据。
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from dotenv import load_dotenv  # noqa: E402

# 显式指定 .env：load_dotenv() 默认按【脚本位置】向上找，不是按 cwd
load_dotenv(BASE / ".env")

from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from app.agent.roles import get_pack  # noqa: E402
from app.models.schemas import ReportParse  # noqa: E402

QUICK = "--quick" in sys.argv

# 刻意用"模糊输入"：口语化、信息不全、带错字 —— 这才是解析器真正的压力测试
ALL_INPUTS = [
    "今天做了点题",
    "极限那块不太行",
    "做了8道错了3道",
    "今天没学",
    "数学看了会儿书，感觉还行",
    "夹逼定理好难啊",
    "10题错3",
    "昨天做的高数题今天复习了一下错题，大概有一半不会",
    "uuu 做了几道题 不会",
    "线代矩阵那块做了5题，对了2题，另外高数没做",
]

INPUTS = ALL_INPUTS[:3] if QUICK else ALL_INPUTS

BASE_PROMPT = get_pack("kaoyan").reporter_prompt()

VARIANTS = [
    ("function_calling", "与生产一致：prompt 不含 JSON 字样", BASE_PROMPT, "function_calling"),
    ("json_mode 裸用", "同一 prompt，不提 JSON", BASE_PROMPT, "json_mode"),
    ("json_mode+提示", "prompt 末尾补一句『以 JSON 输出』", BASE_PROMPT + "\n\n请严格以 JSON 格式输出。", "json_mode"),
]


def user_content(text: str) -> str:
    return (
        "汇报日期：2026-10-11\n\n【用户文字汇报】\n"
        f"{text}\n\n"
        "请按结构化格式输出。注意：只有在你确实能从材料中判断掌握程度时才填写 accuracy；"
        "信息不足时把 needs_followup 设为 true，并在 followup_question 里写出你要问的具体问题。"
    )


def main() -> int:
    print("=" * 80)
    print(f"每组样本数：{len(INPUTS)}" + ("（--quick 模式）" if QUICK else ""))
    print("=" * 80)

    model = init_chat_model(model="deepseek-chat")
    results = []

    for label, desc, sys_prompt, method in VARIANTS:
        print(f"\n[{label}] {desc}")
        parser = model.with_structured_output(ReportParse, method=method)
        ok, errors = 0, []
        t0 = time.perf_counter()

        for i, text in enumerate(INPUTS, 1):
            try:
                r = parser.invoke(
                    [
                        SystemMessage(content=sys_prompt),
                        HumanMessage(content=user_content(text)),
                    ]
                )
                if isinstance(r, ReportParse):
                    ok += 1
                else:
                    errors.append(f"返回类型异常: {type(r).__name__}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {str(exc)[:70]}")
            print(f"    {i}/{len(INPUTS)}", end="\r", flush=True)

        dt = time.perf_counter() - t0
        results.append((label, ok, len(INPUTS) - ok, errors, dt))
        print(f"    -> 成功 {ok}/{len(INPUTS)}，耗时 {dt:.1f}s" + " " * 10)

    print()
    print("=" * 80)
    print(f"{'变体':<18}{'成功':<8}{'失败':<8}{'失败率':<10}耗时")
    print("=" * 80)
    for label, ok, bad, _errors, dt in results:
        print(f"{label:<18}{ok:<8}{bad:<8}{bad / len(INPUTS) * 100:>5.0f}%    {dt:.1f}s")

    print()
    print("失败原因分布：")
    for label, _ok, _bad, errors, _dt in results:
        if not errors:
            continue
        print(f"\n  [{label}]")
        for msg, cnt in Counter(errors).most_common():
            print(f"    {cnt}x  {msg}")

    print()
    print("=" * 80)
    print("字段名对照（模型自造 vs 真实 schema）")
    print("=" * 80)
    print(f"  真实字段: {list(ReportParse.model_fields.keys())}")
    best = min(results, key=lambda r: len(r[3]))
    print(f"  失败最少的变体: {best[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
