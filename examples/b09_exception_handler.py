"""B9 · FastAPI 异常处理器 —— 把"用户填错"和"服务挂了"分开

学什么
    身份包用 goal_schema 动态校验用户填的目标字段（考研要考试日期、
    教师要教学周期）。校验失败会抛 pydantic.ValidationError。
    如果不注册处理器，它会冒泡成 HTTP 500 —— 明明是用户填错了，
    却告诉前端"服务器内部错误"。用户会以为服务挂了，你排查半天发现是脏数据。

    ⚠️ 一个容易踩的坑（写这个示例时实测发现的）：
       FastAPI 内置只处理 RequestValidationError —— 也就是【写在函数签名里】
       的自动校验。你把 Pydantic 模型直接写进函数签名时，本来就会得到 422，
       根本不需要自己操心。
       但司南的目标字段是【运行时按身份包决定的】（考研要考试日期、
       教师要教学周期），没法写进签名，只能在函数体里手动 model_validate()。
       这时抛的是 pydantic.ValidationError —— 它【没有】内置处理器，直接变 500。
       本示例故意用"手动校验"来复现这个真实场景。

对应司南
    src/app/main.py:48-67   ValidationError -> 422 / KeyError -> 400

怎么跑
    python examples/b09_exception_handler.py
    （需要 httpx；不需要 API key）

观察点
    同一个坏请求打两个 app：
      · app_raw    未注册处理器 -> 500 Internal Server Error
      · app_handled 注册了处理器 -> 422 且带字段级的错误说明

TODO（学完自己改）
    1) 再加一个 handler，把 HTTPException 统一成 {code, message, hint} 结构
    2) 想清楚：为什么 KeyError（未知身份包）用 400 而不是 404？

    未知身份包是参数值非法（请求体里 role 不在白名单），不是资源不存在，所以用 400 而不是 404。
    判断标准是「请求的资源（接口路径指向的东西）是否存在」。
"""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, Request,HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError


# ======================================================================
# 一个"身份包的目标 schema"—— 模拟考研身份
# ======================================================================
class KaoyanGoal(BaseModel):
    """考研身份的目标字段定义。前端据此动态生成表单。"""

    target_school: str = Field(min_length=1, description="目标院校")
    exam_date: date = Field(description="初试日期 YYYY-MM-DD")
    daily_hours: float = Field(gt=0, le=16, description="每日可学习时长（小时）")


class ProfileIn(BaseModel):
    """宽松接收：goal_payload 故意写成自由 dict 而不是 KaoyanGoal。

    因为不同身份包的目标字段完全不同，请求体的 schema 是运行时才确定的。
    所以没法把 KaoyanGoal 写进函数签名 —— 只能在函数体里手动校验。
    """

    user_id: str = "default"
    role: str = "kaoyan"
    goal_payload: dict


def validate_goal(goal_payload: dict) -> KaoyanGoal:
    """模拟身份包的业务校验：查表拿到 schema -> 手动 validate。

    注意：这里抛的是 pydantic.ValidationError，
    不是 FastAPI 路由签名自动校验抛的 RequestValidationError。
    后者有内置处理器，前者没有。
    """
    return KaoyanGoal.model_validate(goal_payload)


# ======================================================================
# A. 裸 app —— 不注册任何处理器
# ======================================================================
app_raw = FastAPI()


@app_raw.put("/profile")
def set_profile_raw(req: ProfileIn):
    validate_goal(req.goal_payload)      # 手动校验 -> 抛 pydantic.ValidationError
    return {"status": "ok"}


# ======================================================================
# B. 加了异常处理器的 app
# ======================================================================
app_handled = FastAPI()


@app_handled.exception_handler(ValidationError)
async def _validation_error_handler(request: Request, exc: ValidationError):
    """校验失败 -> 422 Unprocessable Entity。

    422 的语义是"请求格式对，但内容不合法"，正好对应"你填错了"。
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": "目标字段校验失败",
            "detail": [
                {
                    "field": ".".join(str(p) for p in err.get("loc", [])),
                    "msg": err.get("msg"),
                }
                for err in exc.errors()
            ],
        },
    )


@app_handled.exception_handler(KeyError)
async def _key_error_handler(request: Request, exc: KeyError):
    """未知身份包 -> 400（客户端传了个不存在的 role）"""
    return JSONResponse(
        status_code=400,
        content={"error": "未知身份包", "detail": str(exc).strip("'\"")},
    )


@app_handled.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    """统一 HTTPException 的响应格式为 {code, message, hint}。

    detail 可以是 str 或 dict；dict 里可带 hint 字段给前端额外提示。
    """
    detail = exc.detail
    if isinstance(detail, dict):
        message = detail.get("message", str(detail))
        hint = detail.get("hint", "")
    else:
        message = str(detail)
        hint = ""
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.status_code, "message": message, "hint": hint},
    )


@app_handled.put("/profile")
def set_profile_handled(req: ProfileIn):
    # 模拟业务层：查身份包，查不到就抛 KeyError
    if req.role not in {"kaoyan", "zhuanshengben", "teacher"}:
        raise KeyError(req.role)
    goal = validate_goal(req.goal_payload)   # 抛 ValidationError -> 被 handler 接住
    return {"status": "ok", "received": str(goal.model_dump())[:80]}


# ======================================================================
def main() -> None:
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("跳过：缺 httpx（装一下：uv add httpx）")
        return

    # raise_server_exceptions=False 才能看到 500 的响应体，
    # 否则 TestClient 会直接把异常抛给你。
    raw = TestClient(app_raw, raise_server_exceptions=False)
    handled = TestClient(app_handled, raise_server_exceptions=False)

    bad_bodies = [
        ("每日时长填了 99 小时", {"user_id": "u1", "role": "kaoyan",
                              "goal_payload": {"target_school": "某大学",
                                               "exam_date": "2026-12-26",
                                               "daily_hours": 99}}),
        ("考试日期格式不对", {"user_id": "u1", "role": "kaoyan",
                          "goal_payload": {"target_school": "某大学",
                                           "exam_date": "明年这个时候",
                                           "daily_hours": 4}}),
        ("院校名称为空", {"user_id": "u1", "role": "kaoyan",
                      "goal_payload": {"target_school": "",
                                       "exam_date": "2026-12-26",
                                       "daily_hours": 4}}),
    ]

    print("=" * 76)
    print("同一批坏请求，打两个 app")
    print("=" * 76)

    for label, body in bad_bodies:
        print(f"\n【{label}】")

        r1 = raw.put("/profile", json=body)
        print(f"  未注册处理器： status={r1.status_code}  body={r1.text[:90]}")

        r2 = handled.put("/profile", json=body)
        print(f"  已注册处理器： status={r2.status_code}  body={r2.text[:150]}")

    print()
    print("=" * 76)
    print("未知身份包（业务层抛 KeyError）")
    print("=" * 76)
    r = handled.put("/profile", json={
        "user_id": "u1", "role": "doctor",
        "goal_payload": {"target_school": "某大学",
                         "exam_date": "2026-12-26", "daily_hours": 4},
    })
    print(f"  status={r.status_code}  body={r.text[:160]}")

    print()
    print("=" * 76)
    print("正常请求做对照")
    print("=" * 76)
    r = handled.put("/profile", json={
        "user_id": "u1", "role": "kaoyan",
        "goal_payload": {"target_school": "某某大学",
                         "exam_date": "2026-12-26", "daily_hours": 6},
    })
    print(f"  status={r.status_code}  body={r.text[:120]}")

    print()
    print("=" * 76)
    print("为什么 FastAPI 自带的机制没接住这几种错误？")
    print("=" * 76)
    print("  FastAPI 内置处理器只认 RequestValidationError ——")
    print("  就是【写在函数签名里】的那种自动校验。")
    print()
    print("  但司南的目标字段随身份包变化（考研要考试日期、教师要教学周期），")
    print("  请求体 schema 是运行时才定的，写不进签名。")
    print("  只能在函数体里手动 model_validate() ——")
    print("  这时抛的是 pydantic.ValidationError，没有内置处理器，直接 500。")
    print()
    print("  所以司南 main.py 里那 20 行 handler 不是可有可无的装饰。")

    print()
    print("-" * 76)
    print("为什么这件事重要：")
    print("  · 500 = 服务端故障 = 你半夜被叫起来查日志")
    print("  · 422 = 请求内容不合法 = 前端该把字段标红")
    print("  · 400 = 请求本身有问题 = 前端传了不该传的东西")
    print()
    print("司南里这三个 handler 全在 main.py，一共不到 20 行，")
    print("但它决定了'出问题时是骂前端还是骂后端'。")

    # TODO: 加 HTTPException handler，统一错误响应体格式


if __name__ == "__main__":
    main()
