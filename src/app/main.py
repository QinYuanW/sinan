"""司南 · 应用入口

一个多身份学习规划平台：
    目标设定 -> 长线规划 -> 每日作战清单 -> 打卡/汇报 -> 动态重规划

启动方式：
    python -m app.main          （开发模式，热重载）
    或 uvicorn app.main:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.api.v1 import chat, log, oss, plan, profile
from app.common.logger import setup_logging
from app.db.models import init_db

# 初始化日志与数据库（建表幂等）
setup_logging()
init_db()

app = FastAPI(
    title="司南 · 学习规划平台 API",
    description="多身份学习规划 Agent：规划、打卡、汇报、动态重规划",
    version="2.0.0",
)

# 1. 跨域：前端可能跑在不同端口或浏览器扩展环境
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议收敛到具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 1.1 业务层异常 -> 合理的 HTTP 状态码
# 身份包的 goal_schema 校验失败若直接冒泡会变成 500，
# 那是服务端错误码，会把"用户填错"误报成"服务挂了"。
@app.exception_handler(ValidationError)
async def _validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "目标字段校验失败",
            "detail": [
                {"field": ".".join(str(p) for p in err.get("loc", [])), "msg": err.get("msg")}
                for err in exc.errors()
            ],
        },
    )


@app.exception_handler(KeyError)
async def _key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(
        status_code=400,
        content={"error": "未知身份包", "detail": str(exc).strip("'\"")},
    )

# 2. 业务路由（放在静态挂载之前，保证 API 优先匹配）
app.include_router(profile.router, prefix="/api/v1", tags=["身份与画像"])
app.include_router(plan.router, prefix="/api/v1", tags=["计划"])
app.include_router(log.router, prefix="/api/v1", tags=["打卡与汇报"])
app.include_router(chat.router, prefix="/api/v1", tags=["对话"])
app.include_router(oss.router, prefix="/api/v1", tags=["文件上传"])

# 3. 前端静态资源
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# 4. SPA fallback —— 只处理非 API 请求
@app.get("/{path:path}", include_in_schema=False)
async def serve_frontend(path: str):
    if path.startswith("api/"):
        return JSONResponse({"error": "Not Found"}, status_code=404)

    file_path = os.path.join(static_dir, path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)

    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)

    return {"message": "司南 API 已启动，但前端静态资源未就绪", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    # 启动命令：python -m app.main
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=True)
