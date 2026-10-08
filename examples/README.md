# 补课示例（B 档）

这 11 个文件对应《司南_技术账本与补课清单.md》里的 **B 档：你没学过、我替你写了的技术**。

每个文件都**能独立运行**，聚焦一个概念，不掺别的东西。

## 怎么跑

```bash
# 在项目根目录（agent_study）执行
.venv/Scripts/python.exe examples/b01_tool_factory.py
```

## 顺序建议

| 文件 | 技术 | 需要 API key | 建议顺序 |
|---|---|---|---|
| `b01_tool_factory.py` | 工具工厂（闭包绑定上下文） | 否 | ① |
| `b06_agent_cache.py` | Agent 实例缓存 + 日期键 | 否 | ② |
| `b10_lazy_singleton.py` | 懒加载单例 | 否 | ③ |
| `b11_lazy_export.py` | 模块级惰性导出 `__getattr__` | 否 | ④ |
| `b07_async_stream.py` | 异步生成器 + `asyncio.to_thread` | 否 | ⑤ |
| `b02_tool_calls_model.py` | `@tool` 内部再调一次模型 | 是 | ⑥ |
| `b03_multi_model.py` | 多模型编排（视觉 / 推理分工） | 是（含 ZHIPUAI） | ⑦ |
| `b04_structured_compare.py` | `json_mode` vs `function_calling` | 是 | ⑧ |
| `b05_thread_namespace.py` | thread_id 命名空间隔离 | 是 | ⑨ |
| `b08_sse_events.py` | SSE 事件格式 | 否（起小服务） | ⑩ |
| `b09_exception_handler.py` | FastAPI 异常处理器 | 否 | ⑪ |

除 `b02` / `b03` / `b04` / `b05` 外，其余 **7 个都不需要 API key**，可以直接跑。
建议路径：先 `b01 → b06 → b10 → b11 → b07`（建立工程直觉），
再进 `b02 → b03`（模型编排），最后 `b04 → b05`（需要多轮模型调用）。

## 已验证状态

| 文件 | 状态 |
|---|---|
| `b01` `b06` `b07` `b08` `b09` `b10` `b11` | ✅ 已实跑通过（含输出核对） |
| `b02` `b03` `b04` `b05` | ✅ 语法编译通过，配好 key 即可跑 |

> `b09` 在实跑中发现并修正了一个**示例本身的设计错误**：
> FastAPI 内置会处理 `RequestValidationError`（写在函数签名里的校验），本来就返回 422。
> 但司南的目标字段是运行时按身份包决定的，只能在函数体里手动 `model_validate()`，
> 抛的是 `pydantic.ValidationError` —— 这个**没有**内置处理器，会直接变 500。
> 示例已改成"手动校验"来复现真实场景，跑起来能看到 500 vs 422 的对比。

## 每个文件的结构

```
"""Bx · 技术名

学什么      ——  这个技术解决什么问题
对应司南    ——  项目里的文件:行号
怎么跑      ——  命令
观察点      ——  跑起来该盯什么
"""
```

## 学完的验收方式

不是"看懂了"，是**能自己写出来**：

- 每个文件末尾都有 `TODO` 区，照着改一遍
- 改完能跑通，才算过关
- 然后回答《账本》第 5 节"认领清单"里对应的问题
