"""司南 · 身份层

导入本包即完成所有身份包的注册。
新增身份只需：
    1. 在 roles/ 下新建 xxx.py，继承 RolePack 并实现三个抽象方法
    2. 在下面的 _PACKS 中加一行
引擎代码一行都不用改。
"""

from app.agent.roles.base import (  # noqa: F401
    PlanningContext,
    PriorityFactor,
    RolePack,
    get_pack,
    has_pack,
    list_packs,
    register,
)
from app.agent.roles.kaoyan import KaoyanPack
from app.agent.roles.teacher import TeacherPack
from app.agent.roles.zhuanshengben import ZhuanshengbenPack

# 注册顺序决定 list_packs() 的展示顺序
_PACKS = [
    KaoyanPack(),
    ZhuanshengbenPack(),
    TeacherPack(),
]

for _pack in _PACKS:
    register(_pack)

__all__ = [
    "RolePack",
    "PlanningContext",
    "PriorityFactor",
    "register",
    "get_pack",
    "has_pack",
    "list_packs",
]
