from .train import TRAINERS
from .test import TESTERS

# 导入各个模块以确保注册器正常工作
from . import train
from . import test
from . import iseg

__all__ = ["TRAINERS", "TESTERS"]
