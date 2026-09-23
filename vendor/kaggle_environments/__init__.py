# Minimal Kaggle Environments runtime vendored for FarmOS/Kaggriculture only.
# Derived from kaggle-environments 1.32.7 core files; avoids importing unrelated
# OpenSpiel/JAX/Flax environments and their dependency tree.

__version__ = "1.32.7-vendored-kaggriculture"

from . import errors, utils
from .agent import Agent
from .core import environments, evaluate, make, register

from .envs.kaggriculture import kaggriculture as _env

register(
    "kaggriculture",
    {
        "agents": getattr(_env, "agents", []),
        "html_renderer": getattr(_env, "html_renderer", None),
        "interpreter": getattr(_env, "interpreter"),
        "renderer": getattr(_env, "renderer"),
        "specification": getattr(_env, "specification"),
    },
)

__all__ = [
    "Agent",
    "environments",
    "errors",
    "evaluate",
    "make",
    "register",
    "utils",
    "__version__",
]
