import importlib.util
import sys

import torch


def compile_if_available(fn):
    if sys.platform == "win32" or importlib.util.find_spec("triton") is None:
        return fn
    return torch.compile(fn)
