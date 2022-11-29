from pyscf import dh
import os
import importlib.util
import enum
import inspect


def get_default_options_from_module(module):
    op = {}
    for var_name, var_val in inspect.getmembers(module):
        if var_name.startswith("__") or var_name == "Enum":
            continue
        if isinstance(var_val, enum.EnumMeta):
            op[var_name] = var_val.default.value
        else:
            op[var_name] = var_val
    return op


def get_default_options():
    ops = {}
    for path in os.walk(dh.__path__[0]):
        if "options.py" in path[-1]:
            spec = importlib.util.spec_from_file_location("options", path[0] + "/options.py")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            ops.update(get_default_options_from_module(m))
    return ops.copy()


default_options = get_default_options()
"""
Default options collected from various definitions.
"""
