from pyscf import dh
import pkgutil
import importlib
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
    for i in pkgutil.walk_packages(dh.__path__, dh.__name__ + "."):
        p = importlib.import_module(i.name)
        if p.__name__.split(".")[-1] == "options":
            ops.update(get_default_options_from_module(p))
    return ops


default_options = get_default_options()
"""
Default options collected from various definitions.
"""
