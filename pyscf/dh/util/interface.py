from pyscf import dh
import numpy as np
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
    return ops


default_options = get_default_options()
"""
Default options collected from various definitions.
"""


def get_caller_locals():
    """ Get locals variables of caller function (3 layers upper). """
    return inspect.stack()[-2][0].f_locals


def sanity_dimension(array, shape, caller_locals, weak=False):
    """ Sanity check for array dimension.

    Parameters
    ----------
    array : np.ndarray
        The data to be checked. Should have attribute ``shape``.
    shape : tuple[int]
        Shape of data to be checked.
    weak : bool
        If weak, then only check size of array; otherwise, check dimension
        shape. Default to False.
    """
    for key, val in caller_locals.items():
        if id(array) == id(val):
            if not weak:
                if array.shape != shape:
                    raise ValueError(
                        "Dimension sanity check: {:} is not {:}"
                        .format(key, shape))
            else:
                if np.prod(array.shape) != np.prod(shape):
                    pass
                raise ValueError(
                    "Dimension sanity check: Size of {:} is not {:}"
                    .format(np.prod(array.shape), np.prod(array.shape)))
            return
    raise ValueError("Array in dimension sanity check does not included in "
                     "upper caller function.")
