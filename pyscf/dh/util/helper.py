import inspect

import numpy as np
import warnings

from typing import List


def calc_batch_size(unit_flop, mem_avail, pre_flop=0, dtype=float, min_batch=1):
    """ Calculate batch size within possible memory.

    For example, if we want to compute tensor (100, 100, 100), but only 50,000 memory available,
    then this tensor should be splited into 20 batches.

    ``flop`` in parameters is number of data, not refers to FLOPs.

    Parameters
    ----------
    unit_flop : int
        Number of data for unit operation.

        For example, for a tensor with shape (110, 120, 130), the 1st dimension is indexable from
        outer programs, then a unit operation handles 120x130 = 15,600 data. Then we call this function
        with ``unit_flop = 15600``.

        This value will be set to 1 if too small.
    mem_avail : float
        Memory available in MB.
    pre_flop : int
        Number of data preserved in memory. Unit in number.
    dtype : type
        Type of data. Such as np.float64, complex, etc.
    min_batch : int
        Minimum value of batch.

        If this value set to 0 (by default), then when memory overflow
        detected, an exception will be raised. Otherwise, only a warning
        will be raised and try to fill memory as possible.

    Returns
    -------
    batch_size : int
        Size of one batch available for outer function iteration.
    """
    unit_flop = max(unit_flop, 1)
    unit_mb = unit_flop * np.dtype(dtype).itemsize / 1024**2
    max_mb = mem_avail - pre_flop * np.dtype(dtype).itemsize / 1024 ** 2

    if unit_mb * max(min_batch, 1) > max_mb:
        warning_token = "Memory overflow when preparing batch number. " \
                        "Current memory available {:10.3f} MB, minimum required {:10.3f} MB." \
                        .format(max_mb, unit_mb * max(min_batch, 1))
        if min_batch <= 0:
            raise ValueError(warning_token)
        else:
            warnings.warn(warning_token)

    batch_size = int(max(max_mb / unit_mb, unit_mb))
    batch_size = max(batch_size, min_batch)
    return batch_size


def parse_incore_flag(flag, unit_flop, mem_avail, pre_flop=0, dtype=float):
    """ Parse flag of whether tensor can be stored incore.

    ``flop`` in parameters is number of data, not refers to FLOPs.

    Parameters
    ----------
    flag : bool or float or None or str
        Input flag.

        - True: Store tensor in memory.
        - False: Store tensor in disk.
        - None: Store tensor nowhere.
        - "auto": Judge tensor in memory/disk by available memory.
        - (float): Judge tensor in memory/disk by given value in MB.
    unit_flop : int
        Number of data for unit operation.

        For example, for a tensor with shape (110, 120, 130), the 1st dimension is indexable from
        outer programs, then a unit operation handles 120x130 = 15,600 data. Then we call this function
        with ``unit_flop = 15600``.

        This value will be set to 1 if too small.
    mem_avail : float
        Memory available in MB.
    pre_flop : int
        Number of data preserved in memory. Unit in number.
    dtype : type
        Type of data. Such as np.float64, complex, etc.

    Returns
    -------
    True or False or None
        Output flag of whether tensor store in memory/disk/nowhere.
    """
    if flag in [False, True, None]:
        return flag
    if isinstance(flag, str) and flag.lower().strip() == "auto":
        pass
    else:  # assert flag is a number
        mem_avail = float(flag)
    unit_flop = max(unit_flop, 1)
    unit_mb = unit_flop * np.dtype(dtype).itemsize / 1024**2
    max_mb = mem_avail - pre_flop * np.dtype(dtype).itemsize / 1024 ** 2
    return unit_mb < max_mb


def gen_batch(val_min, val_max, batch_size):
    """ Generate slices given numbers of batch.

    Parameters
    ----------
    val_min : int
        Minimum value to be iterated
    val_max : int
        Maximum value to be iterated
    batch_size : int
        Batch size to be sliced.

    Returns
    -------
    List[slice]

    Examples
    --------
    >>> gen_batch(10, 20, 3)
        [slice(10, 13, None), slice(13, 16, None), slice(16, 19, None), slice(19, 20, None)]
    """
    return [slice(i, (i + batch_size) if i + batch_size < val_max else val_max)
            for i in range(val_min, val_max, batch_size)]


def gen_leggauss_0_inf(ngrid):
    x, w = np.polynomial.legendre.leggauss(ngrid)
    return 0.5 * (1 + x) / (1 - x), w / (1 - x)**2


def gen_leggauss_0_1(ngrid):
    x, w = np.polynomial.legendre.leggauss(ngrid)
    return 0.5 * (x + 1), 0.5 * w


def check_real(var, rtol=1e-5, atol=1e-8):
    """ Check and return array or complex number is real.

    Parameters
    ----------
    var : complex or np.ndarray
        Complex value to be checked.
    rtol : float
        Relative error threshold.
    atol : float
        Absolute error threshold.

    Returns
    -------
    complex or np.ndarray
    """
    if not np.allclose(np.real(var), var, rtol=rtol, atol=atol):
        caller_locals = inspect.currentframe().f_back.f_locals
        for key, val in caller_locals.items():
            if id(var) == id(val):
                raise ValueError("Variable `{:}` is not real.".format(key))
    else:
        return np.real(var)


def sanity_dimension(array, shape, weak=False):
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
    caller_locals = inspect.currentframe().f_back.f_locals
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


def pad_omega(s, omega):
    """ Pad omega parameter ``_omega({:.6f})`` after string if RSH parameter omega is not zero.

    Padding always returns 6 float digitals.

    Parameters
    ----------
    s : str
    omega : float

    Returns
    -------
    str
    """
    if omega == 0:
        return s
    return s + "_omega({:.6f})".format(omega)
