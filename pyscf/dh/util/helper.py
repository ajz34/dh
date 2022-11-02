from pyscf import lib
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
    mem_avail : int
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
    return batch_size


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


def restricted_biorthogonalize(t_ijab, cc, c_os, c_ss):
    """
    Biorthogonalize MP2 amplitude for restricted case.

    .. math::
        T_{ij}^{ab} = c_\\mathrm{c} \\big( c_\\mathrm{OS} t_{ij}^{ab} + c_\\mathrm{SS} (t_{ij}^{ab} - t_{ij}^{ba})
        \\big)

    Parameters
    ----------
    t_ijab : np.ndarray
        MP2 amplitude tensor.
    cc : float
        Coefficient of MP2 contribution.
    c_os : float
        Coefficient of MP2 opposite-spin contribution.
    c_ss : float
        Coefficient of MP2 same-spin contribution.

    Returns
    -------
    np.ndarray

    Notes
    -----
    Object of this function is simple. However, numpy's tensor transpose is notoriously slow.
    This function serves an API that can perform such kind of work in parallel efficiently.
    """
    # TODO: Efficiency may be further improved.
    coef_0 = cc * (c_os + c_ss)
    coef_1 = - cc * c_ss
    # handle different situations
    if abs(coef_1) < 1e-7:  # SS, do not make transpose
        return coef_0 * t_ijab
    else:
        t_shape = t_ijab.shape
        t_ijab = t_ijab.reshape((-1, t_ijab.shape[-2], t_ijab.shape[-1]))
        res = lib.transpose(t_ijab, axes=(0, 2, 1)).reshape(t_shape)
        t_ijab = t_ijab.reshape(t_shape)
        res *= coef_1
        res += coef_0 * t_ijab
        return res
