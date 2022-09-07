import numpy as np


def calc_batch_size(unit_flop, mem_avail, pre_flop=0):
    # mem_avail: in MB
    if unit_flop == 0: return 1
    max_memory = 0.8 * mem_avail - pre_flop * 8 / 1024 ** 2
    batch_size = int(max(max_memory // (unit_flop * 8 / 1024 ** 2), 1))
    return batch_size


def gen_leggauss_0_inf(ngrid):
    x, w = np.polynomial.legendre.leggauss(ngrid)
    return 0.5 * (1 + x) / (1 - x), w / (1 - x)**2


def gen_leggauss_0_1(ngrid):
    x, w = np.polynomial.legendre.leggauss(ngrid)
    return 0.5 * (x + 1), 0.5 * w
