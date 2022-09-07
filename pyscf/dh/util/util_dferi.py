import numpy as np
from pyscf import df
from pyscf.ao2mo import _ao2mo
from pyscf.dh.util.util_helper import calc_batch_size


def get_cderi_mo(with_df: df.DF, C, Y_mo=None, pqslice=None, max_memory=2000):
    naux = with_df.get_naoaux()
    nmo = C.shape[-1]
    if pqslice is None:
        pqslice = (0, nmo, 0, nmo)
        nump, numq = nmo, nmo
    else:
        nump, numq = pqslice[1] - pqslice[0], pqslice[3] - pqslice[2]
    if Y_mo is None:
        Y_mo = np.empty((naux, nump, numq))

    p0, p1 = 0, 0
    preflop = 0 if not isinstance(Y_mo, np.ndarray) else Y_mo.size
    nbatch = calc_batch_size(2*nump*numq, max_memory, preflop)
    for Y_ao in with_df.loop(nbatch):
        p1 = p0 + Y_ao.shape[0]
        Y_mo[p0:p1] = _ao2mo.nr_e2(Y_ao, C, pqslice, aosym="s2", mosym="s1").reshape(p1-p0, nump, numq)
        p0 = p1
    return Y_mo
