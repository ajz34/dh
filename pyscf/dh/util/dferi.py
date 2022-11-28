import numpy as np
from pyscf import df, lib
from pyscf.ao2mo import _ao2mo
from pyscf.dh.util import calc_batch_size


def get_cderi_mo(with_df, mo_coeff, Y_mo=None, pqslice=None, max_memory=2000):
    """ Get Cholesky decomposed ERI in MO basis.

    Parameters
    ----------
    with_df : df.DF
        PySCF density fitting object
    mo_coeff : np.ndarray
        Molecular orbital coefficient.
    Y_mo : np.ndarray or None
        (output) Cholesky decomposed ERI in MO basis buffer.
    pqslice : tuple[int]
        Slice of orbitals to be transformed.
    max_memory : float
        Maximum memory available.

    Returns
    -------
    np.ndarray
        Cholesky decomposed ERI in MO basis.
    """
    naux = with_df.get_naoaux()
    nmo = mo_coeff.shape[-1]
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
    aosym = "s2" if len(with_df._cderi.shape) == 2 else "s1"
    for Y_ao in with_df.loop(nbatch):
        p1 = p0 + Y_ao.shape[0]
        if Y_ao.dtype == np.double and mo_coeff.dtype == np.double:
            Y_mo[p0:p1] = _ao2mo.nr_e2(Y_ao, mo_coeff, pqslice, aosym=aosym, mosym="s1").reshape(p1 - p0, nump, numq)
        else:
            assert len(Y_ao.shape) == 3
            nao = Y_ao.shape[-1]
            Y_mo = lib.einsum(
                "Puv, ui, vj -> Pij",
                Y_ao.reshape(naux, nao, nao),
                mo_coeff[:, pqslice[0]:pqslice[1]].conj(), mo_coeff[:, pqslice[2]:pqslice[3]])
        p0 = p1
    return Y_mo
