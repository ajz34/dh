import numpy as np
from pyscf import lib

from pyscf.dh import util


def kernel_resp_eri(params, Y_mo, nocc, hyb,
                    Y_mo_lr=None, alpha=None,
                    max_memory=2000, verbose=lib.logger.NOTE):
    """ Prepare eri evluated in response function

    Cholesky decomposed ERIs in this function should be jk form
    (density-fitting object used in SCF object).

    Parameters
    ----------
    params : util.Params
    Y_mo : np.ndarray
    nocc : int
    hyb : float
    Y_mo_lr : np.ndarray
    alpha : float
    max_memory : float
    verbose : int

    Returns
    -------
    np.ndarray
    """
    log = lib.logger.new_logger(verbose=verbose)
    naux, nmo, _ = Y_mo.shape
    nvir = nmo - nocc
    is_rsh = Y_mo_lr is not None
    einsum = lib.einsum
    so, sv = slice(0, nocc), slice(nocc, nmo)
    # sanity check
    assert Y_mo.dtype is not np.complex
    assert Y_mo.shape == (naux, nmo, nmo)
    if is_rsh:
        assert Y_mo_lr.shape == (naux, nmo, nmo)
        assert alpha is not None
    # create space bulk
    incore_resp_eri = params.flags["incore_resp_eri"]
    incore_resp_eri = util.parse_incore_flag(
        incore_resp_eri, nocc**2 * nvir**2, max_memory)
    log.debug("Actual incore strategy of incore_resp_eri: {:}".format(incore_resp_eri))
    log.debug("Creating `resp_eri`, shape {:}.".format((nvir, nocc, nvir, nocc)))
    resp_eri = params.tensors.create("resp_eri", shape=(nvir, nocc, nvir, nocc), chunk=(1, 1, nvir, nocc))
    # take some parts of cholesky eri into memory
    Y_vo = np.asarray(Y_mo[:, sv, so])
    Y_oo = np.asarray(Y_mo[:, so, so])
    Y_vo_lr = Y_oo_lr = None
    if is_rsh:
        Y_vo_lr = np.asarray(Y_mo_lr[:, sv, so])
        Y_oo_lr = np.asarray(Y_mo_lr[:, so, so])

    nbatch = util.calc_batch_size(
        nvir * naux + 3 * nocc ** 2 * nvir, max_memory, 2 * Y_mo.size)

    def save_eri_cpks(sA, buf):
        resp_eri[sA] = buf

    with lib.call_in_background(save_eri_cpks) as async_write:
        for sA in util.gen_batch(nocc, nmo, nbatch):
            sAvir = slice(sA.start - nocc, sA.stop - nocc)
            print(sAvir)
            resp_eri_buf = (
                + 4 * einsum("Pai, Pbj -> aibj", Y_vo[:, sAvir], Y_vo)
                - hyb * einsum("Paj, Pbi -> aibj", Y_vo[:, sAvir], Y_vo)
                - hyb * einsum("Pij, Pab -> aibj", Y_oo, Y_mo[:, sA, sv]))
            if is_rsh:
                resp_eri_buf += (
                    - (alpha - hyb) * einsum("Paj, Pbi -> aibj", Y_vo_lr[:, sAvir], Y_vo_lr)
                    - (alpha - hyb) * einsum("Pij, Pab -> aibj", Y_oo_lr, Y_mo_lr[:, sA, sv]))
            async_write(sAvir, resp_eri_buf)

    return dict()


def resp_Ax0_HF(si, sa, sj, sb, hyb, Y_mo,
                alpha=None, Y_mo_lr=None,
                max_memory=2000, verbose=lib.logger.NOTE):
    naux, nmo, _ = Y_mo.shape
    ni, na = si.stop - si.start, sa.stop - sa.start
    # sanity check
    is_rsh = Y_mo_lr is not None

    def resp_Ax0_HF_inner(X):
        X_shape = X.shape
        X = X.reshape((-1, X_shape[-2], X_shape[-1]))
        res = np.zeros((X.shape[0], ni, na))
        nbatch = util.calc_batch_size(nmo**2, max_memory, X.size + res.size)
        for saux in util.gen_batch(0, naux, nbatch):
            pass




