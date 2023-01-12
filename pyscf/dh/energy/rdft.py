import typing

from pyscf.dh import util
from pyscf import dft
import numpy as np

if typing.TYPE_CHECKING:
    from pyscf.dh.energy import RDH
    from pyscf import gto


def driver_energy_dh(mf_dh, xc_code):
    """ Driver of multiple exchange-correlation energy component evaluation.

    Parameters
    ----------
    mf_dh : RDH
        Object of doubly hybrid (restricted).
    xc_code : str
        Exchange-correlation code. Doubly hybrid components can be included in xc code.
    """
    log = mf_dh.log
    xc_pure, xc_adv_list, xc_other_list = util.parse_dh_xc_code_string(xc_code)
    log.debug1("pure xc: " + str(xc_pure))
    log.debug1("advanced xc: " + str(xc_adv_list))
    log.debug1("other xc: " + str(xc_other_list))
    ni = dft.numint.NumInt()
    result = dict()
    eng_tot = 0.
    # 0. noxc part
    result.update(kernel_energy_restricted_noxc(mf_dh.mf, mf_dh.mf.make_rdm1()))
    eng_tot += result["eng_noxc"]
    # 1. parse energy of xc_pure
    # exact exchange
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc_pure)
    if abs(omega) > 1e-10:
        result.update(kernel_energy_restricted_exactx(mf_dh.mf, mf_dh.mf.make_rdm1(), omega))
        eng_tot += (alpha - hyb) * result["eng_lr_hf({:})".format(omega)]
    if abs(hyb) > 1e-10:
        result.update(kernel_energy_restricted_exactx(mf_dh.mf, mf_dh.mf.make_rdm1()))
        eng_tot += hyb * result["eng_hf".format(omega)]
    # general xc
    if xc_pure != "":
        grids = mf_dh.mf.grids
        rho = get_rho(mf_dh.mol, grids, mf_dh.mf.make_rdm1())
        result.update(kernel_energy_purexc([xc_pure], rho, grids.weights, mf_dh.restricted))
        eng_tot += result["eng_purexc_{:}".format(xc_pure)]
    # 2. advanced correlation (5th-rung)
    for xc_key, xc_param in xc_adv_list:
        if xc_key == "mp2":
            with mf_dh.params.temporary_flags({"coef_os": xc_param[0], "coef_ss": xc_param[1]}):
                mf_dh.driver_energy_mp2()
                result["eng_mp2"] = mf_dh.params.results["eng_mp2"]
                eng_tot += result["eng_mp2"]
        elif xc_key in ["mp2cr", "mp2cr2", "iepa", "siepa"]:
            with mf_dh.params.temporary_flags({
                    "coef_os": xc_param[0], "coef_ss": xc_param[1], "iepa_scheme": xc_key}):
                mf_dh.driver_energy_iepa()
                eng_tot += result["eng_{:}".format(xc_key)]
    # 3. other xc cases
    for xc_other in xc_other_list:
        if xc_other[0] == "vv10":
            # vv10
            fac, nlc_pars = xc_other[1:]
            grids = mf_dh.mf.grids
            rho = get_rho(mf_dh.mol, grids, mf_dh.mf.make_rdm1())
            nlcgrids = mf_dh.mf.nlcgrids
            nlcgrids.build()
            vvrho = get_rho(mf_dh.mol, nlcgrids, mf_dh.mf.make_rdm1())
            exc_vv10, _ = dft.numint._vv10nlc(rho, grids.coords, vvrho, nlcgrids.weights, nlcgrids.coords, nlc_pars)
            eng_vv10 = (rho[0] * grids.weights * exc_vv10).sum()
            result["eng_vv10({:}; {:})".format(*nlc_pars)] = eng_vv10
            eng_tot += fac * eng_vv10
        else:
            raise KeyError("Currently only VV10 is accepted as other special component of xc.")
    # finalize
    result["eng_dh_{:}".format(xc_code)] = eng_tot
    log.log("[RESULT] Energy of xc {:}: {:20.12f}".format(xc_code, eng_tot))
    mf_dh.params.update_results(result)
    return mf_dh


def kernel_energy_restricted_exactx(mf, dm, omega=None):
    """ Evaluate exact exchange energy (for either HF and long-range).

    Parameters
    ----------
    mf : dft.rks.RKS
        SCF object that have ``get_k`` member function.
    dm : np.ndarray
        Density matrix.
    omega : float or None
        Parameter of long-range ERI integral :math:`\\mathrm{erfc} (\\omega r_{12}) / r_{12}`.
    """
    hermi = 1 if np.allclose(dm, dm.T.conj()) else 0
    vk = mf.get_k(dm=dm, hermi=hermi, omega=omega)
    ex = - 0.25 * np.einsum('ij, ji ->', dm, vk)
    ex = util.check_real(ex)
    # results
    result = dict()
    if omega is None:
        result["eng_hf"] = ex
    else:
        result["eng_lr_hf({:})".format(omega)] = ex
    return result


def kernel_energy_restricted_noxc(mf, dm):
    """ Evaluate energy contributions that is not exchange-correlation.

    Note that some contributions (such as vdw) is not considered.

    Parameters
    ----------
    mf : dft.rks.RKS
        SCF object that have ``get_hcore``, ``get_j`` member functions.
    dm : np.ndarray
        Density matrix.
    """
    hermi = 1 if np.allclose(dm, dm.T.conj()) else 0
    hcore = mf.get_hcore()
    vj = mf.get_j(dm=dm, hermi=hermi)
    eng_nuc = mf.mol.energy_nuc()
    eng_hcore = np.einsum('ij, ji ->', dm, hcore)
    eng_j = 0.5 * np.einsum('ij, ji ->', dm, vj)
    eng_hcore = util.check_real(eng_hcore)
    eng_j = util.check_real(eng_j)
    eng_noxc = eng_hcore + eng_nuc + eng_j
    # results
    results = dict()
    results["eng_nuc"] = eng_nuc
    results["eng_hcore"] = eng_hcore
    results["eng_j"] = eng_j
    results["eng_noxc"] = eng_noxc
    return results


def get_rho(mol, grids, dm):
    """ Obtain density on DFT grids.

    Note that this function always give density ready for meta-GGA.
    Thus, returned density grid dimension is (6, ngrid).
    For more details, see docstring in ``pyscf.dft.numint.eval_rho``.

    This function accepts either restricted or unrestricted density matrices.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    grids : dft.grid.Grids
        DFT grids object. Dimension (nao, nao) or (nset, nao, nao).
    dm : np.ndarray
        Density matrix.

    Returns
    -------
    np.ndarray
        Density grid of dimension (6, ngrid) or (nset, 6, ngrid).
    """
    ngrid = grids.weights.size
    ni = dft.numint.NumInt()
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm)
    rho = np.empty((nset, 6, ngrid))
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, deriv=2, max_memory=2000):
        p0, p1 = p1, p1 + weight.size
        for idm in range(nset):
            rho[idm, :, p0:p1] = make_rho(idm, ao, mask, 'MGGA')
    if nset == 1:
        rho = rho[0]
    return rho


def kernel_energy_purexc(xc_list, rho, weights, restricted):
    """ Evaluate energy contributions of exchange-correlation effects.

    Note that this kernel does not count HF, LR_HF and advanced correlation into account.
    To evaluate exact exchange (HF or LR_HF), use ``kernel_energy_restricted_exactx``.

    Parameters
    ----------
    xc_list : list
        List of xc codes.
    rho : np.ndarray
        Full list of density grids. Dimension (>1, ngrid) or (nset, >1, ngrid).
    weights : np.ndarray
        Full list of DFT grid weights.
    restricted : bool
        Indicator of restricted or unrestricted of incoming rho.

    See Also
    --------
    kernel_energy_restricted_exactx
    """
    if isinstance(xc_list, str):
        xc_list = [xc_list]
    results = {}
    ni = dft.numint.NumInt()
    if restricted:
        wrho0 = rho[0] * weights
    else:
        wrho0 = rho[:, 0].sum(axis=0) * weights

    for xc in xc_list:
        spin = 0 if restricted else 1
        exc = ni.eval_xc(xc, rho, spin=spin, deriv=0)[0]
        results["eng_purexc_" + xc] = exc @ wrho0
    return results
