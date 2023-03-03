import typing

from pyscf.dh import util
from pyscf import dft, lib
import numpy as np

if typing.TYPE_CHECKING:
    from pyscf.dh.energy import RDH
    from pyscf import gto


def driver_energy_dh(mf_dh):
    """ Driver of multiple exchange-correlation energy component evaluation.

    Parameters
    ----------
    mf_dh : RDH
        Object of doubly hybrid (restricted).
    """
    xc = mf_dh.xc
    log = mf_dh.log
    if mf_dh.mf.mo_coeff is None:
        log.warn("SCF object is not initialized. Build DH object (run SCF) first.")
        mf_dh.build()
    xc_info = util.parse_dh_xc_code(xc, is_scf=False)
    xc_hyb = util.extract_xc_code_low_rung(xc_info)
    ni = dft.numint.NumInt()
    result = dict()
    eng_tot = 0.
    # 0. noxc part
    result.update(mf_dh.kernel_energy_noxc(mf_dh.mf, mf_dh.make_rdm1_scf()))
    eng_tot += result["eng_noxc"]
    # 1. parse energy of xc_hyb
    # exact exchange
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc_hyb)
    if abs(omega) > 1e-10:
        result.update(mf_dh.kernel_energy_exactx(mf_dh.mf, mf_dh.make_rdm1_scf(), omega))
        eng_tot += (alpha - hyb) * result["eng_LR_HF({:})".format(omega)]
    if abs(hyb) > 1e-10:
        result.update(mf_dh.kernel_energy_exactx(mf_dh.mf, mf_dh.make_rdm1_scf()))
        eng_tot += hyb * result["eng_HF".format(omega)]
    # general xc
    if xc_hyb != "":
        grids = mf_dh.mf.grids
        rho = get_rho(mf_dh.mol, grids, mf_dh.make_rdm1_scf())
        result.update(mf_dh.kernel_energy_purexc([xc_hyb], rho, grids.weights, mf_dh.restricted))
        eng_tot += result["eng_purexc_{:}".format(xc_hyb)]
    # 2. other correlation
    for info in xc_info:
        if info["low_rung"]:
            continue
        xc_key, xc_param = info["name"], info["parameters"]
        if xc_key == "MP2":
            with mf_dh.params.temporary_flags({"coef_os": xc_param[0], "coef_ss": xc_param[1]}):
                mf_dh.driver_energy_mp2()
                eng_tot += mf_dh.params.results["eng_MP2"]
        elif xc_key in ["MP2CR", "MP2CR2", "IEPA", "SIEPA"]:
            with mf_dh.params.temporary_flags({
                    "coef_os": xc_param[0], "coef_ss": xc_param[1], "iepa_scheme": xc_key}):
                mf_dh.driver_energy_iepa()
                eng_tot += mf_dh.params.results["eng_{:}".format(xc_key)]
        elif xc_key == "VV10":
            fac, nlc_pars = info["fac"], info["parameters"]
            grids = mf_dh.mf.grids
            nlcgrids = mf_dh.mf.nlcgrids
            result.update(
                mf_dh.kernel_energy_vv10(mf_dh.mol, mf_dh.make_rdm1_scf(), nlc_pars, grids, nlcgrids,
                                         verbose=mf_dh.verbose))
            eng_vv10 = result["eng_VV10({:}; {:})".format(*nlc_pars)]
            eng_tot += fac * eng_vv10
        else:
            raise KeyError("Unknown XC component {:}!".format(xc_key))
    # finalize
    result["eng_dh_{:}".format(xc)] = eng_tot
    log.note("[RESULT] Energy of xc {:}: {:20.12f}".format(xc, eng_tot))
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
        result["eng_HF"] = ex
    else:
        result["eng_LR_HF({:})".format(omega)] = ex
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


def kernel_energy_vv10(mol, dm, nlc_pars, grids=None, nlcgrids=None, verbose=lib.logger.NOTE):
    log = lib.logger.new_logger(verbose=verbose)
    if grids is None:
        log.warn("VV10 grids not found. Use default grids of PySCF for VV10.")
        grids = dft.Grids(mol).build()
    rho = get_rho(mol, grids, dm)
    if nlcgrids is None:
        nlcgrids = grids
        vvrho = rho
    else:
        nlcgrids.build()
        vvrho = get_rho(mol, nlcgrids, dm)
    # handle unrestricted case
    if len(rho.shape) == 3:
        rho = rho[0] + rho[1]
        vvrho = vvrho[0] + vvrho[1]
    exc_vv10, _ = dft.numint._vv10nlc(rho, grids.coords, vvrho, nlcgrids.weights, nlcgrids.coords, nlc_pars)
    eng_vv10 = (rho[0] * grids.weights * exc_vv10).sum()
    result = dict()
    result["eng_VV10({:}; {:})".format(*nlc_pars)] = eng_vv10
    return result
