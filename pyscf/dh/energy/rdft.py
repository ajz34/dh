import typing

from pyscf.dh import util
from pyscf import dft, lib
import numpy as np

from pyscf.dh.util import XCType

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
    mf_dh.build()
    ni = dft.numint.NumInt()
    result = dict()
    eng_tot = 0.
    # 0. noxc part
    result.update(mf_dh.kernel_energy_noxc(mf_dh.scf, mf_dh.make_rdm1_scf()))
    eng_tot += result["eng_noxc"]
    # 1. parse energy of xc_hyb
    # exact exchange contributions
    xc_exx = xc.xc_eng.extract_by_xctype(XCType.EXX)
    for info in xc_exx:
        log.info("[INFO] EXX to be evaluated: {:}".format(info.token))
        if info.name == "HF":
            result.update(mf_dh.kernel_energy_exactx(mf_dh.scf, mf_dh.make_rdm1_scf()))
            eng = result["eng_HF"]
            log.note("[RESULT] eng_HF {:20.12f}".format(eng))
            eng_tot += info.fac * eng
        else:
            assert info.name == "LR_HF"
            assert len(info.parameters) == 1
            omega = info.parameters[0]
            result.update(mf_dh.kernel_energy_exactx(mf_dh.scf, mf_dh.make_rdm1_scf(), omega))
            eng = result["eng_LR_HF({:})".format(omega)]
            log.note("[RESULT] eng_LR_HF({:}) {:20.12f}".format(omega, eng))
            eng_tot += info.fac * eng
    # general xc
    xc_general = xc.xc_eng.extract_by_xctype(XCType.RUNG_LOW).extract_by_xctype(lambda t: not (t & XCType.EXX))
    token = xc_general.token
    if len(token) > 0:
        # pure contribution
        log.info("DFT integral XC to be evaluated: {:}".format(token))
        grids = mf_dh.scf.grids
        rho = get_rho(mf_dh.mol, grids, mf_dh.make_rdm1_scf())
        result.update(mf_dh.kernel_energy_purexc([token], rho, grids.weights, mf_dh.restricted))
        eng = result["eng_purexc_{:}".format(token)]
        eng_tot += eng
        log.note("[RESULT] eng_purexc_{:} {:20.12f}".format(token, eng))
        # exchange contribution
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(token)
        if abs(hyb) > 1e-10:
            if "eng_HF" in result:
                eng = result["eng_HF"]
            else:
                result.update(mf_dh.kernel_energy_exactx(mf_dh.scf, mf_dh.make_rdm1_scf()))
                eng = result["eng_HF"]
                log.note("[RESULT] eng_HF {:20.12f}".format(eng))
            eng_tot += hyb * eng
        if abs(omega) > 1e-10 and abs(alpha - hyb) > 1e-10:
            if "eng_LR_HF({:})".format(omega) in result:
                eng = result["eng_LR_HF({:})".format(omega)]
            else:
                result.update(mf_dh.kernel_energy_exactx(mf_dh.scf, mf_dh.make_rdm1_scf(), omega))
                eng = result["eng_LR_HF({:})".format(omega)]
                log.note("[RESULT] eng_LR_HF({:}) {:20.12f}".format(omega, eng))
            eng_tot += (alpha - hyb) * eng

    # 2. other correlation
    # 2.1 IEPA (may also resolve evaluation of MP2)
    xc_iepa = xc.xc_eng.extract_by_xctype(XCType.IEPA)
    if len(xc_iepa) > 0:
        xc_iepa = xc.xc_eng.extract_by_xctype(XCType.IEPA | XCType.MP2)
        mf_dh.driver_energy_iepa()
        iepa_scheme = [info.name for info in xc_iepa]
        log.info("[INFO] Detected IEPAs: {:}".format(str(iepa_scheme)))
        with mf_dh.params.temporary_flags({"iepa_scheme": iepa_scheme}):
            mf_dh.driver_energy_iepa()
        for info in xc_iepa:
            eng = info.fac * (
                + info.parameters[0] * mf_dh.params.results["eng_{:}_OS".format(info.name)]
                + info.parameters[1] * mf_dh.params.results["eng_{:}_SS".format(info.name)])
            log.info("[RESULT] energy of {:} correlation: {:20.12f}".format(info.name, eng))
            eng_tot += eng
    # 2.2 MP2
    if len(xc_iepa) == 0:
        xc_mp2 = xc.xc_eng.extract_by_xctype(XCType.MP2 | XCType.RSMP2)
    else:
        xc_mp2 = xc.xc_eng.extract_by_xctype(XCType.RSMP2)
    if len(xc_mp2) > 0:
        log.info("[INFO] MP2 detected")
        # generate omega list
        # parameter of RSMP2: omega, c_os, c_ss
        omega_list = []
        for info in xc_mp2:
            if XCType.MP2 in info.type:
                omega_list.append(0)
            else:
                assert XCType.RSMP2 in info.type
                omega_list.append(info.parameters[0])
        assert len(set(omega_list)) == len(omega_list)
        # run mp2
        with mf_dh.params.temporary_flags({"omega_list_mp2": omega_list}):
            mf_dh.driver_energy_mp2()
        # parse results
        for info in xc_mp2:
            if XCType.MP2 in info.type:
                c_os, c_ss = info.parameters
                omega = 0
            else:
                omega, c_os, c_ss = info.parameters
            eng = info.fac * (
                + c_os * mf_dh.params.results[util.pad_omega("eng_MP2_OS", omega)]
                + c_ss * mf_dh.params.results[util.pad_omega("eng_MP2_SS", omega)])
            log.info("[RESULT] energy of {:} correlation: {:20.12f}".format(util.pad_omega("MP2", omega), eng))
            eng_tot += eng
    elif len(xc_mp2) > 1:
        raise ValueError("MP2 terms is larger than 1. Consider trim xc first.")
    # 2.3 VV10
    xc_vdw = xc.xc_eng.extract_by_xctype(XCType.VDW)
    for info in xc_vdw:
        # currently only implement VV10
        if info.name == "VV10":
            nlc_pars = info.parameters
            assert len(nlc_pars) == 2
            grids = mf_dh.scf.grids
            nlcgrids = mf_dh.scf.nlcgrids
            res = mf_dh.kernel_energy_vv10(
                mf_dh.mol, mf_dh.make_rdm1_scf(), nlc_pars, grids, nlcgrids,
                verbose=mf_dh.verbose)
            result.update(res)
            eng = info.fac * result["eng_VV10({:}; {:})".format(*nlc_pars)]
            log.info("[RESULT] energy of VV10({:}; {:}): {:20.12f}".format(*nlc_pars, eng))
            eng_tot += eng
        else:
            raise NotImplementedError("Currently VDW only accepts VV10!")
    # finalize
    result["eng_dh_{:}".format(xc.xc_eng.token)] = eng_tot
    log.note("[RESULT] Energy of xc {:}: {:20.12f}".format(xc.xc_eng.token, eng_tot))
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
