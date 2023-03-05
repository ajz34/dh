from pyscf.dh import util
from pyscf.dh.energy.riepa import get_pair_mp2, get_pair_iepa, get_pair_siepa, get_pair_dcpt2

from pyscf import lib, ao2mo
import numpy as np
from scipy.special import erfc
import typing
import warnings

if typing.TYPE_CHECKING:
    from pyscf.dh.energy import UDH


def driver_energy_uiepa(mf_dh):
    """ Driver of pair occupied energy methods (unrestricted).

    Parameters
    ----------
    mf_dh : UDH
        Unrestricted doubly hybrid object.

    Returns
    -------
    UDH

    See Also
    --------
    pyscf.dh.energy.riepa.driver_energy_riepa
    """
    flags = mf_dh.params.flags
    log = mf_dh.log
    c_os = flags["coef_os"]
    c_ss = flags["coef_ss"]
    mo_energy_act = mf_dh.mo_energy_act
    nOcc = mf_dh.nOcc
    # kernel
    if flags["integral_scheme"].lower().startswith("ri"):
        # generate ri-eri
        Y_OV = mf_dh.get_Y_OV()

        def gen_g_IJab(s0, s1, i, j):
            return Y_OV[s0][:, i].T @ Y_OV[s1][:, j]
    elif flags["integral_scheme"].lower().startswith("conv"):
        log.warn("Conventional integral of MP2 is not recommended!\n"
                 "Use density fitting approximation is recommended.")
        eri_or_mol = mf_dh.mf._eri
        if eri_or_mol is None:
            eri_or_mol = mf_dh.mol
        nVir = mf_dh.nVir
        mo_coeff_act = mf_dh.mo_coeff_act
        CO = [mo_coeff_act[s][:, :nOcc[s]] for s in (0, 1)]
        CV = [mo_coeff_act[s][:, nOcc[s]:] for s in (0, 1)]
        g_iajb = [np.array([])] * 3
        for s0, s1, ss in zip((0, 0, 1), (0, 1, 1), (0, 1, 2)):
            g_iajb[ss] = ao2mo.general(eri_or_mol, (CO[s0], CV[s0], CO[s1], CV[s1])) \
                              .reshape(nOcc[s0], nVir[s0], nOcc[s1], nVir[s1])
            log.debug("Spin {:}{:} ao2mo finished".format(s0, s1))

        def gen_g_IJab(s0, s1, i, j):
            if (s0, s1) == (0, 0):
                return g_iajb[0][i, :, j]
            elif (s0, s1) == (1, 1):
                return g_iajb[2][i, :, j]
            elif (s0, s1) == (0, 1):
                return g_iajb[1][i, :, j]
            elif (s0, s1) == (1, 0):
                return g_iajb[1][j, :, i].T
            else:
                raise ValueError("Not accepted spin!")
    else:
        raise NotImplementedError

    results = kernel_energy_uiepa(
        mf_dh.params, mo_energy_act, gen_g_IJab, nOcc,
        c_os=c_os, c_ss=c_ss,
        screen_func=mf_dh.siepa_screen,
        verbose=mf_dh.verbose
    )
    mf_dh.params.update_results(results)


def kernel_energy_uiepa(
        params, mo_energy, gen_g_IJab, nocc,
        c_os=1., c_ss=1., screen_func=erfc,
        thresh=1e-10, max_cycle=64,
        verbose=lib.logger.NOTE):
    """ Kernel of restricted IEPA-like methods.

    Parameters of these methods are controled by flags.

    Parameters
    ----------
    params : util.Params
        (flag and intermediates)
        Flags will choose how pair energy is evaluated.
        Tensors will be updated to store pair energies and norms (MP2/cr).
    mo_energy : list[np.ndarray]
        Molecular orbital energy levels.
    gen_g_IJab : callable
        Generate ERI block :math:`(ij|ab)` where :math:`i, j` is specified.
        Function signature should be ``gen_g_IJab(s0: int, s1: int, i: int, j: int) -> np.ndarray``
        with shape of returned array (a, b) and spin of s0 (i, a) and spin of s1 (j, b).
    nocc : list[int]
        Number of occupied molecular orbitals.

    c_os : float
        MP2 opposite-spin contribution coefficient.
    c_ss : float
        MP2 same-spin contribution coefficient.
    screen_func : callable
        Function used in screened IEPA. Default is erfc, as applied in functional ZRPS.
    thresh : float
        Threshold of pair energy convergence for IEPA or sIEPA methods.
    max_cycle : int
        Maximum iteration number of energy convergence for IEPA or sIEPA methods.
    verbose : int
        Verbose level for PySCF.

    See Also
    --------
    pyscf.dh.energy.riepa.kernel_energy_riepa_ri
    """
    log = lib.logger.new_logger(verbose=verbose)
    eo = [mo_energy[s][:nocc[s]] for s in (0, 1)]
    ev = [mo_energy[s][nocc[s]:] for s in (0, 1)]

    # parse IEPA schemes
    # `iepa_schemes` option is either str or list[str]; change to list
    if not isinstance(params.flags["iepa_scheme"], str):
        iepa_schemes = [i.upper() for i in params.flags["iepa_scheme"]]
    else:
        iepa_schemes = [params.flags["iepa_scheme"].upper()]
    check_iepa_scheme = set(iepa_schemes).difference(["MP2", "MP2CR", "DCPT2", "IEPA", "SIEPA"])
    if len(check_iepa_scheme) != 0:
        if "MP2CR2" in check_iepa_scheme:
            warnings.warn("MP2/cr II is not available for unrestricted methods!")
        raise ValueError("Several schemes are not recognized IEPA schemes: " + ", ".join(check_iepa_scheme))
    log.info("[INFO] Recognized IEPA schemes: " + ", ".join(iepa_schemes))

    # allocate pair energies
    for scheme in iepa_schemes:
        params.tensors.create("pair_{:}_aa".format(scheme), shape=(nocc[0], nocc[0]))
        params.tensors.create("pair_{:}_ab".format(scheme), shape=(nocc[0], nocc[1]))
        params.tensors.create("pair_{:}_bb".format(scheme), shape=(nocc[1], nocc[1]))
        if scheme == "MP2CR":
            if "pair_MP2_aa" not in params.tensors:
                params.tensors.create("pair_MP2_aa", shape=(nocc[0], nocc[0]))
                params.tensors.create("pair_MP2_ab", shape=(nocc[0], nocc[1]))
                params.tensors.create("pair_MP2_bb", shape=(nocc[1], nocc[1]))
            params.tensors.create("n2_pair_aa", shape=(nocc[0], nocc[0]))
            params.tensors.create("n2_pair_ab", shape=(nocc[0], nocc[1]))
            params.tensors.create("n2_pair_bb", shape=(nocc[1], nocc[1]))

    # In evaluation of MP2/cr, MP2 pair energy is evaluated first.
    schemes_for_pair = set(iepa_schemes)
    if "MP2CR" in schemes_for_pair:
        schemes_for_pair.difference_update(["MP2CR"])
        schemes_for_pair.add("MP2")

    for ssn, s0, s1 in [("aa", 0, 0), ("ab", 0, 1), ("bb", 1, 1)]:
        log.debug("In IEPA kernel, spin {:}".format(ssn))
        is_same_spin = s0 == s1
        D_ab = - ev[s0][:, None] - ev[s1][None, :]
        for I in range(nocc[s0]):
            maxJ = I if is_same_spin else nocc[s1]
            for J in range(maxJ):
                log.debug("In IEPA kernel, pair ({:}, {:})".format(I, J))
                D_IJab = eo[s0][I] + eo[s1][J] + D_ab
                g_IJab = gen_g_IJab(s0, s1, I, J)  # Y_OV[s0][:, I].T @ Y_OV[s1][:, J]
                if is_same_spin:
                    g_IJab = g_IJab - g_IJab.T
                # evaluate pair energy for different schemes
                for scheme in schemes_for_pair:
                    pair_mat = params.tensors["pair_{:}_{:}".format(scheme, ssn)]
                    scale = 0.5 if is_same_spin else 1
                    if scheme == "MP2":
                        e_pair = get_pair_mp2(g_IJab, D_IJab, scale)
                    elif scheme == "DCPT2":
                        e_pair = get_pair_dcpt2(g_IJab, D_IJab, scale)
                    elif scheme == "IEPA":
                        e_pair = get_pair_iepa(g_IJab, D_IJab, scale, thresh=thresh, max_cycle=max_cycle)
                    elif scheme == "SIEPA":
                        e_pair = get_pair_siepa(g_IJab, D_IJab, scale,
                                                screen_func=screen_func, thresh=thresh, max_cycle=max_cycle)
                    else:
                        assert False
                    pair_mat[I, J] = e_pair
                    if is_same_spin:
                        pair_mat[J, I] = e_pair
                if "MP2CR" in iepa_schemes:
                    n2_mat = params.tensors["n2_pair_{:}".format(ssn)]
                    n2_val = ((g_IJab / D_IJab)**2).sum()
                    n2_mat[I, J] = n2_val
                    if is_same_spin:
                        n2_mat[J, I] = n2_val

    # process MP2/cr afterwards
    if "MP2CR" in iepa_schemes:
        n2_aa = params.tensors["n2_pair_aa"]
        n2_ab = params.tensors["n2_pair_ab"]
        n2_bb = params.tensors["n2_pair_bb"]
        norms = get_ump2cr_norm(n2_aa, n2_ab, n2_bb)
        params.tensors["norm_MP2CR_aa"], params.tensors["norm_MP2CR_ab"], params.tensors["norm_MP2CR_bb"] = norms
        params.tensors["pair_MP2CR_aa"] = params.tensors["pair_MP2_aa"] / norms[0]
        params.tensors["pair_MP2CR_ab"] = params.tensors["pair_MP2_ab"] / norms[1]
        params.tensors["pair_MP2CR_bb"] = params.tensors["pair_MP2_bb"] / norms[2]

    # Finalize energy evaluation
    results = dict()
    for scheme in iepa_schemes:
        eng_os = eng_ss = 0
        for ssn in ("aa", "ab", "bb"):
            is_same_spin = ssn[0] == ssn[1]
            scale = 0.25 if is_same_spin else 1
            eng_pair = scale * params.tensors["pair_{:}_{:}".format(scheme, ssn)].sum()
            results["eng_{:}_{:}".format(scheme, ssn)] = eng_pair
            log.info("[RESULT] Energy {:} of spin {:}: {:18.10f}".format(scheme, ssn, eng_pair))
            if is_same_spin:
                eng_ss += eng_pair
            else:
                eng_os += eng_pair
        eng_tot = c_os * eng_os + 2 * c_ss * eng_ss
        results["eng_{:}".format(scheme)] = eng_tot
        log.info("[RESULT] Energy {:} of total: {:18.10f}".format(scheme, eng_tot))
    return results


def get_ump2cr_norm(n2_aa, n2_ab, n2_bb):
    """ Comput Norm of MP2/cr (unrestricted). """
    nocc = n2_ab.shape
    # case 1: i, j -> alpha, beta
    np_ab = np.ones((nocc[0], nocc[1]))
    np_ab += 0.5 * (n2_ab.sum(axis=1)[:, None] + n2_ab.sum(axis=0)[None, :])
    np_ab += 0.25 * (n2_aa.sum(axis=1)[:, None] + n2_bb.sum(axis=0)[None, :])
    # case 2: i, j -> alpha, alpha
    np_aa = np.ones((nocc[0], nocc[0]))
    np_aa += 0.5 * (n2_ab.sum(axis=1)[:, None] + n2_ab.sum(axis=1)[None, :])
    np_aa += 0.25 * (n2_aa.sum(axis=1)[:, None] + n2_aa.sum(axis=0)[None, :])
    # case 3: i, j -> beta, beta
    np_bb = np.ones((nocc[1], nocc[1]))
    np_bb += 0.5 * (n2_ab.sum(axis=0)[:, None] + n2_ab.sum(axis=0)[None, :])
    np_bb += 0.25 * (n2_bb.sum(axis=1)[:, None] + n2_bb.sum(axis=0)[None, :])
    return np_aa, np_ab, np_bb

