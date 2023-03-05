from pyscf.dh import util

from pyscf import lib, ao2mo
import numpy as np
from scipy.special import erfc
import typing
import warnings

if typing.TYPE_CHECKING:
    from pyscf.dh.energy import RDH


def driver_energy_riepa(mf_dh):
    """ Driver of pair occupied energy methods (restricted).

    Methods included in this pair occupied energy driver are
    - IEPA (independent electron pair approximation)
    - sIEPA (screened IEPA, using erfc function)
    - DCPT2 (degeneracy-corrected second-order perturbation)
    - MP2/cr (enhanced second-order treatment of electron pair)
    - MP2 (as a basic pair method)

    Parameters of these methods are controled by flags.

    Parameters
    ----------
    mf_dh : RDH
        Restricted doubly hybrid object.

    Returns
    -------
    RDH

    Notes
    -----
    This function does not make checks, such as SCF convergence.

    Calculation of this driver forces using density fitting MP2.
    """
    flags = mf_dh.params.flags
    log = mf_dh.log
    c_os = flags["coef_os"]
    c_ss = flags["coef_ss"]
    # parse frozen orbitals
    mo_energy_act = mf_dh.mo_energy_act
    nOcc = mf_dh.nOcc
    # kernel
    if flags["integral_scheme"].lower().startswith("ri"):
        # generate ri-eri
        Y_OV = mf_dh.get_Y_OV()

        def gen_g_IJab(i, j):
            return Y_OV[:, i].T @ Y_OV[:, j]
    elif flags["integral_scheme"].lower().startswith("conv"):
        log.warn("Conventional integral of MP2 is not recommended!\n"
                 "Use density fitting approximation is recommended.")
        eri_or_mol = mf_dh.mf._eri
        if eri_or_mol is None:
            eri_or_mol = mf_dh.mol
        nVir = mf_dh.nVir
        mo_coeff_act = mf_dh.mo_coeff_act
        CO = mo_coeff_act[:, :nOcc]
        CV = mo_coeff_act[:, nOcc:]
        g_iajb = ao2mo.general(eri_or_mol, (CO, CV, CO, CV)).reshape(nOcc, nVir, nOcc, nVir)

        def gen_g_IJab(i, j):
            return g_iajb[i, :, j]
    else:
        raise NotImplementedError

    results = kernel_energy_riepa(
        mf_dh.params, mo_energy_act, gen_g_IJab, nOcc,
        c_os=c_os, c_ss=c_ss,
        screen_func=mf_dh.siepa_screen,
        verbose=mf_dh.verbose
    )
    mf_dh.params.update_results(results)


def kernel_energy_riepa(
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
    mo_energy : np.ndarray
        Molecular orbital energy levels.
    gen_g_IJab : callable
        Generate ERI block :math:`(ij|ab)` where :math:`i, j` is specified.
        Function signature should be ``gen_g_IJab(i: int, j: int) -> np.ndarray``
        with shape of returned array (a, b).
    nocc : int
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

    Notes
    -----
    This kernel generates several intermediate tensor entries:

    - ``pair_METHOD_aa`` and ``pair_METHOD_ab``: pair energy of specified IEPA ``METHOD``.
    - ``n2_pair_aa`` and ``n2_pair_ab``: sum of squares of tensor :math:`n_{ij} = \\sum_{ab} (t_{ij}^{ab})^2`.
    - ``norm_METHOD``: normalization factors of ``MP2CR`` or ``MP2CR2``.
    """
    log = lib.logger.new_logger(verbose=verbose)
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:]

    # parse IEPA schemes
    # `iepa_schemes` option is either str or list[str]; change to list
    if not isinstance(params.flags["iepa_scheme"], str):
        iepa_schemes = [i.upper() for i in params.flags["iepa_scheme"]]
    else:
        iepa_schemes = [params.flags["iepa_scheme"].upper()]

    # check IEPA scheme sanity
    check_iepa_scheme = set(iepa_schemes).difference(["MP2", "MP2CR", "MP2CR2", "DCPT2", "IEPA", "SIEPA"])
    if len(check_iepa_scheme) != 0:
        raise ValueError("Several schemes are not recognized IEPA schemes: " + ", ".join(check_iepa_scheme))
    log.info("[INFO] Recognized IEPA schemes: " + ", ".join(iepa_schemes))

    # allocate pair energies
    for scheme in iepa_schemes:
        params.tensors.create("pair_{:}_aa".format(scheme), shape=(nocc, nocc))
        params.tensors.create("pair_{:}_ab".format(scheme), shape=(nocc, nocc))
        if scheme in ["MP2CR", "MP2CR2"]:
            if "pair_MP2_aa" not in params.tensors:
                params.tensors.create("pair_MP2_aa", shape=(nocc, nocc))
                params.tensors.create("pair_MP2_ab", shape=(nocc, nocc))
            if "n2_pair_aa" not in params.tensors:
                params.tensors.create("n2_pair_aa", shape=(nocc, nocc))
                params.tensors.create("n2_pair_ab", shape=(nocc, nocc))

    # In evaluation of MP2/cr or MP2/cr2, MP2 pair energy is evaluated first.
    schemes_for_pair = set(iepa_schemes)
    if "MP2CR" in schemes_for_pair or "MP2CR2" in schemes_for_pair:
        schemes_for_pair.difference_update(["MP2CR", "MP2CR2"])
        schemes_for_pair.add("MP2")
    # scratch tensor of - e_a - e_b
    D_ab = - ev[:, None] - ev[None, :]
    # main driver
    for I in range(nocc):
        for J in range(I + 1):
            log.debug("In IEPA kernel, pair ({:}, {:})".format(I, J))
            D_IJab = eo[I] + eo[J] + D_ab
            g_IJab = gen_g_IJab(I, J)  # Y_OV[:, I].T @ Y_OV[:, J]
            g_IJab_asym = g_IJab - g_IJab.T
            # evaluate pair energy for different schemes
            for scheme in schemes_for_pair:
                pair_aa = params.tensors["pair_{:}_aa".format(scheme)]
                pair_ab = params.tensors["pair_{:}_ab".format(scheme)]
                if scheme == "MP2":
                    e_pair_os = get_pair_mp2(g_IJab, D_IJab, 1)
                    e_pair_ss = get_pair_mp2(g_IJab_asym, D_IJab, 0.5)
                elif scheme == "DCPT2":
                    e_pair_os = get_pair_dcpt2(g_IJab, D_IJab, 1)
                    e_pair_ss = get_pair_dcpt2(g_IJab_asym, D_IJab, 0.5)
                elif scheme == "IEPA":
                    e_pair_os = get_pair_iepa(g_IJab, D_IJab, 1, thresh=thresh, max_cycle=max_cycle)
                    e_pair_ss = get_pair_iepa(g_IJab_asym, D_IJab, 0.5, thresh=thresh, max_cycle=max_cycle)
                elif scheme == "SIEPA":
                    e_pair_os = get_pair_siepa(g_IJab, D_IJab, 1,
                                               screen_func=screen_func, thresh=thresh, max_cycle=max_cycle)
                    e_pair_ss = get_pair_siepa(g_IJab_asym, D_IJab, 0.5,
                                               screen_func=screen_func, thresh=thresh, max_cycle=max_cycle)
                else:
                    assert False
                pair_aa[I, J] = pair_aa[J, I] = e_pair_ss
                pair_ab[I, J] = pair_ab[J, I] = e_pair_os
            # MP2/cr methods require norm
            if "MP2CR" in iepa_schemes or "MP2CR2" in iepa_schemes:
                n2_aa = params.tensors["n2_pair_aa"]
                n2_ab = params.tensors["n2_pair_ab"]
                n2_aa[I, J] = n2_aa[J, I] = ((g_IJab_asym / D_IJab)**2).sum()
                n2_ab[I, J] = n2_ab[J, I] = ((g_IJab / D_IJab)**2).sum()

    # process MP2/cr afterwards
    # MP2/cr I
    if "MP2CR" in iepa_schemes:
        n2_aa = params.tensors["n2_pair_aa"]
        n2_ab = params.tensors["n2_pair_ab"]
        norm = get_rmp2cr_norm(n2_aa, n2_ab)
        params.tensors["norm_MP2CR"] = norm
        params.tensors["pair_MP2CR_aa"] = params.tensors["pair_MP2_aa"] / norm
        params.tensors["pair_MP2CR_ab"] = params.tensors["pair_MP2_ab"] / norm
    # MP2/cr II
    if "MP2CR2" in iepa_schemes:
        n2_aa = params.tensors["n2_pair_aa"]
        n2_ab = params.tensors["n2_pair_ab"]
        norm = get_rmp2cr2_norm(n2_aa, n2_ab)
        params.tensors["norm_MP2CR2"] = norm
        params.tensors["pair_MP2CR2_aa"] = params.tensors["pair_MP2_aa"] * norm
        params.tensors["pair_MP2CR2_ab"] = params.tensors["pair_MP2_ab"] * norm

    # Finalize energy evaluation
    results = dict()
    for scheme in iepa_schemes:
        eng_aa = 0.5 * params.tensors["pair_{:}_aa".format(scheme)].sum()
        eng_ab = params.tensors["pair_{:}_ab".format(scheme)].sum()
        eng_tot = c_os * eng_ab + 2 * c_ss * eng_aa
        results["eng_{:}_aa".format(scheme)] = eng_aa
        results["eng_{:}_ab".format(scheme)] = eng_ab
        results["eng_{:}".format(scheme)] = eng_tot
        log.info("[RESULT] Energy {:} of same-spin: {:18.10f}".format(scheme, eng_aa))
        log.info("[RESULT] Energy {:} of oppo-spin: {:18.10f}".format(scheme, eng_ab))
        log.info("[RESULT] Energy {:} of total: {:18.10f}".format(scheme, eng_tot))
    return results


def get_pair_mp2(g_ab, D_ab, scale_e):
    """ Pair energy evaluation for MP2.

    .. math::
        e_{ij} = s (\\tilde g_{ij}^{ab})^2 / D_{ij}^{ab}

    In this function, :math:`i, j` are defined.

    Parameters
    ----------
    g_ab : np.ndarray
        :math:`\\tilde g_{ij}^{ab}` refers to :math:`\\langle ij || ab \\rangle`.

        For oppo-spin, :math:`\\tilde g_{ij}^{ab} = (ia|jb)`;
        for same-spin, :math:`\\tilde g_{ij}^{ab} = (ia|jb) - (ib|ja)`.

        Should be matrix of indices (a, b).
    D_ab : np.ndarray
        :math:`D_{ij}^{ab}` refers to :math:`\\varepsilon_i + \\varepsilon_j - \\varepsilon_a - \\varepsilon_b`.

        Should be matrix of indices (a, b).
    scale_e : float
        :math:`s` is scale of MP2.

        Generally, :math:`s = 1` for oppo-spin, :math:`s = 0.5` for same-spin.

    Returns
    -------
    float
        Pair energy :math:`e_{ij}`.
    """
    return scale_e * (g_ab * g_ab / D_ab).sum()


def get_pair_dcpt2(g_ab, D_ab, scale_e):
    """ Pair energy evaluation for DCPT2.

    .. math::
        e_{ij} = s \\frac{1}{2} (D_{ij}^{ab} - \\sqrt{(D_{ij}^{ab})^2 + 4 (\\tilde g_{ij}^{ab})^2})

    See Also
    --------
    get_pair_mp2
    """
    return 0.5 * scale_e * (- D_ab - np.sqrt(D_ab**2 + 4 * g_ab**2)).sum()


def get_pair_siepa(g_ab, D_ab, scale_e, screen_func, thresh=1e-10, max_cycle=64):
    """ Pair energy evaluation for screened IEPA.

    .. math::
        e_{ij} = s \\frac{(\\tilde g_{ij}^{ab})^2}{D_{ij}^{ab} + s \\times \\mathrm{screen}(- D_{ij}^{ab})} e_{ij}

    Parameters
    ----------
    g_ab : np.ndarray
        :math:`\\tilde g_{ij}^{ab}` refers to :math:`\\langle ij || ab \\rangle`.
    D_ab : np.ndarray
        :math:`D_{ij}^{ab}` refers to :math:`\\varepsilon_i + \\varepsilon_j - \\varepsilon_a - \\varepsilon_b`.
    scale_e : float
        :math:`s` is scale of MP2.
    screen_func : callable
        Function used in screened IEPA. For example erfc, which is applied in functional ZRPS.
    thresh : float
        Threshold of pair energy convergence for IEPA or sIEPA methods.
    max_cycle : int
        Maximum iteration number of energy convergence for IEPA or sIEPA methods.

    Returns
    -------
    float
        Pair energy :math:`e_{ij}`.

    See Also
    --------
    get_pair_mp2
    get_pair_iepa
    """
    g2_ab = g_ab * g_ab
    sD_ab = screen_func(-D_ab)
    e = (g2_ab / D_ab).sum()
    e_old = 1e8
    n_cycle = 0
    while abs(e_old - e) > thresh and n_cycle < max_cycle:
        e_old = e
        e = scale_e * (g2_ab / (D_ab + sD_ab * e)).sum()
        n_cycle += 1
    if n_cycle >= max_cycle:
        warnings.warn("[WARN] Maximum cycle {:d} exceeded! Pair energy error: {:12.6e}"
                      .format(n_cycle, abs(e_old - e)))
    return e


def get_pair_iepa(g_ab, D_ab, scale_e, thresh=1e-10, max_cycle=64):
    """ Pair energy evaluation for IEPA.

    This procedure sets screen function to 1.

    See Also
    --------
    get_pair_mp2
    get_pair_iepa
    """
    g2_ab = g_ab * g_ab
    e = (g2_ab / D_ab).sum()
    e_old = 1e8
    n_cycle = 0
    while abs(e_old - e) > thresh and n_cycle < max_cycle:
        e_old = e
        e = scale_e * (g2_ab / (D_ab + e)).sum()
        n_cycle += 1
    if n_cycle >= max_cycle:
        warnings.warn("[WARN] Maximum cycle {:d} exceeded! Pair energy error: {:12.6e}"
                      .format(n_cycle, abs(e_old - e)))
    return e


def get_rmp2cr_norm(n2_aa, n2_ab):
    """ Comput Norm of MP2/cr (restricted). """
    nocc = n2_aa.shape[0]
    norm = np.ones((nocc, nocc))
    n2_aa_sum, n2_ab_sum = n2_aa.sum(axis=1), n2_ab.sum(axis=1)
    norm += 0.5 * (n2_ab_sum[:, None] + n2_ab_sum[None, :])
    norm += 0.25 * (n2_aa_sum[:, None] + n2_aa_sum[None, :])
    return norm


def get_rmp2cr2_norm(n2_aa, n2_ab):
    """ Comput Norm of MP2/cr II (restricted). """
    nocc = n2_aa.shape[0]
    norm = np.zeros((nocc, nocc))
    n2_aa_sum, n2_ab_sum = n2_aa.sum(axis=1), n2_ab.sum(axis=1)
    norm2 = 1 + n2_ab.sum() + 0.5 * n2_aa.sum()
    norm -= 2 * (n2_ab_sum[:, None] + n2_ab_sum[None, :])
    norm -= n2_aa_sum[:, None] + n2_aa_sum[None, :]
    norm += n2_ab.diagonal()[:, None] + n2_ab.diagonal()[None, :]
    norm += 2 * n2_ab + n2_aa
    for i in range(nocc):
        norm[i, i] /= 2
        norm[i, i] -= n2_ab[i, i] + 0.5 * n2_aa[i, i]
    norm = norm / norm2 + 1
    return norm
