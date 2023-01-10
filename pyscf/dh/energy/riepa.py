from pyscf.dh import util

from pyscf import lib
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
    mol = mf_dh.mol
    mo_energy = mf_dh.mo_energy
    mo_coeff = mf_dh.mo_coeff
    nao, nmo, nocc = mf_dh.nao, mf_dh.nmo, mf_dh.nocc
    c_c = mf_dh.params.flags["coef_mp2"]
    c_os = mf_dh.params.flags["coef_mp2_os"]
    c_ss = mf_dh.params.flags["coef_mp2_ss"]
    # parse frozen orbitals
    frozen_rule = mf_dh.params.flags["frozen_rule"]
    frozen_list = mf_dh.params.flags["frozen_list"]
    mask_act = util.parse_frozen_list(mol, nmo, frozen_list, frozen_rule)
    nmo_f = mask_act.sum()
    nocc_f = mask_act[:nocc].sum()
    mo_coeff_f = mo_coeff[:, mask_act]
    mo_energy_f = mo_energy[mask_act]
    # generate ri-eri
    Y_ov_f = util.get_cderi_mo(
        mf_dh.df_ri, mo_coeff_f, None, (0, nocc_f, nocc_f, nmo_f),
        mol.max_memory - lib.current_memory()[0])
    results = kernel_energy_riepa_ri(
        mf_dh.params, mo_energy_f, Y_ov_f,
        c_c=c_c, c_os=c_os, c_ss=c_ss,
        verbose=mf_dh.verbose
    )
    mf_dh.params.update_results(results)


def kernel_energy_riepa_ri(
        params, mo_energy, Y_ov,
        c_c=1., c_os=1., c_ss=1., screen_func=erfc,
        thresh=1e-10, max_cycle=64,
        verbose=None):
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
    Y_ov : np.ndarray
        Cholesky decomposed 3c2e ERI in MO basis (occ-vir part).

    c_c : float
        MP2 contribution coefficient.
    c_os : float
        MP2 opposite-spin contribution coefficient.
    c_ss : float
        MP2 same-spin contribution coefficient.
    screen_func : function
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
    - ``norm_METHOD``: normalization factors of ``mp2cr`` or ``mp2cr2``.
    """
    log = lib.logger.new_logger(verbose=verbose)
    naux, nocc, nvir = Y_ov.shape
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:]

    # parse IEPA schemes
    # `iepa_schemes` option is either str or list[str]; change to list
    if not isinstance(params.flags["iepa_scheme"], str):
        iepa_schemes = [i.lower() for i in params.flags["iepa_scheme"]]
    else:
        iepa_schemes = [params.flags["iepa_scheme"].lower()]

    # check IEPA scheme sanity
    check_iepa_scheme = set(iepa_schemes).difference(["mp2", "mp2cr", "mp2cr2", "dcpt2", "iepa", "siepa"])
    if len(check_iepa_scheme) != 0:
        raise ValueError("Several schemes are not recognized IEPA schemes: " + ", ".join(check_iepa_scheme))
    log.info("[INFO] Recognized IEPA schemes: " + ", ".join(iepa_schemes))

    # allocate pair energies
    for scheme in iepa_schemes:
        params.tensors.create("pair_{:}_aa".format(scheme), shape=(nocc, nocc))
        params.tensors.create("pair_{:}_ab".format(scheme), shape=(nocc, nocc))
        if scheme in ["mp2cr", "mp2cr2"]:
            if "pair_mp2_aa" not in params.tensors:
                params.tensors.create("pair_mp2_aa", shape=(nocc, nocc))
                params.tensors.create("pair_mp2_ab", shape=(nocc, nocc))
            if "n2_pair_aa" not in params.tensors:
                params.tensors.create("n2_pair_aa", shape=(nocc, nocc))
                params.tensors.create("n2_pair_ab", shape=(nocc, nocc))

    # In evaluation of MP2/cr or MP2/cr2, MP2 pair energy is evaluated first.
    schemes_for_pair = set(iepa_schemes)
    if "mp2cr" in schemes_for_pair or "mp2cr2" in schemes_for_pair:
        schemes_for_pair.difference_update(["mp2cr", "mp2cr2"])
        schemes_for_pair.add("mp2")
    # scratch tensor of - e_a - e_b
    D_ab = - ev[:, None] - ev[None, :]
    # main driver
    for I in range(nocc):
        for J in range(I + 1):
            log.debug1("In IEPA kernel, pair ({:}, {:})".format(I, J))
            D_IJab = eo[I] + eo[J] + D_ab
            g_IJab = Y_ov[:, I].T @ Y_ov[:, J]  # PIa, PJb -> IJab
            g_IJab_asym = g_IJab - g_IJab.T
            # evaluate pair energy for different schemes
            for scheme in schemes_for_pair:
                pair_aa = params.tensors["pair_{:}_aa".format(scheme)]
                pair_ab = params.tensors["pair_{:}_ab".format(scheme)]
                if scheme == "mp2":
                    e_pair_os = get_pair_mp2(g_IJab, D_IJab, 1)
                    e_pair_ss = get_pair_mp2(g_IJab_asym, D_IJab, 0.5)
                elif scheme == "dcpt2":
                    e_pair_os = get_pair_dcpt2(g_IJab, D_IJab, 1)
                    e_pair_ss = get_pair_dcpt2(g_IJab_asym, D_IJab, 0.5)
                elif scheme == "iepa":
                    e_pair_os = get_pair_iepa(g_IJab, D_IJab, 1, thresh=thresh, max_cycle=max_cycle)
                    e_pair_ss = get_pair_iepa(g_IJab_asym, D_IJab, 0.5, thresh=thresh, max_cycle=max_cycle)
                elif scheme == "siepa":
                    e_pair_os = get_pair_siepa(g_IJab, D_IJab, 1,
                                               screen_func=screen_func, thresh=thresh, max_cycle=max_cycle)
                    e_pair_ss = get_pair_siepa(g_IJab_asym, D_IJab, 0.5,
                                               screen_func=screen_func, thresh=thresh, max_cycle=max_cycle)
                else:
                    assert False
                pair_aa[I, J] = pair_aa[J, I] = e_pair_ss
                pair_ab[I, J] = pair_ab[J, I] = e_pair_os
            # MP2/cr methods require norm
            if "mp2cr" in iepa_schemes or "mp2cr2" in iepa_schemes:
                n2_aa = params.tensors["n2_pair_aa"]
                n2_ab = params.tensors["n2_pair_ab"]
                n2_aa[I, J] = n2_aa[J, I] = ((g_IJab_asym / D_IJab)**2).sum()
                n2_ab[I, J] = n2_ab[J, I] = ((g_IJab / D_IJab)**2).sum()

    # process MP2/cr afterwards
    # MP2/cr I
    if "mp2cr" in iepa_schemes:
        n2_aa = params.tensors["n2_pair_aa"]
        n2_ab = params.tensors["n2_pair_ab"]
        norm = get_rmp2cr_norm(n2_aa, n2_ab)
        params.tensors["norm_mp2cr"] = norm
        params.tensors["pair_mp2cr_aa"] = params.tensors["pair_mp2_aa"] / norm
        params.tensors["pair_mp2cr_ab"] = params.tensors["pair_mp2_ab"] / norm
    # MP2/cr II
    if "mp2cr2" in iepa_schemes:
        n2_aa = params.tensors["n2_pair_aa"]
        n2_ab = params.tensors["n2_pair_ab"]
        norm = get_rmp2cr2_norm(n2_aa, n2_ab)
        params.tensors["norm_mp2cr2"] = norm
        params.tensors["pair_mp2cr2_aa"] = params.tensors["pair_mp2_aa"] * norm
        params.tensors["pair_mp2cr2_ab"] = params.tensors["pair_mp2_ab"] * norm

    # Finalize energy evaluation
    results = dict()
    for scheme in iepa_schemes:
        eng_aa = 0.5 * params.tensors["pair_{:}_aa".format(scheme)].sum()
        eng_ab = params.tensors["pair_{:}_ab".format(scheme)].sum()
        eng_tot = c_c * (c_os * eng_ab + 2 * c_ss * eng_aa)
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
    screen_func : function
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
