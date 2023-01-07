from pyscf.dh import util

from pyscf import ao2mo, lib
import numpy as np
from scipy.special import erfc
import typing

if typing.TYPE_CHECKING:
    from pyscf.dh.energy import RDH


def driver_energy_riepa(mf):
    """ Driver of pair occupied energy methods.

    Methods included in this pair occupied energy driver are
    - IEPA (independent electron pair approximation) and screened IEPA
    - DCPT2 (degeneracy-corrected second-order perturbation)
    - MP2/cr (enhanced second-order treatment of electron pair)
    - MP2 (as a basic pair method)

    Parameters of these methods are controled by flags.

    Parameters
    ----------
    mf : RDH
        Restricted doubly hybrid object.

    Returns
    -------
    RDH

    Notes
    -----
    This function does not make checks, such as SCF convergence.

    Calculation of this driver forces using density fitting MP2.
    """
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    nao, nmo, nocc = mf.nao, mf.nmo, mf.nocc
    c_c = mf.params.flags["coef_mp2"]
    c_os = mf.params.flags["coef_mp2_os"]
    c_ss = mf.params.flags["coef_mp2_ss"]
    # parse frozen orbitals
    frozen_rule = mf.params.flags["frozen_rule"]
    frozen_list = mf.params.flags["frozen_list"]
    mask_act = util.parse_frozen_list(mol, nmo, frozen_list, frozen_rule)
    nmo_f = mask_act.sum()
    nocc_f = mask_act[:nocc].sum()
    mo_coeff_f = mo_coeff[:, mask_act]
    mo_energy_f = mo_energy[mask_act]
    # generate ri-eri
    Y_ov_f = util.get_cderi_mo(
        mf.df_ri, mo_coeff_f, None, (0, nocc_f, nocc_f, nmo_f),
        mol.max_memory - lib.current_memory()[0])
    kernel_energy_riepa_ri(
        mf.params, mo_energy_f, Y_ov_f,
        c_c=c_c, c_os=c_os, c_ss=c_ss,
        screen_func=erfc,
        verbose=mf.verbose
    )


def kernel_energy_riepa_ri(
        params, mo_energy, Y_ov,
        c_c=1., c_os=1., c_ss=1., screen_func=erfc,
        verbose=None):
    """ Kernel of restricted IEPA-like methods.

    Parameters of these methods are controled by flags.

    Parameters
    ----------
    params : util.Params
        (flag and output) Results and Flags.
        In this kernel, flags will choose how pair energy is evaluated.
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
    log.info("[INFO] Recognized IEPA schemes: " + ", ".join(iepa_schemes))
    # generate result of pair energies
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
            params.tensors.create("norm_{:}".format(scheme), shape=(nocc, nocc))

    # scratch tensor of - e_a - e_b
    D_ab = - ev[:, None] - ev[None, :]
    # main driver
    for I in range(nocc):
        for J in range(I + 1):
            log.debug1("In IEPA kernel, pair ({:}, {:})".format(I, J))
            D_IJab = eo[I] + eo[J] + D_ab
            g_IJab = Y_ov[:, I].T @ Y_ov[:, J]  # PIa, PJb -> IJab
            g_IJab_asym = g_IJab - g_IJab.T
            for scheme in iepa_schemes:
                if scheme in ["mp2", "mp2cr", "mp2cr2"]:
                    pair_aa = params.tensors["pair_mp2_aa"]
                    pair_ab = params.tensors["pair_mp2_ab"]
                    e_pair_os = get_pair_mp2(g_IJab, D_IJab, 1)
                    e_pair_ss = get_pair_mp2(g_IJab_asym, D_IJab, 0.5)
                    pair_aa[I, J] = pair_aa[J, I] = e_pair_ss
                    pair_ab[I, J] = pair_ab[J, I] = e_pair_os
                if scheme in ["mp2cr", "mp2cr2"]:
                    n2_aa = params.tensors["n2_pair_aa"]
                    n2_ab = params.tensors["n2_pair_ab"]
                    n2_aa[I, J] = n2_aa[J, I] = ((g_IJab_asym / D_IJab)**2).sum()
                    n2_ab[I, J] = n2_ab[J, I] = ((g_IJab / D_IJab)**2).sum()

    # process MP2/cr afterwards
    # MP2/cr I
    if "mp2cr" in iepa_schemes:
        norm = params.tensors["norm_mp2cr"]
        n2_aa = params.tensors["n2_pair_aa"]
        n2_ab = params.tensors["n2_pair_ab"]
        pair_mp2_aa = params.tensors["pair_mp2_aa"]
        pair_mp2_ab = params.tensors["pair_mp2_ab"]

        n2_aa_sum, n2_ab_sum = n2_aa.sum(axis=1), n2_ab.sum(axis=1)
        norm[:] = 1
        norm += 0.5 * (n2_ab_sum[:, None] + n2_ab_sum[None, :])
        norm += 0.25 * (n2_aa_sum[:, None] + n2_aa_sum[None, :])
        pair_mp2cr_aa = pair_mp2_aa / norm
        pair_mp2cr_ab = pair_mp2_ab / norm

        params.tensors["pair_mp2cr_aa"] = pair_mp2cr_aa
        params.tensors["pair_mp2cr_ab"] = pair_mp2cr_ab
    # MP2/cr II
    if "mp2cr2" in iepa_schemes:
        norm = params.tensors["norm_mp2cr2"]
        n2_aa = params.tensors["n2_pair_aa"]
        n2_ab = params.tensors["n2_pair_ab"]
        pair_mp2_aa = params.tensors["pair_mp2_aa"]
        pair_mp2_ab = params.tensors["pair_mp2_ab"]

        n2_aa_sum, n2_ab_sum = n2_aa.sum(axis=1), n2_ab.sum(axis=1)
        norm2 = 1 + n2_ab.sum() + 0.5 * n2_aa.sum()
        norm -= 2 * (n2_ab_sum[:, None] + n2_ab_sum[None, :])
        norm -= n2_aa_sum[:, None] + n2_aa_sum[None, :]
        norm += n2_ab.diagonal()[:, None] + n2_ab.diagonal()[None, :]
        norm += 2 * n2_ab + n2_aa
        for i in range(nocc):
            norm[i, i] /= 2
            norm[i, i] -= n2_ab[i, i] + 0.5 * n2_aa[i, i]
        norm[:] = norm2 / (norm + norm2)
        pair_mp2cr2_aa = pair_mp2_aa / norm
        pair_mp2cr2_ab = pair_mp2_ab / norm

        params.tensors["pair_mp2cr2_aa"] = pair_mp2cr2_aa
        params.tensors["pair_mp2cr2_ab"] = pair_mp2cr2_ab

    for scheme in iepa_schemes:
        eng_aa = 0.5 * params.tensors["pair_{:}_aa".format(scheme)].sum()
        eng_ab = params.tensors["pair_{:}_ab".format(scheme)].sum()
        eng_tot = c_c * (c_os * eng_ab + 2 * c_ss * eng_aa)
        params.results["eng_{:}_aa".format(scheme)] = eng_aa
        params.results["eng_{:}_ab".format(scheme)] = eng_ab
        params.results["eng_{:}".format(scheme)] = eng_tot
        log.info("[RESULT] Energy {:} of same-spin: {:18.10f}".format(scheme, eng_aa))
        log.info("[RESULT] Energy {:} of oppo-spin: {:18.10f}".format(scheme, eng_ab))
        log.info("[RESULT] Energy {:} of total: {:18.10f}".format(scheme, eng_tot))


def get_pair_mp2(g_ab, D_ab, scale_e):
    """ Pair energy evaluation for MP2.

    .. math::
        e_\\mathrm{pair} = s (\\tilde g_{ij}^{ab})^2 / D_{ij}^{ab}

    :math:`s` is scale of MP2; :math:`s = 1` for oppo-spin, :math:`s = 0.5` for same-spin.

    :math:`\\tilde g_{ij}^{ab}` refers to :math:`\\langle ij || ab \\rangle`.
    For oppo-spin, :math:`\\tilde g_{ij}^{ab} = (ia|jb)`;
    for same-spin, :math:`\\tilde g_{ij}^{ab} = (ia|jb) - (ib|ja)`.
    """
    return scale_e * (g_ab * g_ab / D_ab).sum()



