from pyscf.dh import util
from pyscf import ao2mo, lib
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from pyscf.dh.energy import UDH


def driver_energy_ump2(mf):
    """ Driver of unrestricted MP2 energy.

    Parameters
    ----------
    mf : UDH
        Restricted doubly hybrid object.

    Returns
    -------
    UDH

    Notes
    -----
    See also ``pyscf.mp.ump2.kernel``.

    This function does not make checks, such as SCF convergence.
    """
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    nao, nmo, nocc = mf.nao, mf.nmo, mf.nocc
    c_c = mf.params.flags["coef_mp2"]
    c_os = mf.params.flags["coef_mp2_os"]
    c_ss = mf.params.flags["coef_mp2_ss"]
    frac_num = mf.params.flags["frac_num"]
    # parse frozen orbitals
    frozen_rule = mf.params.flags["frozen_rule"]
    frozen_list = mf.params.flags["frozen_list"]
    mask_act = util.parse_frozen_list(mol, nmo, frozen_list, frozen_rule)
    if len(mask_act.shape) == 1:
        # enforce mask to be [mask_act_alpha, mask_act_beta]
        mask_act = np.array([mask_act, mask_act])
    nmo_f = [mask_act[s].sum() for s in (0, 1)]
    nocc_f = [mask_act[s][:nocc[s]].sum() for s in (0, 1)]
    nvir_f = [mask_act[s][nocc[s]:].sum() for s in (0, 1)]
    mo_coeff_f = [mo_coeff[s][:, mask_act[s]] for s in (0, 1)]
    mo_energy_f = [mo_energy[s][mask_act[s]] for s in (0, 1)]
    frac_num_f = frac_num if frac_num is None else [frac_num[s][mask_act[s]] for s in (0, 1)]
    # parse space of t_ijab
    incore_t_ijab = util.parse_incore_flag(
        mf.params.flags["incore_t_ijab"], 3 * max(nocc_f)**2 * max(nvir_f)**2,
        mol.max_memory - lib.current_memory()[0], dtype=mo_coeff_f[0].dtype)
    if incore_t_ijab is None:
        t_ijab = None
    else:
        t_ijab = [None] * 3
        for s0, s1, ss, ssn in ((0, 0, 0, "aa"), (0, 1, 1, "ab"), (1, 1, 2, "bb")):
            t_ijab[ss] = mf.params.tensors.create(
                "t_ijab_{:}".format(ssn),
                shape=(nocc_f[s0], nocc_f[s1], nvir_f[s0], nvir_f[s1]), incore=incore_t_ijab,
                dtype=mo_coeff.dtype)
    # MP2 kernels
    if mf.params.flags["integral_scheme"].lower() == "conv":
        ao_eri = mf.mf._eri
        kernel_energy_ump2_conv_full_incore(
            mo_energy_f, mo_coeff_f, ao_eri, nocc_f, nvir_f,
            t_ijab, mf.params.results,
            c_c=c_c, c_os=c_os, c_ss=c_ss,
            frac_num=frac_num_f,
            verbose=mf.verbose)
    elif mf.params.flags["integral_scheme"].lower() in ["ri", "rimp2"]:
        Y_ov_f = [util.get_cderi_mo(
            mf.df_ri, mo_coeff_f[s], None, (0, nocc_f[s], nocc_f[s], nmo_f[s]),
            mol.max_memory - lib.current_memory()[0]
        ) for s in (0, 1)]
        Y_ov_2_f = None
        if mf.df_ri_2 is not None:
            Y_ov_2_f = [util.get_cderi_mo(
                mf.df_ri_2, mo_coeff_f[s], None, (0, nocc_f[s], nocc_f[s], nmo_f[s]),
                mol.max_memory - lib.current_memory()[0]
            ) for s in (0, 1)]
        kernel_energy_ump2_ri(
            mo_energy_f, Y_ov_f, t_ijab, mf.params.results,
            c_c=c_c, c_os=c_os, c_ss=c_ss,
            frac_num=frac_num_f,
            verbose=mf.verbose,
            max_memory=mol.max_memory - lib.current_memory()[0],
            Y_ov_2=Y_ov_2_f
        )
    else:
        raise NotImplementedError("Not implemented currently!")
    return mf


def kernel_energy_ump2_conv_full_incore(
        mo_energy, mo_coeff, ao_eri,
        nocc, nvir,
        t_ijab, results,
        c_c=1., c_os=1., c_ss=1., frac_num=None, verbose=None):
    """ Kernel of unrestricted MP2 energy by conventional method.

    Parameters
    ----------
    mo_energy : list[np.ndarray]
        Molecular orbital energy levels.
    mo_coeff : list[np.ndarray]
        Molecular coefficients.
    ao_eri : np.ndarray
        ERI that is recognized by ``pyscf.ao2mo.general``.

    nocc : list[int]
        Number of occupied orbitals.
    nvir : list[int]
        Number of virtual orbitals.

    t_ijab : list[np.ndarray] or None
        (output) Amplitude of MP2. If None, this variable is not to be generated.
    results : dict
        (output) Result dictionary of Params.

    c_c : float
        MP2 contribution coefficient.
    c_os : float
        MP2 opposite-spin contribution coefficient.
    c_ss : float
        MP2 same-spin contribution coefficient.
    frac_num : list[np.ndarray]
        Fractional occupation number list.
    verbose : int
        Verbose level for PySCF.

    Notes
    -----
    For energy of fractional occupation system, computation method is chosen
    by eq (7) from Su2016 (10.1021/acs.jctc.6b00197).

    Since conventional integral of MP2 is computationally costly,
    purpose of this function should only be benchmark.

    In this program, we assume that frozen orbitals are paired. Thus,
    different size of alpha and beta MO number is not allowed.
    """
    log = lib.logger.new_logger(verbose=verbose)
    log.warn("Conventional integral of MP2 is not recommended!\n"
             "Use density fitting approximation is recommended.")

    if frac_num:
        frac_occ = [frac_num[s][:nocc[s]] for s in (0, 1)]
        frac_vir = [frac_num[s][nocc[s]:] for s in (0, 1)]
    else:
        frac_occ = frac_vir = None

    # ERI conversion
    Co = [mo_coeff[s][:, :nocc[s]] for s in (0, 1)]
    Cv = [mo_coeff[s][:, nocc[s]:] for s in (0, 1)]
    eo = [mo_energy[s][:nocc[s]] for s in (0, 1)]
    ev = [mo_energy[s][nocc[s]:] for s in (0, 1)]
    log.debug1("Start ao2mo")
    g_iajb = [np.array([])] * 3
    for s0, s1, ss in zip((0, 0, 1), (0, 1, 1), (0, 1, 2)):
        g_iajb[ss] = ao2mo.general(ao_eri, (Co[s0], Cv[s0], Co[s1], Cv[s1])) \
                          .reshape(nocc[s0], nvir[s0], nocc[s1], nvir[s1])
        log.debug1("Spin {:}{:} ao2mo finished".format(s0, s1))

    # loops
    eng_spin = np.array([0, 0, 0], dtype=mo_coeff[0].dtype)
    for s0, s1, ss in zip((0, 0, 1), (0, 1, 1), (0, 1, 2)):
        for i in range(nocc[s0]):
            g_Iajb = g_iajb[ss][i]
            D_Ijab = eo[s0][i] + eo[s1][:, None, None] - ev[s0][None, :, None] - ev[s1][None, None, :]
            t_Ijab = lib.einsum("ajb, jab -> jab", g_Iajb, 1 / D_Ijab)
            if s0 == s1:
                t_Ijab -= lib.einsum("bja, jab -> jab", g_Iajb, 1 / D_Ijab)
            if t_ijab is not None:
                t_ijab[ss][i] = t_Ijab
            if frac_num is not None:
                n_Ijab = frac_occ[s0][i] * frac_occ[s1][:, None, None] \
                    * (1 - frac_vir[s0][None, :, None]) * (1 - frac_vir[s1][None, None, :])
                eng_spin[ss] += lib.einsum("jab, jab, jab, jab ->", n_Ijab, t_Ijab.conj(), t_Ijab, D_Ijab)
            else:
                eng_spin[ss] += lib.einsum("jab, jab, jab ->", t_Ijab.conj(), t_Ijab, D_Ijab)
    eng_spin[0] *= 0.25
    eng_spin[2] *= 0.25
    eng_spin = util.check_real(eng_spin)
    eng_mp2 = c_c * (c_os * eng_spin[1] + c_ss * (eng_spin[0] + eng_spin[2]))
    results["eng_spin"] = eng_spin
    results["eng_mp2"] = eng_mp2


def kernel_energy_ump2_ri(
        mo_energy, Y_ov,
        t_ijab, results,
        c_c=1., c_os=1., c_ss=1., frac_num=None, verbose=None, max_memory=2000, Y_ov_2=None):
    """ Kernel of unrestricted MP2 energy by RI integral.

    For RI approximation, ERI integral is set to be

    .. math::
        g_{ij}^{ab} &= (ia|jb) = Y_{ia, P} Y_{jb, P}

    Parameters
    ----------
    mo_energy : list[np.ndarray]
        Molecular orbital energy levels.
    Y_ov : list[np.ndarray]
        Cholesky decomposed 3c2e ERI in MO basis (occ-vir part). Spin in (aa, bb).

    t_ijab : np.ndarray or None
        (output) Amplitude of MP2. If None, this variable is not to be generated. Spin in (aa, bb).
    results : dict
        (output) Result dictionary of Params.

    c_c : float
        MP2 contribution coefficient.
    c_os : float
        MP2 opposite-spin contribution coefficient.
    c_ss : float
        MP2 same-spin contribution coefficient.
    frac_num : list[np.ndarray]
        Fractional occupation number list.
    verbose : int
        Verbose level for PySCF.
    max_memory : float
        Allocatable memory in MB.
    Y_ov_2 : list[np.ndarray]
        Another part of 3c2e ERI in MO basis (occ-vir part). This is mostly used in magnetic computations.

    Notes
    -----
    For energy of fractional occupation system, computation method is chosen
    by eq (7) from Su2016 (10.1021/acs.jctc.6b00197).

    Since conventional integral of MP2 is computationally costly,
    purpose of this function should only be benchmark.
    """
    log = lib.logger.new_logger(verbose=verbose)
    nocc, nvir = np.array([0, 0]), np.array([0, 0])
    naux, nocc[0], nvir[0] = Y_ov[0].shape
    naux, nocc[1], nvir[1] = Y_ov[1].shape

    if frac_num:
        frac_occ = [frac_num[s][:nocc[s]] for s in (0, 1)]
        frac_vir = [frac_num[s][nocc[s]:] for s in (0, 1)]
    else:
        frac_occ = frac_vir = None

    eo = [mo_energy[s][:nocc[s]] for s in (0, 1)]
    ev = [mo_energy[s][nocc[s]:] for s in (0, 1)]

    # loops
    eng_spin = np.array([0, 0, 0], dtype=Y_ov[0].dtype)
    log.debug1("Start RI-MP2 loop")
    nbatch = util.calc_batch_size(4 * max(nocc) * max(nvir) ** 2, max_memory, dtype=Y_ov[0].dtype)
    for s0, s1, ss in zip((0, 0, 1), (0, 1, 1), (0, 1, 2)):
        log.debug1("Starting spin {:}{:}".format(s0, s1))
        for sI in util.gen_batch(0, nocc[s0], nbatch):
            log.debug1("MP2 loop i: [{:}, {:})".format(sI.start, sI.stop))
            if Y_ov_2 is None:
                g_Iajb = lib.einsum("PIa, Pjb -> Iajb", Y_ov[s0][:, sI], Y_ov[s1])
            else:
                g_Iajb = 0.5 * lib.einsum("PIa, Pjb -> Iajb", Y_ov[s0][:, sI], Y_ov_2[s1])
                g_Iajb += 0.5 * lib.einsum("PIa, Pjb -> Iajb", Y_ov_2[s0][:, sI], Y_ov[s1])
            D_Ijab = (
                + eo[s0][sI, None, None, None] + eo[s1][None, :, None, None]
                - ev[s0][None, None, :, None] - ev[s1][None, None, None, :])
            t_Ijab = lib.einsum("Iajb, Ijab -> Ijab", g_Iajb, 1 / D_Ijab)
            if s0 == s1:
                t_Ijab -= lib.einsum("Ibja, Ijab -> Ijab", g_Iajb, 1 / D_Ijab)
            if t_ijab is not None:
                t_ijab[ss][sI] = t_Ijab
            if frac_num is not None:
                n_Ijab = frac_occ[s0][sI] * frac_occ[s1][:, None, None] \
                    * (1 - frac_vir[s0][None, :, None]) * (1 - frac_vir[s1][None, None, :])
                eng_spin[ss] += lib.einsum("Ijab, Ijab, Ijab, Ijab ->", n_Ijab, t_Ijab.conj(), t_Ijab, D_Ijab)
            else:
                eng_spin[ss] += lib.einsum("Ijab, Ijab, Ijab ->", t_Ijab.conj(), t_Ijab, D_Ijab)
    eng_spin[0] *= 0.25
    eng_spin[2] *= 0.25
    eng_spin = util.check_real(eng_spin)
    eng_mp2 = c_c * (c_os * eng_spin[1] + c_ss * (eng_spin[0] + eng_spin[2]))
    results["eng_spin"] = eng_spin
    results["eng_mp2"] = eng_mp2

