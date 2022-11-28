from pyscf.dh import util

from pyscf import ao2mo, lib
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from pyscf.dh.energy import RDH


def driver_energy_mp2(mf):
    """
    Driver of MP2 energy.

    Parameters
    ----------
    mf : RDH
        Restricted doubly hybrid object.

    Returns
    -------
    RDH

    Notes
    -----
    See also ``pyscf.mp.mp2.kernel``.

    This function does not make checks, such as SCF convergence.
    """
    mol = mf.mf_s.mol
    mo_energy = mf.mf_s.mo_energy
    mo_coeff = mf.mf_s.mo_coeff
    nao, nmo = mo_coeff.shape
    nocc = mf.mf_s.mol.nelectron // 2
    nvir = nmo - nocc
    c_c = mf.params.flags["coef_mp2"]
    c_os = mf.params.flags["coef_mp2_os"]
    c_ss = mf.params.flags["coef_mp2_ss"]
    frac_num = mf.params.flags["frac_num"]
    # parse frozen orbitals
    frozen_rule = mf.params.flags["frozen_rule"]
    frozen_list = mf.params.flags["frozen_list"]
    frozen_orb, frozen_act = util.parse_frozen_list(mol, nmo, frozen_list, frozen_rule)
    nmo_f = len(frozen_act)
    nocc_f = (frozen_act < nocc).sum()
    nvir_f = nmo_f - nocc_f
    mo_coeff_f = mo_coeff[:, frozen_act]
    mo_energy_f = mo_energy[frozen_act]
    frac_num_f = frac_num[frozen_act] if frac_num else None
    # MP2 kernels
    if mf.params.flags["integral_scheme"].lower() == "conv":
        ao_eri = mf.mf_s._eri
        kernel_energy_mp2_conv_full_incore(
            mo_energy_f, mo_coeff_f, ao_eri, nocc_f, nvir_f,
            None, mf.params.results,
            c_c=c_c, c_os=c_os, c_ss=c_ss,
            frac_num=frac_num_f,
            verbose=mf.verbose)
    elif mf.params.flags["integral_scheme"].lower() in ["ri", "rimp2"]:
        Y_ov_f = util.get_cderi_mo(mf.df_ri, mo_coeff_f, None, (0, nocc_f, nocc_f, nmo_f),
                                   mol.max_memory - lib.current_memory()[0])
        Y_ov_2_f = None
        if mf.df_ri_2:
            Y_ov_2_f = util.get_cderi_mo(mf.df_ri_2, mo_coeff_f, None, (0, nocc_f, nocc_f, nmo_f),
                                         mol.max_memory - lib.current_memory()[0])
        kernel_energy_mp2_ri(
            mo_energy_f, Y_ov_f, None, mf.params.results,
            c_c=c_c, c_os=c_os, c_ss=c_ss,
            frac_num=frac_num_f,
            verbose=mf.verbose,
            max_memory=mol.max_memory - lib.current_memory()[0],
            Y_ov_2=Y_ov_2_f)
    else:
        raise NotImplementedError("Not implemented currently!")


def kernel_energy_mp2_conv_full_incore(
        mo_energy, mo_coeff, ao_eri,
        nocc, nvir,
        t_ijab, results,
        c_c=1., c_os=1., c_ss=1., frac_num=None, verbose=None):
    """ Kernel of MP2 energy by conventional method.

    .. math::
        g_{ij}^{ab} &= (ia|jb) = (\\mu \\nu | \\kappa \\lambda) C_{\\mu i} C_{\\nu a} C_{\\kappa j} C_{\\lambda b}

        D_{ij}^{ab} &= \\varepsilon_i + \\varepsilon_j - \\varepsilon_a - \\varepsilon_b

        t_{ij}^{ab} &= g_{ij}^{ab} / D_{ij}^{ab}

        n_{ij}^{ab} &= n_i n_j (1 - n_a) (1 - n_b)

        E_\\mathrm{OS} &= n_{ij}^{ab} t_{ij}^{ab} g_{ij}^{ab}

        E_\\mathrm{SS} &= n_{ij}^{ab} t_{ij}^{ab} (g_{ij}^{ab} - g_{ij}^{ba})

        E_\\mathrm{corr,MP2} &= c_\\mathrm{c} (c_\\mathrm{OS} E_\\mathrm{OS} + c_\\mathrm{SS} E_\\mathrm{SS})

    Parameters
    ----------
    mo_energy : np.ndarray
        Molecular orbital energy levels.
    mo_coeff : np.ndarray
        Molecular coefficients.
    ao_eri : np.ndarray
        ERI that is recognized by ``pyscf.ao2mo.general``.

    nocc : int
        Number of occupied orbitals.
    nvir : int
        Number of virtual orbitals.

    t_ijab : np.ndarray or None
        (output) Amplitude of MP2. If None, this variable is not to be generated.
    results : dict
        (output) Result dictionary of Params.

    c_c : float
        MP2 contribution coefficient.
    c_os : float
        MP2 opposite-spin contribution coefficient.
    c_ss : float
        MP2 same-spin contribution coefficient.
    frac_num : np.ndarray
        Fractional occupation number list.
    verbose : int
        Verbose level for PySCF.

    Notes
    -----
    For energy of fractional occupation system, computation method is chosen
    by eq (7) from Su2016 (10.1021/acs.jctc.6b00197).

    Since conventional integral of MP2 is computationally costly,
    purpose of this function should only be benchmark.
    """
    log = lib.logger.new_logger(verbose=verbose)
    log.warn("Conventional integral of MP2 is not recommended!\n"
             "Use density fitting approximation is recommended.")

    if frac_num:
        frac_occ, frac_vir = frac_num[:nocc], frac_num[nocc:]
    else:
        frac_occ = frac_vir = None

    # ERI conversion
    Co = mo_coeff[:, :nocc]
    Cv = mo_coeff[:, nocc:]
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:]
    log.debug1("Start ao2mo")
    g_iajb = ao2mo.general(ao_eri, (Co, Cv, Co, Cv)).reshape(nocc, nvir, nocc, nvir)

    # loops
    eng_bi1 = eng_bi2 = 0
    for i in range(nocc):
        log.debug1("MP2 loop i: {:}".format(i))
        g_Iajb = g_iajb[i]
        D_Ijab = eo[i] + eo[:, None, None] - ev[None, :, None] - ev[None, None, :]
        t_Ijab = lib.einsum("ajb, jab -> jab", g_Iajb, 1 / D_Ijab)
        if t_ijab:
            t_ijab[i] = t_Ijab
        if frac_num:
            n_Ijab = frac_occ[i] * frac_occ[:, None, None] \
                * (1 - frac_vir[None, :, None]) * (1 - frac_vir[None, None, :])
            eng_bi1 += lib.einsum("jab, jab, ajb ->", n_Ijab, t_Ijab.conj(), g_Iajb)
            eng_bi2 += lib.einsum("jab, jab, bja ->", n_Ijab, t_Ijab.conj(), g_Iajb)
        else:
            eng_bi1 += lib.einsum("jab, ajb ->", t_Ijab.conj(), g_Iajb)
            eng_bi2 += lib.einsum("jab, bja ->", t_Ijab.conj(), g_Iajb)
    eng_bi1 = util.check_real(eng_bi1)
    eng_bi2 = util.check_real(eng_bi2)
    log.debug1("MP2 energy computation finished.")
    # report
    eng_os = eng_bi1
    eng_ss = eng_bi1 - eng_bi2
    eng_mp2 = c_c * (c_os * eng_os + c_ss * eng_ss)
    results["eng_os"] = eng_os
    results["eng_ss"] = eng_ss
    results["eng_mp2"] = eng_mp2


def kernel_energy_mp2_ri(
        mo_energy, Y_ov,
        t_ijab, results,
        c_c=1., c_os=1., c_ss=1., frac_num=None, verbose=None, max_memory=2000, Y_ov_2=None):
    """ Kernel of MP2 energy by RI integral.

    .. math::
        g_{ij}^{ab} &= (ia|jb) = Y_{ia, P} Y_{jb, P}

        D_{ij}^{ab} &= \\varepsilon_i + \\varepsilon_j - \\varepsilon_a - \\varepsilon_b

        t_{ij}^{ab} &= g_{ij}^{ab} / D_{ij}^{ab}

        n_{ij}^{ab} &= n_i n_j (1 - n_a) (1 - n_b)

        E_\\mathrm{OS} &= n_{ij}^{ab} t_{ij}^{ab} g_{ij}^{ab}

        E_\\mathrm{SS} &= n_{ij}^{ab} t_{ij}^{ab} (g_{ij}^{ab} - g_{ij}^{ba})

        E_\\mathrm{corr,MP2} &= c_\\mathrm{c} (c_\\mathrm{OS} E_\\mathrm{OS} + c_\\mathrm{SS} E_\\mathrm{SS})

    Parameters
    ----------
    mo_energy : np.ndarray
        Molecular orbital energy levels.
    Y_ov : np.ndarray
        Cholesky decomposed 3c2e ERI in MO basis (occ-vir part).

    t_ijab : np.ndarray or None
        (output) Amplitude of MP2. If None, this variable is not to be generated.
    results : dict
        (output) Result dictionary of Params.

    c_c : float
        MP2 contribution coefficient.
    c_os : float
        MP2 opposite-spin contribution coefficient.
    c_ss : float
        MP2 same-spin contribution coefficient.
    frac_num : np.ndarray
        Fractional occupation number list.
    verbose : int
        Verbose level for PySCF.
    max_memory : float
        Allocatable memory in MB.
    Y_ov_2 : np.ndarray
        Another part of 3c2e ERI in MO basis (occ-vir part). This is mostly used in magnetic computations.

    Notes
    -----
    For energy of fractional occupation system, computation method is chosen
    by eq (7) from Su2016 (10.1021/acs.jctc.6b00197).

    Since conventional integral of MP2 is computationally costly,
    purpose of this function should only be benchmark.
    """
    log = lib.logger.new_logger(verbose=verbose)
    log.warn("Conventional integral of MP2 is not recommended!\n"
             "Use density fitting approximation is recommended.")

    naux, nocc, nvir = Y_ov.shape
    if frac_num:
        frac_occ, frac_vir = frac_num[:nocc], frac_num[nocc:]
    else:
        frac_occ = frac_vir = None
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:]

    # loops
    log.debug1("Start RI-MP2 loop")
    nbatch = util.calc_batch_size(4 * nocc * nvir ** 2, max_memory, dtype=Y_ov.dtype)
    eng_bi1 = eng_bi2 = 0
    for sI in util.gen_batch(0, nocc, nbatch):
        log.debug1("MP2 loop i: [{:}, {:})".format(sI.start, sI.stop))
        if Y_ov_2 is None:
            g_Iajb = lib.einsum("PIa, Pjb -> Iajb", Y_ov[:, sI], Y_ov)
        else:
            g_Iajb = 0.5 * lib.einsum("PIa, Pjb -> Iajb", Y_ov[:, sI], Y_ov_2)
            g_Iajb += 0.5 * lib.einsum("PIa, Pjb -> Iajb", Y_ov_2[:, sI], Y_ov)
        D_Ijab = eo[sI, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
        t_Ijab = lib.einsum("Iajb, Ijab -> Ijab", g_Iajb, 1 / D_Ijab)
        if t_ijab:
            t_ijab[sI] = t_Ijab
        if frac_num:
            n_Ijab = frac_occ[sI] * frac_occ[None, :, None, None] \
                * (1 - frac_vir[None, None, :, None]) * (1 - frac_vir[None, None, None, :])
            eng_bi1 += lib.einsum("Ijab, Ijab, Iajb ->", n_Ijab, t_Ijab.conj(), g_Iajb)
            eng_bi2 += lib.einsum("Ijab, Ijab, Ibja ->", n_Ijab, t_Ijab.conj(), g_Iajb)
        else:
            eng_bi1 += lib.einsum("Ijab, Iajb ->", t_Ijab.conj(), g_Iajb)
            eng_bi2 += lib.einsum("Ijab, Ibja ->", t_Ijab.conj(), g_Iajb)
    eng_bi1 = util.check_real(eng_bi1)
    eng_bi2 = util.check_real(eng_bi2)
    log.debug1("MP2 energy computation finished.")
    # report
    eng_os = eng_bi1
    eng_ss = eng_bi1 - eng_bi2
    eng_mp2 = c_c * (c_os * eng_os + c_ss * eng_ss)
    results["eng_os"] = eng_os
    results["eng_ss"] = eng_ss
    results["eng_mp2"] = eng_mp2
