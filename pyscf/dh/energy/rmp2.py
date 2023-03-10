from pyscf.dh import util

from pyscf import ao2mo, lib
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from pyscf.dh.energy import RDH


def driver_energy_rmp2(mf_dh):
    """ Driver of MP2 energy.

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
    mf_dh : RDH
        Restricted doubly hybrid object.

    Returns
    -------
    RDH

    Notes
    -----
    See also ``pyscf.mp.mp2.kernel``.

    This function does not make checks, such as SCF convergence.
    """
    mol = mf_dh.mol
    frac_num = mf_dh.params.flags["frac_num"]
    # parse frozen orbitals
    mask_act = mf_dh.get_mask_act()
    nact, nOcc, nVir = mf_dh.nact, mf_dh.nOcc, mf_dh.nVir
    mo_coeff_act = mf_dh.mo_coeff_act
    mo_energy_act = mf_dh.mo_energy_act
    frac_num_f = frac_num[mask_act] if frac_num else None
    omega_list = mf_dh.params.flags["omega_list_mp2"]
    integral_scheme = mf_dh.params.flags["integral_scheme"].lower()
    for omega in omega_list:
        # parse t_ijab
        t_ijab_name = "t_ijab" if omega == 0 else "t_ijab_omega({:6f})".format(omega)
        params = mf_dh.params
        max_memory = mol.max_memory - lib.current_memory()[0]
        incore_t_ijab = util.parse_incore_flag(
            params.flags["incore_t_ijab"], nOcc**2 * nVir**2,
            max_memory, dtype=mo_coeff_act.dtype)
        if incore_t_ijab is None:
            t_ijab = None
        else:
            t_ijab = params.tensors.create(
                name=t_ijab_name, shape=(nOcc, nOcc, nVir, nVir),
                incore=incore_t_ijab, dtype=mo_coeff_act.dtype)
        # MP2 kernels
        if integral_scheme.startswith("conv"):
            # Conventional MP2
            eri_or_mol = mf_dh.mf._eri if omega == 0 else mol
            with mol.with_range_coulomb(omega):
                results = kernel_energy_rmp2_conv_full_incore(
                    mo_energy_act, mo_coeff_act, eri_or_mol, nOcc, nVir,
                    t_ijab=t_ijab,
                    frac_num=frac_num_f,
                    verbose=mf_dh.verbose)
            if omega != 0:
                results = {key + "_omega({:.6f})".format(omega): val for (key, val) in results}
            mf_dh.params.update_results(results)
        elif integral_scheme.startswith("ri"):
            # RI MP2
            with_df = mf_dh.get_with_df_omega(omega)
            Y_OV = mf_dh.get_MO_cholesky_eri(
                slc=(mf_dh.nCore, mf_dh.nocc, mf_dh.nocc, mf_dh.nocc + mf_dh.nVir),
                with_df=with_df)
            # Y_OV_2 is rarely called, so do not try to build omega for this special case
            Y_OV_2 = None
            if mf_dh.with_df_2 is not None:
                Y_OV_2 = mf_dh.get_MO_cholesky_eri(
                    slc=(mf_dh.nCore, mf_dh.nocc, mf_dh.nocc, mf_dh.nocc + mf_dh.nVir),
                    with_df=mf_dh.with_df_2)
            results = kernel_energy_rmp2_ri(
                mo_energy_act, Y_OV,
                t_ijab=t_ijab,
                frac_num=frac_num_f,
                verbose=mf_dh.verbose,
                max_memory=mol.max_memory - lib.current_memory()[0],
                Y_OV_2=Y_OV_2)
            if omega != 0:
                results = {key + "_omega({:.6f})".format(omega): val for (key, val) in results}
            mf_dh.params.update_results(results)
        else:
            raise NotImplementedError("Not implemented currently!")
    return mf_dh


def kernel_energy_rmp2_conv_full_incore(
        mo_energy, mo_coeff, eri_or_mol,
        nocc, nvir,
        t_ijab=None,
        frac_num=None, verbose=lib.logger.NOTE):
    """ Kernel of restricted MP2 energy by conventional method.

    Parameters
    ----------
    mo_energy : np.ndarray
        Molecular orbital energy levels.
    mo_coeff : np.ndarray
        Molecular coefficients.
    eri_or_mol : np.ndarray or gto.Mole
        ERI that is recognized by ``pyscf.ao2mo.general``.

    nocc : int
        Number of occupied orbitals.
    nvir : int
        Number of virtual orbitals.

    t_ijab : np.ndarray
        Store space for ``t_ijab``
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
    log.debug("Start ao2mo")
    g_iajb = ao2mo.general(eri_or_mol, (Co, Cv, Co, Cv)).reshape(nocc, nvir, nocc, nvir)

    # loops
    eng_bi1 = eng_bi2 = 0
    for i in range(nocc):
        log.debug("MP2 loop i: {:}".format(i))
        g_Iajb = g_iajb[i]
        D_Ijab = eo[i] + eo[:, None, None] - ev[None, :, None] - ev[None, None, :]
        t_Ijab = lib.einsum("ajb, jab -> jab", g_Iajb, 1 / D_Ijab)
        if t_ijab is not None:
            t_ijab[i] = t_Ijab
        if frac_num is not None:
            n_Ijab = frac_occ[i] * frac_occ[:, None, None] \
                * (1 - frac_vir[None, :, None]) * (1 - frac_vir[None, None, :])
            eng_bi1 += lib.einsum("jab, jab, ajb ->", n_Ijab, t_Ijab.conj(), g_Iajb)
            eng_bi2 += lib.einsum("jab, jab, bja ->", n_Ijab, t_Ijab.conj(), g_Iajb)
        else:
            eng_bi1 += lib.einsum("jab, ajb ->", t_Ijab.conj(), g_Iajb)
            eng_bi2 += lib.einsum("jab, bja ->", t_Ijab.conj(), g_Iajb)
    eng_bi1 = util.check_real(eng_bi1)
    eng_bi2 = util.check_real(eng_bi2)
    log.debug("MP2 energy computation finished.")
    # report
    eng_os = eng_bi1
    eng_ss = eng_bi1 - eng_bi2
    eng_mp2 = eng_os + eng_ss
    # results
    results = dict()
    results["eng_bi1"] = eng_bi1
    results["eng_bi2"] = eng_bi2
    results["eng_MP2_OS"] = eng_os
    results["eng_MP2_SS"] = eng_ss
    results["eng_MP2"] = eng_mp2
    log.info("[RESULT] Energy MP2 of same-spin: {:18.10f}".format(eng_ss))
    log.info("[RESULT] Energy MP2 of oppo-spin: {:18.10f}".format(eng_os))
    log.info("[RESULT] Energy MP2 of total: {:18.10f}".format(eng_mp2))
    return results


def kernel_energy_rmp2_ri(
        mo_energy, Y_OV,
        t_ijab=None,
        frac_num=None, verbose=lib.logger.NOTE, max_memory=2000, Y_OV_2=None):
    """ Kernel of MP2 energy by RI integral.

    For RI approximation, ERI integral is set to be

    .. math::
        g_{ij}^{ab} &= (ia|jb) = Y_{ia, P} Y_{jb, P}

    Parameters
    ----------
    mo_energy : np.ndarray
        Molecular orbital energy levels.
    Y_OV : np.ndarray
        Cholesky decomposed 3c2e ERI in MO basis (occ-vir part).

    t_ijab : np.ndarray
        Store space for ``t_ijab``
    frac_num : np.ndarray
        Fractional occupation number list.
    verbose : int
        Verbose level for PySCF.
    max_memory : float
        Allocatable memory in MB.
    Y_OV_2 : np.ndarray
        Another part of 3c2e ERI in MO basis (occ-vir part). This is mostly used in magnetic computations.

    Notes
    -----
    For energy of fractional occupation system, computation method is chosen
    by eq (7) from Su2016 (10.1021/acs.jctc.6b00197).

    Since conventional integral of MP2 is computationally costly,
    purpose of this function should only be benchmark.
    """
    log = lib.logger.new_logger(verbose=verbose)

    naux, nocc, nvir = Y_OV.shape
    if frac_num:
        frac_occ, frac_vir = frac_num[:nocc], frac_num[nocc:]
    else:
        frac_occ = frac_vir = None
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:]

    # loops
    log.debug("Start RI-MP2 loop")
    nbatch = util.calc_batch_size(4 * nocc * nvir ** 2, max_memory, dtype=Y_OV.dtype)
    eng_bi1 = eng_bi2 = 0
    for sI in util.gen_batch(0, nocc, nbatch):
        log.debug("MP2 loop i: [{:}, {:})".format(sI.start, sI.stop))
        if Y_OV_2 is None:
            g_Iajb = lib.einsum("PIa, Pjb -> Iajb", Y_OV[:, sI], Y_OV)
        else:
            g_Iajb = 0.5 * lib.einsum("PIa, Pjb -> Iajb", Y_OV[:, sI], Y_OV_2)
            g_Iajb += 0.5 * lib.einsum("PIa, Pjb -> Iajb", Y_OV_2[:, sI], Y_OV)
        D_Ijab = eo[sI, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
        t_Ijab = lib.einsum("Iajb, Ijab -> Ijab", g_Iajb, 1 / D_Ijab)
        if t_ijab is not None:
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
    log.debug("MP2 energy computation finished.")
    # report
    eng_os = eng_bi1
    eng_ss = eng_bi1 - eng_bi2
    eng_mp2 = eng_os + eng_ss
    # results
    results = dict()
    results["eng_bi1"] = eng_bi1
    results["eng_bi2"] = eng_bi2
    results["eng_MP2_OS"] = eng_os
    results["eng_MP2_SS"] = eng_ss
    results["eng_MP2"] = eng_mp2
    log.info("[RESULT] Energy MP2 of same-spin: {:18.10f}".format(eng_ss))
    log.info("[RESULT] Energy MP2 of oppo-spin: {:18.10f}".format(eng_os))
    log.info("[RESULT] Energy MP2 of total: {:18.10f}".format(eng_mp2))
    return results
