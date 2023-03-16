from pyscf.dh import util
from pyscf import ao2mo, lib
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from pyscf.dh.energy import UDH


def driver_energy_ump2(mf_dh):
    """ Driver of unrestricted MP2 energy.

    Parameters
    ----------
    mf_dh : UDH
        Restricted doubly hybrid object.

    Returns
    -------
    UDH

    Notes
    -----
    See also ``pyscf.mp.ump2.kernel``.

    This function does not make checks, such as SCF convergence.
    """
    mol = mf_dh.mol
    frac_num = mf_dh.params.flags["frac_num"]
    # parse frozen orbitals
    mask_act = mf_dh.get_mask_act()
    nact, nOcc, nVir = mf_dh.nact, mf_dh.nOcc, mf_dh.nVir
    mo_coeff_act = mf_dh.mo_coeff_act
    mo_energy_act = mf_dh.mo_energy_act
    frac_num_f = frac_num if frac_num is None else [frac_num[s][mask_act[s]] for s in (0, 1)]
    omega_list = mf_dh.params.flags["omega_list_mp2"]
    integral_scheme = mf_dh.params.flags["integral_scheme"].lower()
    for omega in omega_list:
        # prepare t_ijab space
        params = mf_dh.params
        max_memory = mol.max_memory - lib.current_memory()[0]
        incore_t_ijab = util.parse_incore_flag(
            params.flags["incore_t_ijab"], 3 * max(nOcc) ** 2 * max(nVir) ** 2,
            max_memory, dtype=mo_coeff_act[0].dtype)
        if incore_t_ijab is None:
            t_ijab = None
        else:
            t_ijab = [np.zeros(0)] * 3  # IDE type cheat
            for s0, s1, ss, ssn in ((0, 0, 0, "aa"), (0, 1, 1, "ab"), (1, 1, 2, "bb")):
                t_ijab[ss] = params.tensors.create(
                    name=util.pad_omega("t_ijab_{:}".format(ssn), omega),
                    shape=(nOcc[s0], nOcc[s1], nVir[s0], nVir[s1]), incore=incore_t_ijab,
                    dtype=mo_coeff_act[0].dtype)

        # MP2 kernels
        if integral_scheme.startswith("conv"):
            eri_or_mol = mf_dh.scf._eri if omega == 0 else mol
            if eri_or_mol is None:
                eri_or_mol = mol
            with mol.with_range_coulomb(omega):
                results = kernel_energy_ump2_conv_full_incore(
                    mo_energy_act, mo_coeff_act, eri_or_mol,
                    nOcc, nVir,
                    t_ijab=t_ijab,
                    frac_num=frac_num_f,
                    verbose=mf_dh.verbose)
            if omega != 0:
                results = {util.pad_omega(key, omega): val for (key, val) in results.items()}
            mf_dh.params.update_results(results)
        elif mf_dh.params.flags["integral_scheme"].lower().startswith("ri"):
            with_df = util.get_with_df_omega(mf_dh.with_df, omega)
            Y_OV = [
                params.tensors.get(util.pad_omega("Y_OV_{:}".format(sn), omega), None)
                for sn in ("a", "b")]
            if Y_OV[0] is None:
                for s, sn in [(0, "a"), (1, "b")]:
                    Y_OV[s] = util.get_cderi_mo(
                        with_df, mo_coeff_act[s], None, (0, nOcc[s], nOcc[s], nact[s]),
                        mol.max_memory - lib.current_memory()[0])
                    params.tensors[util.pad_omega("Y_OV_{:}".format(sn), omega)] = Y_OV[s]
            # Y_OV_2 is rarely called, so do not try to build omega for this special case
            Y_OV_2 = None
            if mf_dh.with_df_2 is not None:
                Y_OV_2 = [[], []]
                for s, sn in [(0, "a"), (1, "b")]:
                    Y_OV_2[s] = util.get_cderi_mo(
                        mf_dh.with_df_2, mo_coeff_act[s], None, (0, nOcc[s], nOcc[s], nact[s]),
                        mol.max_memory - lib.current_memory()[0])
            results = kernel_energy_ump2_ri(
                mo_energy_act, Y_OV,
                t_ijab=t_ijab,
                frac_num=frac_num_f,
                verbose=mf_dh.verbose,
                max_memory=mol.max_memory - lib.current_memory()[0],
                Y_OV_2=Y_OV_2
            )
            if omega != 0:
                results = {util.pad_omega(key, omega): val for (key, val) in results.items()}
            mf_dh.params.update_results(results)
        else:
            raise NotImplementedError("Not implemented currently!")
    return mf_dh


def kernel_energy_ump2_conv_full_incore(
        mo_energy, mo_coeff, eri_or_mol,
        nocc, nvir,
        t_ijab=None,
        frac_num=None, verbose=lib.logger.NOTE):
    """ Kernel of unrestricted MP2 energy by conventional method.

    Parameters
    ----------
    mo_energy : list[np.ndarray]
        Molecular orbital energy levels.
    mo_coeff : list[np.ndarray]
        Molecular coefficients.
    eri_or_mol : np.ndarray or gto.Mole
        ERI that is recognized by ``pyscf.ao2mo.general``.

    t_ijab : list[np.ndarray]
        Store space for ``t_ijab``
    nocc : list[int]
        Number of occupied orbitals.
    nvir : list[int]
        Number of virtual orbitals.

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
    log.debug("Start ao2mo")
    g_iajb = [np.array([])] * 3
    for s0, s1, ss in zip((0, 0, 1), (0, 1, 1), (0, 1, 2)):
        g_iajb[ss] = ao2mo.general(eri_or_mol, (Co[s0], Cv[s0], Co[s1], Cv[s1])) \
                          .reshape(nocc[s0], nvir[s0], nocc[s1], nvir[s1])
        log.debug("Spin {:}{:} ao2mo finished".format(s0, s1))

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
    eng_mp2 = eng_spin[1] + (eng_spin[0] + eng_spin[2])
    # finalize results
    results = dict()
    results["eng_MP2_aa"] = eng_spin[0]
    results["eng_MP2_ab"] = eng_spin[1]
    results["eng_MP2_bb"] = eng_spin[2]
    results["eng_MP2_OS"] = eng_spin[1]
    results["eng_MP2_SS"] = eng_spin[0] + eng_spin[2]
    results["eng_MP2"] = eng_mp2
    log.info("[RESULT] Energy MP2 of spin aa: {:18.10f}".format(eng_spin[0]))
    log.info("[RESULT] Energy MP2 of spin ab: {:18.10f}".format(eng_spin[1]))
    log.info("[RESULT] Energy MP2 of spin bb: {:18.10f}".format(eng_spin[2]))
    log.info("[RESULT] Energy MP2 of total: {:18.10f}".format(eng_mp2))
    return results


def kernel_energy_ump2_ri(
        mo_energy, Y_OV,
        t_ijab=None,
        frac_num=None, verbose=lib.logger.NOTE, max_memory=2000, Y_OV_2=None):
    """ Kernel of unrestricted MP2 energy by RI integral.

    For RI approximation, ERI integral is set to be

    .. math::
        g_{ij}^{ab} &= (ia|jb) = Y_{ia, P} Y_{jb, P}

    Parameters
    ----------
    mo_energy : list[np.ndarray]
        Molecular orbital energy levels.
    Y_OV : list[np.ndarray]
        Cholesky decomposed 3c2e ERI in MO basis (occ-vir part). Spin in (aa, bb).

    t_ijab : list[np.ndarray]
        Store space for ``t_ijab``
    frac_num : list[np.ndarray]
        Fractional occupation number list.
    verbose : int
        Verbose level for PySCF.
    max_memory : float
        Allocatable memory in MB.
    Y_OV_2 : list[np.ndarray]
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
    naux, nocc[0], nvir[0] = Y_OV[0].shape
    naux, nocc[1], nvir[1] = Y_OV[1].shape

    if frac_num:
        frac_occ = [frac_num[s][:nocc[s]] for s in (0, 1)]
        frac_vir = [frac_num[s][nocc[s]:] for s in (0, 1)]
    else:
        frac_occ = frac_vir = None

    eo = [mo_energy[s][:nocc[s]] for s in (0, 1)]
    ev = [mo_energy[s][nocc[s]:] for s in (0, 1)]

    # loops
    eng_spin = np.array([0, 0, 0], dtype=Y_OV[0].dtype)
    log.debug("Start RI-MP2 loop")
    nbatch = util.calc_batch_size(4 * max(nocc) * max(nvir) ** 2, max_memory, dtype=Y_OV[0].dtype)
    for s0, s1, ss in zip((0, 0, 1), (0, 1, 1), (0, 1, 2)):
        log.debug("Starting spin {:}{:}".format(s0, s1))
        for sI in util.gen_batch(0, nocc[s0], nbatch):
            log.debug("MP2 loop i: [{:}, {:})".format(sI.start, sI.stop))
            if Y_OV_2 is None:
                g_Iajb = lib.einsum("PIa, Pjb -> Iajb", Y_OV[s0][:, sI], Y_OV[s1])
            else:
                g_Iajb = 0.5 * lib.einsum("PIa, Pjb -> Iajb", Y_OV[s0][:, sI], Y_OV_2[s1])
                g_Iajb += 0.5 * lib.einsum("PIa, Pjb -> Iajb", Y_OV_2[s0][:, sI], Y_OV[s1])
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
    eng_mp2 = eng_spin[1] + (eng_spin[0] + eng_spin[2])
    # finalize results
    results = dict()
    results["eng_MP2_aa"] = eng_spin[0]
    results["eng_MP2_ab"] = eng_spin[1]
    results["eng_MP2_bb"] = eng_spin[2]
    results["eng_MP2_OS"] = eng_spin[1]
    results["eng_MP2_SS"] = eng_spin[0] + eng_spin[2]
    results["eng_MP2"] = eng_mp2
    log.info("[RESULT] Energy MP2 of spin aa: {:18.10f}".format(eng_spin[0]))
    log.info("[RESULT] Energy MP2 of spin ab: {:18.10f}".format(eng_spin[1]))
    log.info("[RESULT] Energy MP2 of spin bb: {:18.10f}".format(eng_spin[2]))
    log.info("[RESULT] Energy MP2 of total: {:18.10f}".format(eng_mp2))
    return results
