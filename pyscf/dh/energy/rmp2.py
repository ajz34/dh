from pyscf.dh.util import Params

from pyscf import ao2mo, lib
import numpy as np

from pyscf.dh.util import sanity_dimension
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
    mo_energy = mf.mf_s.mo_energy
    mo_coeff = mf.mf_s.mo_coeff
    nao, nmo = mo_coeff.shape
    nocc = mf.mf_s.mol.nelectron // 2
    c_c = mf.params.flags["coef_mp2"]
    c_os = mf.params.flags["coef_mp2_os"]
    c_ss = mf.params.flags["coef_mp2_ss"]
    frac_occ = mf.params.flags["frac_occ"]
    frac_vir = mf.params.flags["frac_vir"]
    if mf.params.flags["integral_scheme"].lower() == "conv":
        ao_eri = mf.mf_s._eri
        kernel_energy_mp2_conv_full_incore(
            mo_energy, mo_coeff, ao_eri, nao, nocc, nmo,
            None, mf.params.results,
            c_c=c_c, c_os=c_os, c_ss=c_ss,
            frac_occ=frac_occ, frac_vir=frac_vir,
            verbose=mf.verbose)
    else:
        raise NotImplementedError("Not implemented currently!")


def kernel_energy_mp2_conv_full_incore(
        mo_energy, mo_coeff, ao_eri, nao, nocc, nmo,
        t_ijab, results,
        c_c=1., c_os=1., c_ss=1.,
        frac_occ=None, frac_vir=None,
        verbose=None):
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

    nao : int
        Number of atomic orbitals (basis sets).
    nocc : int
        Number of occupied orbitals.
    nmo : int
        Number of molecular orbitals.

    t_ijab : np.ndarray or None
        (output) Amplitude of MP2. If None, this variable is not to
        be generated.
    results : dict
        (output) Result dictionary of Params.

    c_c : float
        MP2 contribution coefficient.
    c_os : float
        MP2 opposite-spin contribution coefficient.
    c_ss : float
        MP2 same-spin contribution coefficient.
    frac_occ : np.ndarray
        Fractional occupation number for occupied orbitals.
    frac_vir : np.ndarray
        Fractional occupation number for virtual orbitals.
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

    # Dimensions
    nvir = nmo - nocc
    sanity_dimension(mo_energy, (nmo, ), locals())
    sanity_dimension(mo_coeff, (nao, nmo), locals())
    if t_ijab:
        sanity_dimension(t_ijab, (nocc, nocc, nmo, nmo), locals())
    if frac_occ or frac_vir:
        if frac_occ:
            sanity_dimension(frac_occ, (nocc,), locals())
        else:
            frac_occ = np.ones(nocc)
        if frac_vir:
            sanity_dimension(frac_vir, (nvir,), locals())
        else:
            frac_vir = np.zeros(nvir)

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
        if frac_occ or frac_vir:
            n_Ijab = frac_occ[i] * frac_occ[:, None, None] \
                * (1 - frac_vir[None, :, None]) * (1 - frac_vir[None, None, :])
            eng_bi1 += lib.einsum("jab, jab, ajb ->", n_Ijab, t_Ijab, g_Iajb)
            eng_bi2 += lib.einsum("jab, jab, bja ->", n_Ijab, t_Ijab, g_Iajb)
        else:
            eng_bi1 += lib.einsum("jab, ajb ->", t_Ijab, g_Iajb)
            eng_bi2 += lib.einsum("jab, bja ->", t_Ijab, g_Iajb)

    # report
    eng_os = eng_bi1
    eng_ss = eng_bi1 - eng_bi2
    eng_mp2 = c_c * (c_os * eng_os + c_ss * eng_ss)
    results["eng_os"] = eng_os
    results["eng_ss"] = eng_ss
    results["eng_mp2"] = eng_mp2
