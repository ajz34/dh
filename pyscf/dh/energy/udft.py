import typing

from pyscf.dh import util
from pyscf import dft, lib
import numpy as np

if typing.TYPE_CHECKING:
    from pyscf.dh.energy import RDH, UDH
    from pyscf import gto


def kernel_energy_unrestricted_exactx(mf, dm, omega=None):
    """ Evaluate exact exchange energy (for either HF and long-range).

    Parameters
    ----------
    mf : dft.rks.RKS
        SCF object that have ``get_k`` member function.
    dm : np.ndarray
        Density matrix.
    omega : float or None
        Parameter of long-range ERI integral :math:`\\mathrm{erfc} (\\omega r_{12}) / r_{12}`.

    See Also
    --------
    pyscf.dh.energy.rdft.kernel_energy_restricted_exactx
    """
    hermi = 1 if np.allclose(dm, dm.swapaxes(-1, -2).conj()) else 0
    vk = mf.get_k(dm=dm, hermi=hermi, omega=omega)
    ex = - 0.5 * np.einsum('sij, sji ->', dm, vk)
    ex = util.check_real(ex)
    # results
    result = dict()
    if omega is None:
        result["eng_exx_HF"] = ex
    else:
        result["eng_exx_LR_HF({:})".format(omega)] = ex
    return result


def kernel_energy_unrestricted_noxc(mf, dm):
    """ Evaluate energy contributions that is not exchange-correlation.

    Note that some contributions (such as vdw) is not considered.

    Parameters
    ----------
    mf : dft.rks.RKS
        SCF object that have ``get_hcore``, ``get_j`` member functions.
    dm : np.ndarray
        Density matrix.

    See Also
    --------
    pyscf.dh.energy.rdft.kernel_energy_restricted_noxc
    """
    hermi = 1 if np.allclose(dm, dm.swapaxes(-1, -2).conj()) else 0
    hcore = mf.get_hcore()
    vj = mf.get_j(dm=dm, hermi=hermi)
    eng_nuc = mf.mol.energy_nuc()
    eng_hcore = np.einsum('sij, ji ->', dm, hcore)
    eng_j = 0.5 * np.einsum('ij, ji ->', dm.sum(axis=0), vj.sum(axis=0))
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


if __name__ == '__main__':
    from pyscf import gto, dft, df
    from pyscf.dh.energy.rdft import kernel_energy_restricted_noxc, kernel_energy_restricted_exactx
    from pyscf.dh.energy import UDH, RDH
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
    mf = dft.RKS(mol, xc="wB97M-V").density_fit(df.aug_etb(mol)).run()
    mf.nlc = "VV10"
    print(mf.energy_tot(dm=mf.make_rdm1()))

    res = kernel_energy_restricted_exactx(mf, dm=mf.make_rdm1())
    print(res)
    res = kernel_energy_restricted_noxc(mf, dm=mf.make_rdm1())
    print(res)
    # mf = mf.to_uks()
    mo_r = mf.mo_coeff
    mf = dft.UKS(mol, xc="wB97M_V").density_fit(df.aug_etb(mol)).run()
    # mf.mo_coeff[0] = mf.mo_coeff[1] = mo_r
    res = kernel_energy_unrestricted_exactx(mf, dm=mf.make_rdm1())
    print(res)
    res = kernel_energy_unrestricted_noxc(mf, dm=mf.make_rdm1())
    print(res)

    mf_dh = UDH(mol, xc="wB97M_V + VV10(6.0; 0.01)").run()
    print(mf_dh.e_tot)
    # mf_dh = UDH(mf, xc="B3LYPg").run()
    # print(mf_dh.e_tot)

    mf_dh = RDH(mol, xc="XYG3").run()
    print(mf_dh.e_tot)
    mf_dh = UDH(mol, xc="XYG3").run()
    print(mf_dh.e_tot)

    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1).build()
    mf_dh = UDH(mol, xc="XYG3").run()
    print(mf_dh.e_tot)


if __name__ == '__main__':
    from pyscf import gto, dft, df
    from pyscf.dh.energy.rdft import kernel_energy_restricted_noxc, kernel_energy_restricted_exactx
    from pyscf.dh.energy import UDH, RDH

    mol = gto.Mole()
    mol.atom = """
    O  0.0  0.0  0.0
    O  0.0  0.0  1.5
    H  1.0  0.0  0.0
    H  0.0  0.7  1.0
    """
    mol.basis = "6-31G"
    mol.verbose = 0
    mol.build()
    mf = dft.UKS(mol, xc="B3LYPg").run()
    mf_dh = UDH(mol, xc="XYG3")
    mf_dh.params.flags["integral_scheme"] = "ri"
    mf_dh.run()
    print(mf_dh.e_tot)

    mf_dh = UDH(mol, xc="XYG3")
    mf_dh.params.flags["integral_scheme"] = "conv"
    mf_dh.run()
    print(mf_dh.e_tot)

    mf_dh = UDH(mf, xc="XYG3")
    mf_dh.params.flags["integral_scheme"] = "conv"
    mf_dh.run()
    print(mf_dh.e_tot)


