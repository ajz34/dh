from pyscf import dh, gto, scf
from pyscf.dh.util import Params, HybridDict, default_options
import numpy as np


def test_rmp2_conv():
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
    mf_s = scf.RHF(mol).run()

    mf = dh.energy.RDH(mf_s)
    mf.params = Params(default_options, HybridDict(), {})
    with mf.params.temporary_flags({"integral_scheme": "conv"}):
        mf.driver_energy_mp2()
    print(mf.params.results)
    assert np.allclose(mf.params.results["eng_mp2"], -0.12722216721906485)


def test_rmp2_conv_fc():
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ").build()
    mf_s = scf.RHF(mol).run()

    mf = dh.energy.RDH(mf_s)
    mf.params = Params(default_options, HybridDict(), {})
    mf.params.flags["frozen_rule"] = "FreezeNobleGasCore"
    with mf.params.temporary_flags({"integral_scheme": "conv"}):
        mf.driver_energy_mp2()
    print(mf.params.results)
    assert np.allclose(mf.params.results["eng_mp2"], -0.2004088295118168)


def test_rmp2_conv_giao():
    from pyscf import mp

    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()

    hcore_1_B = - 1j * (
        + 0.5 * mol.intor('int1e_giao_irjxp', 3)
        + mol.intor('int1e_ignuc', 3)
        + mol.intor('int1e_igkin', 3))
    ovlp_1_B = - 1j * mol.intor("int1e_igovlp")
    eri_1_B = -1j * (
        + np.einsum("tuvkl -> tuvkl", mol.intor('int2e_ig1'))
        + np.einsum("tkluv -> tuvkl", mol.intor('int2e_ig1')))

    mf_s = scf.RHF(mol)
    dev_xyz_B = np.array([1e-1, 2e-1, -1e-1])

    def get_hcore(mol_=mol):
        hcore_total = np.asarray(scf.rhf.get_hcore(mol_), dtype=np.complex128)
        hcore_total += np.einsum("tuv, t -> uv", hcore_1_B, dev_xyz_B)
        return hcore_total

    def get_ovlp(mol_):
        ovlp_total = np.asarray(scf.rhf.get_ovlp(mol_), dtype=np.complex128)
        ovlp_total += np.einsum("tuv, t -> uv", ovlp_1_B, dev_xyz_B)
        return ovlp_total

    mf_s.get_hcore = get_hcore
    mf_s.get_ovlp = get_ovlp
    mf_s._eri = mol.intor("int2e") + np.einsum("tuvkl, t -> uvkl", eri_1_B, dev_xyz_B)
    mf_s.run()

    mf = dh.energy.RDH(mf_s)
    mf.params = Params(default_options, HybridDict(), {})
    with mf.params.temporary_flags({"integral_scheme": "conv"}):
        mf.driver_energy_mp2()
    print(mf.params.results)
    assert np.allclose(mf.params.results["eng_mp2"], -0.12862913348211558)


