import unittest
from pyscf import dh, gto, scf, df, lib, mp
import numpy as np


class TestUMP2(unittest.TestCase):
    def test_ump2_conv(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", spin=1, charge=1, basis="cc-pVTZ").build()
        mf_s = scf.UHF(mol).run()

        mf = dh.energy.UDH(mf_s, xc="MP2")
        with mf.params.temporary_flags({"integral_scheme": "conv"}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.209174918573074))

    def test_ump2_conv_fc(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", spin=1, charge=1, basis="cc-pVTZ").build()
        mf_s = scf.UHF(mol).run()

        # test default frozen core
        mf = dh.energy.UDH(mf_s, xc="MP2")
        mf.params.flags["frozen_rule"] = "FreezeNobleGasCore"
        with mf.params.temporary_flags({"integral_scheme": "conv", "incore_t_ijab": True}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.195783018787701))

        # test frozen core with different indices on alpha/beta orbitals
        with mf.params.temporary_flags({"integral_scheme": "conv", "incore_t_ijab": True,
                                        "frozen_list": [[0, 1], [0, 2]]}):
            # Note that this clear of tensors is essential.
            # Frozen list (mask) and cderi (Y_ov_f) is not regenerated by default.
            mf.params.tensors.clear()
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.10559463994349409))

        # test frozen core with different sizes of indices on alpha/beta orbitals
        with mf.params.temporary_flags({"integral_scheme": "conv", "incore_t_ijab": True,
                                        "frozen_list": [[0, 1], [2]]}):
            mf.params.tensors.clear()
            mf.run()
        print(mf.params.results)
        # self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.10820654836616792))

    def test_ump2_conv_giao(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", spin=1, charge=1, basis="cc-pVTZ").build()

        hcore_1_B = - 1j * (
            + 0.5 * mol.intor('int1e_giao_irjxp', 3)
            + mol.intor('int1e_ignuc', 3)
            + mol.intor('int1e_igkin', 3))
        ovlp_1_B = - 1j * mol.intor("int1e_igovlp")
        eri_1_B = -1j * (
            + np.einsum("tuvkl -> tuvkl", mol.intor('int2e_ig1'))
            + np.einsum("tkluv -> tuvkl", mol.intor('int2e_ig1')))

        mf_s = scf.UHF(mol)
        dev_xyz_B = np.array([1e-2, 2e-2, -1e-2])

        def get_hcore(mol_=mol):
            hcore_total = np.asarray(scf.rhf.get_hcore(mol_), dtype=np.complex128)
            hcore_total += np.einsum("tuv, t -> uv", hcore_1_B, dev_xyz_B)
            return hcore_total

        def get_ovlp(mol_=mol):
            ovlp_total = np.asarray(scf.rhf.get_ovlp(mol_), dtype=np.complex128)
            ovlp_total += np.einsum("tuv, t -> uv", ovlp_1_B, dev_xyz_B)
            return ovlp_total

        mf_s.get_hcore = get_hcore
        mf_s.get_ovlp = get_ovlp
        mf_s._eri = mol.intor("int2e") + np.einsum("tuvkl, t -> uvkl", eri_1_B, dev_xyz_B)
        mf_s.run()

        mf = dh.energy.UDH(mf_s, xc="MP2")
        with mf.params.temporary_flags({"integral_scheme": "conv"}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.209474427130422))

    def test_ump2_ri(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", spin=1, charge=1, basis="cc-pVTZ").build()
        mf_s = scf.UHF(mol).run()

        mf = dh.energy.UDH(mf_s, xc="MP2")
        mf.df_ri = df.DF(mol, df.aug_etb(mol))
        with mf.params.temporary_flags({"integral_scheme": "ri"}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.20915836544854347))

        mf = dh.energy.UDH(mf_s, xc="MP2")
        mf.df_ri = df.DF(mol, df.aug_etb(mol))
        with mf.params.temporary_flags({"integral_scheme": "ri", "incore_t_ijab": True,
                                        "frozen_list": [[0, 1], [0, 2]]}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.1055960581512246))

    def test_ump2_ri_fc(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", spin=1, charge=1, basis="cc-pVTZ").build()
        mf_s = scf.UHF(mol).run()

        mf = dh.energy.UDH(mf_s, xc="MP2")
        mf.df_ri = df.DF(mol, df.aug_etb(mol))
        mf.params.flags["frozen_rule"] = "FreezeNobleGasCore"
        with mf.params.temporary_flags({"integral_scheme": "ri"}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.19576646982349294))

    def test_rmp2_ri_giao(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", spin=1, charge=1, basis="cc-pVTZ").build()

        hcore_1_B = - 1j * (
            + 0.5 * mol.intor('int1e_giao_irjxp', 3)
            + mol.intor('int1e_ignuc', 3)
            + mol.intor('int1e_igkin', 3))
        ovlp_1_B = - 1j * mol.intor("int1e_igovlp")
        eri_1_B = -1j * (
            + np.einsum("tuvkl -> tuvkl", mol.intor('int2e_ig1'))
            + np.einsum("tkluv -> tuvkl", mol.intor('int2e_ig1')))

        mf_s = scf.UHF(mol)
        dev_xyz_B = np.array([1e-2, 2e-2, -1e-2])

        def get_hcore(mol_=mol):
            hcore_total = np.asarray(scf.rhf.get_hcore(mol_), dtype=np.complex128)
            hcore_total += np.einsum("tuv, t -> uv", hcore_1_B, dev_xyz_B)
            return hcore_total

        def get_ovlp(mol_=mol):
            ovlp_total = np.asarray(scf.rhf.get_ovlp(mol_), dtype=np.complex128)
            ovlp_total += np.einsum("tuv, t -> uv", ovlp_1_B, dev_xyz_B)
            return ovlp_total

        mf_s.get_hcore = get_hcore
        mf_s.get_ovlp = get_ovlp
        mf_s._eri = mol.intor("int2e") + np.einsum("tuvkl, t -> uvkl", eri_1_B, dev_xyz_B)
        mf_s.run()

        auxmol = df.make_auxmol(mol, df.aug_etb(mol))
        int3c2e = df.incore.aux_e2(mol, auxmol, "int3c2e")
        int3c2e_ig1 = df.incore.aux_e2(mol, auxmol, "int3c2e_ig1")
        int2c2e = auxmol.intor("int2c2e")
        L = np.linalg.cholesky(int2c2e)
        int3c2e_cd = np.linalg.solve(L, int3c2e.reshape(mol.nao**2, -1).T).reshape(-1, mol.nao, mol.nao)
        int3c2e_ig1_cd = np.linalg.solve(L, int3c2e_ig1.reshape(3 * mol.nao**2, -1).T).reshape(-1, 3, mol.nao, mol.nao)
        int3c2e_2_cd = int3c2e_cd + 2 * lib.einsum("Ptuv, t -> Puv", -1j * int3c2e_ig1_cd, dev_xyz_B)

        mf = dh.energy.UDH(mf_s, xc="MP2")
        mf.df_ri = df.DF(mol, df.aug_etb(mol))
        mf.df_ri._cderi = int3c2e_cd
        mf.df_ri_2 = df.DF(mol, df.aug_etb(mol))
        mf.df_ri_2._cderi = int3c2e_2_cd

        with mf.params.temporary_flags({"integral_scheme": "ri", "incore_t_ijab": True}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.20945698217515063))
