import unittest
from pyscf import dh, gto, scf, df, lib
import numpy as np


class TestRMP2(unittest.TestCase):
    def test_rmp2_conv(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
        mf_s = scf.RHF(mol).run()

        mf = dh.energy.RDH(mf_s, xc="MP2")
        with mf.params.temporary_flags({"integral_scheme": "conv"}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.273944755130888))

    def test_rmp2_conv_fc(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
        mf_s = scf.RHF(mol).run()

        mf = dh.energy.RDH(mf_s, xc="MP2")
        mf.params.flags["frozen_rule"] = "FreezeNobleGasCore"
        with mf.params.temporary_flags({"integral_scheme": "conv"}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.2602324295391498))

        mf = dh.energy.RDH(mf_s, xc="MP2")
        mf.params.flags["frozen_list"] = [0, 2]
        with mf.params.temporary_flags({"integral_scheme": "conv"}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.13840970089836263))

    def test_rmp2_conv_giao(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()

        hcore_1_B = - 1j * (
            + 0.5 * mol.intor('int1e_giao_irjxp', 3)
            + mol.intor('int1e_ignuc', 3)
            + mol.intor('int1e_igkin', 3))
        ovlp_1_B = - 1j * mol.intor("int1e_igovlp")
        eri_1_B = -1j * (
            + np.einsum("tuvkl -> tuvkl", mol.intor('int2e_ig1'))
            + np.einsum("tkluv -> tuvkl", mol.intor('int2e_ig1')))

        mf_s = scf.RHF(mol)
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

        mf = dh.energy.RDH(mf_s, xc="MP2")
        with mf.params.temporary_flags({"integral_scheme": "conv"}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.27425584824874516))

    def test_rmp2_ri(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
        mf_s = scf.RHF(mol).run()

        mf = dh.energy.RDH(mf_s, xc="MP2")
        mf.df_ri = df.DF(mol, df.aug_etb(mol))
        with mf.params.temporary_flags({"integral_scheme": "ri"}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.27393741308994124))

    def test_rmp2_ri_fc(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
        mf_s = scf.RHF(mol).run()

        mf = dh.energy.RDH(mf_s, xc="MP2")
        mf.df_ri = df.DF(mol, df.aug_etb(mol))
        mf.params.flags["frozen_rule"] = "FreezeNobleGasCore"
        with mf.params.temporary_flags({"integral_scheme": "ri"}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.2602250917785774))

        mf = dh.energy.RDH(mf_s, xc="MP2")
        mf.params.flags["frozen_list"] = [0, 2]
        with mf.params.temporary_flags({"integral_scheme": "ri"}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.13839934020349923))

    def test_rmp2_ri_giao(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()

        hcore_1_B = - 1j * (
            + 0.5 * mol.intor('int1e_giao_irjxp', 3)
            + mol.intor('int1e_ignuc', 3)
            + mol.intor('int1e_igkin', 3))
        ovlp_1_B = - 1j * mol.intor("int1e_igovlp")
        eri_1_B = -1j * (
            + np.einsum("tuvkl -> tuvkl", mol.intor('int2e_ig1'))
            + np.einsum("tkluv -> tuvkl", mol.intor('int2e_ig1')))

        mf_s = scf.RHF(mol)
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

        mf = dh.energy.RDH(mf_s, xc="MP2")
        mf.df_ri = df.DF(mol, df.aug_etb(mol))
        mf.df_ri._cderi = int3c2e_cd
        mf.df_ri_2 = df.DF(mol, df.aug_etb(mol))
        mf.df_ri_2._cderi = int3c2e_2_cd

        with mf.params.temporary_flags({"integral_scheme": "ri", "incore_t_ijab": True}):
            mf.run()
        print(mf.params.results)
        self.assertTrue(np.allclose(mf.params.results["eng_MP2"], -0.27424683619063206))
