import unittest
from pyscf import gto, dh, dft, df


class TestRDFTXC(unittest.TestCase):
    def test_various_xc_combinations(self):
        # make sure some complicated xc code is able to be computed
        # note that the following code does not make sure result is correct
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        xc_scf_token = "B3LYPg"
        xc_eng_token = \
            "0.25*HF + 0.65*LR_HF(0.7) + 0.35*SR_HF(0.5) + 0.4*B3LYPg + 0.2*PBE0 - 0.5*B88," \
            "0.6*LYP + 0.25*SSR(GGA_C_P86, 0.65)" \
            "+ 0.2*MP2 + 0.7*SR_MP2(1.5) + 2.8*LR_MP2(0.6, 1.5, 0.75) + 1.25*SIEPA(1.6, 0.1)"
        mf_dh = dh.RDH(mol, xc=(xc_scf_token, xc_eng_token)).run()
        print(mf_dh.e_tot)

    def test_complicated_xc_scf(self):
        # make sure energy functional gives e_tot of pyscf-evaluatable xc
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        xc_scf_token = "0.25*HF + 0.65*LR_HF(0.7) + 0.27*SR_HF(0.7) + 0.4*B3LYPg + 0.2*PBE0 - 0.5*B88, 0.6*LYP"
        mf_dh = dh.RDH(mol, xc=xc_scf_token)
        mf_dh.params.flags["debug_force_eng_low_rung_revaluate"] = True
        mf_dh.run()
        mf_scf = dft.RKS(mol, xc=xc_scf_token).density_fit(df.aug_etb(mol)).run()
        self.assertAlmostEqual(mf_dh.scf.e_tot, mf_dh.e_tot, 8)
        self.assertAlmostEqual(mf_dh.scf.e_tot, mf_scf.e_tot, 8)
