import unittest
from pyscf import dh, gto, scf, df
import numpy as np


class TestUIEPA(unittest.TestCase):

    def test_ump2cr_by_restricted(self):
        # Note of this testing
        # This molecule have degenerate orbitals
        # DCPT2 may give different values for this case, even when SCF converges tightly

        # region rmp2cr mole
        mol = gto.Mole()
        mol.atom = """
        H
        F 1 2.25
        """
        mol.basis = {
            "H": gto.basis.parse("""
                H S
                 68.1600   0.00255
                 10.2465   0.01938
                  2.34648  0.09280
                H S
                  0.673320 1.0
                H S
                  0.224660 1.0
                H S
                  0.082217 1.0
                H S
                  0.043    1.0
                H P
                  0.9      1.0
                H P
                  0.3      1.0
                H P
                  0.1      1.0
                H D
                  0.8      1.0  
            """),
            "F": gto.basis.parse("""
                F S
                  23340.    0.000757
                   3431.    0.006081
                    757.7   0.032636
                    209.2   0.131704
                     66.73  0.396240
                F S
                     23.37  1.0
                F S
                      8.624 1.0
                F S
                      2.692 1.0
                F S
                      1.009 1.0
                F S
                      0.3312 1.0
                F P
                     65.66   0.037012
                     15.22   0.243943
                      4.788  0.808302
                F P
                      1.732  1.0
                F P
                      0.6206 1.0
                F P
                      0.2070 1.0
                F S
                      0.1000000              1.0000000        
                F P
                      0.0690000              1.0000000        
                F D
                      1.6400000              1.0000000        
                F D
                      0.5120000              1.0000000        
                F D
                      0.1600000              1.0000000        
                F D
                      0.0500000              1.0000000        
                F F
                      0.5000000              1.0000000
            """),
        }
        mol.build()
        # endregion
        mf_s = scf.UHF(mol)
        # mf_s.conv_tol_grad = 1e-10
        mf_s.run()
        mf = dh.energy.UDH(mf_s, xc="MP2")
        mf.params.flags.update({
            "iepa_scheme": ["mp2cr", "dcpt2", "iepa", "siepa"],
            "integral_scheme": "ri",
        })
        mf.driver_energy_iepa()
        self.assertTrue(np.allclose(mf.params.results["eng_MP2CR"], -0.3362883633558271))
        self.assertTrue(np.allclose(mf.params.results["eng_SIEPA"], -0.3503855844336058))
        # self.assertTrue(np.allclose(mf.params.results["eng_dcpt2"], -0.34933565145777545))
        mf = dh.energy.UDH(mf_s, xc="MP2")
        mf.params.flags.update({
            "iepa_scheme": ["mp2cr", "dcpt2", "iepa", "siepa"],
            "integral_scheme": "conv",
        })
        mf.driver_energy_iepa()
        self.assertTrue(np.allclose(mf.params.results["eng_MP2CR"], -0.3362937710915961))
        self.assertTrue(np.allclose(mf.params.results["eng_SIEPA"], -0.3503916656239522))

    def test_usiepa(self):
        mol = gto.Mole(atom="N; N 1 3", basis="cc-pVTZ", charge=1, spin=1).build()
        mf_s = scf.UHF(mol).run()
        print(mf_s.converged)
        mf = dh.energy.UDH(mf_s, xc="MP2")
        mf.with_df = df.DF(mol, df.aug_etb(mol))
        with mf.params.temporary_flags({"iepa_scheme": ["siepa"]}):
            mf.driver_energy_iepa()
        self.assertTrue(np.allclose(mf.params.results["eng_SIEPA"], -1.354538466289851))
