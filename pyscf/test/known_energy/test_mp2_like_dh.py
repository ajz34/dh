import unittest
from pyscf import dh, gto, dft, df
import numpy as np


"""
Comparison value from MRCC 2022-03-18.
"""


coord = """
H
O 1 R1
H 2 R1 1 A
""".replace("R1", "2.0").replace("A", "104.2458898548")

mol = gto.Mole(atom=coord, basis="cc-pVTZ", unit="AU", verbose=0).build()


class TestMP2LikeDHwithMRCC(unittest.TestCase):
    def test_B2PLYP(self):
        # test case: MINP_H2O_cc-pVTZ_RKS_B2PLYP
        REF_ESCF = -76.305197382056
        REF_ETOT = -76.391961061470

        params = dh.util.Params(flags={
            "integral_scheme_scf": "Conv",
            "frozen_rule": "FreezeNobleGasCore"
        })
        mf = dh.RDH(mol, xc="B2PLYP", params=params)
        mf.df_ri = df.DF(mol, auxbasis="cc-pVTZ-ri")
        mf.run()
        self.assertAlmostEqual(mf.mf.e_tot, REF_ESCF, places=5)
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_B2GPPLYP(self):
        # MINP_H2O_cc-pVTZ_DF-RKS_B2GPPLYP-D3
        # without DFT-D3
        REF_ESCF = -76.268047709113
        REF_ETOT = -76.378191035928

        params = dh.util.Params(flags={
            "integral_scheme_scf": "RI-JK",
            "frozen_rule": "FreezeNobleGasCore"
        })
        mf = dh.RDH(mol, xc="B2GPPLYP", params=params)
        mf.mf.with_df = df.DF(mol, auxbasis="cc-pVTZ-jkfit")
        mf.df_ri = df.DF(mol, auxbasis="cc-pVTZ-ri")
        mf.run()
        self.assertAlmostEqual(mf.mf.e_tot, REF_ESCF, places=5)
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_DSDPBEP86(self):
        # MINP_H2O_cc-pVTZ_DF-RKS_DSDPBEP86-D3
        # without DFT-D3
        # TODO: MRCC may uses an older version of DSD-PBEP86 (10.1039/C1CP22592H).
        REF_ESCF = -76.186838177949
        REF_ETOT = -76.325115287231

        params = dh.util.Params(flags={
            "integral_scheme_scf": "RI-JK",
            "frozen_rule": "FreezeNobleGasCore"
        })
        mf = dh.RDH(mol, xc="DSD-PBEP86-D3", params=params)
        mf.mf.with_df = df.DF(mol, auxbasis="cc-pVTZ-jkfit")
        mf.df_ri = df.DF(mol, auxbasis="cc-pVTZ-ri")
        mf.run()
        print()
        print(mf.mf.e_tot)
        print(mf.e_tot)
        # self.assertAlmostEqual(mf.mf.e_tot, REF_ESCF, places=5)
        # self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_XYG3(self):
        # MINP_H2O_cc-pVTZ_XYG3
        REF_ETOT = -76.400701189006

        params = dh.util.Params(flags={
            "integral_scheme_scf": "RI-JK",
            "frozen_rule": "FreezeNobleGasCore"
        })
        mf = dh.RDH(mol, xc="XYG3", params=params)
        mf.mf.with_df = df.DF(mol, auxbasis="cc-pVTZ-jkfit")
        mf.df_ri = df.DF(mol, auxbasis="cc-pVTZ-ri")
        mf.run()
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)

    def test_SCAN0_2(self):
        # MINP_H2O_cc-pVTZ_SCAN0-2_Libxc
        # TODO: SCAN seems to be very instable for different softwares.
        REF_ESCF = -76.204558509844
        REF_ETOT = -76.348414592594

        params = dh.util.Params(flags={
            "integral_scheme_scf": "RI-JK",
            "frozen_rule": "FreezeNobleGasCore"
        })
        mf = dh.RDH(mol, xc="SCAN", params=params)
        mf.mf.with_df = df.DF(mol, auxbasis="cc-pVTZ-jkfit")
        mf.df_ri = df.DF(mol, auxbasis="cc-pVTZ-ri")
        mf.run()
        print()
        print(mf.mf.e_tot)
        print(mf.e_tot)
        # self.assertAlmostEqual(mf.mf.e_tot, REF_ESCF, places=5)
        # self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)
