import unittest
from pyscf import dh, gto


"""
Comparison value from
- MRCC 2022-03-18.
"""


class TestRSDH(unittest.TestCase):
    def test_RS_PBE_P86(self):
        # reference: MRCC
        # test case: MINP_H2O_cc-pVTZ_RKS_B2PLYP
        REF_ESCF = -76.219885498301
        REF_ETOT = -76.315858865489

        mol = gto.Mole(atom="""
        O     0.00000000    0.00000000   -0.12502304
        H     0.00000000    1.43266384    0.99210317
        H     0.00000000   -1.43266384    0.99210317
        """, basis="aug-cc-pVDZ", unit="AU").build()
        params = dh.util.Params(flags={
            "frozen_rule": "FreezeNobleGasCore",
            "auxbasis_jk": "aug-cc-pVDZ-jkfit",
            "auxbasis_ri": "aug-cc-pVDZ-ri",
        })
        mf = dh.RDH(mol, xc="RS-PBE-P86", params=params)
        mf.run()
        self.assertAlmostEqual(mf._scf.e_tot, REF_ESCF, places=5)
        self.assertAlmostEqual(mf.e_tot, REF_ETOT, places=5)