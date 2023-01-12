import unittest
from pyscf import dh, gto, dft
import numpy as np


class TestRDFT(unittest.TestCase):
    def test_wB97M_V(self):
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = dft.RKS(mol, xc="wB97M_V")
        mf.nlc = "VV10"
        mf.run()
        print(mf.e_tot)

        mf_dh = dh.energy.RDH(mf)
        xc_code = "wB97M_V + VV10(6.0; 0.01)"
        mf_dh.driver_energy_dh(xc_code)
        print(mf_dh.params.results["eng_dh_{:}".format(xc_code)])
        self.assertTrue(np.allclose(mf_dh.params.results["eng_dh_{:}".format(xc_code)], mf.e_tot))
