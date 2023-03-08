import unittest
from pyscf import dh, gto, dft, df, mp
import numpy as np


class TestRDFT(unittest.TestCase):
    def test_wB97M_V(self):
        # SCF with VV10
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = dft.RKS(mol, xc="wB97M_V").density_fit(df.aug_etb(mol))
        mf.nlc = "VV10"
        mf.nlcgrids = dft.Grids(mol)
        mf.nlcgrids.atom_grid = (50, 194)
        mf.nlcgrids.prune = dft.gen_grid.sg1_prune
        mf.nlcgrids.build()
        mf.run()
        self.assertTrue(np.allclose(mf.e_tot, -76.3514767379862, rtol=0, atol=1e-6))
        # SCF without VV10
        mf_nonlc = dft.RKS(mol, xc="wB97M_V").density_fit(df.aug_etb(mol)).run()
        # DH object from SCF with VV10
        # note that VV10 parameters should be explicitly defined in xc_code
        xc_code = "wB97M_V + VV10(6.0; 0.01)"
        mf_dh = dh.energy.RDH(mf, xc_code).run()
        self.assertTrue(np.allclose(mf_dh.e_tot, mf.e_tot, rtol=0, atol=1e-6))
        # DH object from SCF without VV10
        # For this example, VV10 added not by vxc in SCF veff, only merely changes final energy
        mf_dh = dh.energy.RDH(mf_nonlc, xc_code).run()
        self.assertTrue(np.allclose(mf_dh.e_tot, mf.e_tot, rtol=0, atol=1e-6))
        # DH object from molecule and xc_code directly
        mf_dh = dh.energy.RDH(mol, xc_code).run()
        self.assertTrue(np.allclose(mf_dh.e_tot, mf.e_tot, rtol=0, atol=1e-6))

    def test_B2PLYP(self):
        # SCF part of B2PLYP
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = dft.RKS(mol, xc="0.53*HF + 0.47*B88, 0.73*LYP").density_fit(df.aug_etb(mol)).run()
        # Give DH energy by PySCF only
        mf_mp = mp.dfmp2.DFMP2(mf).run()
        print(mf.e_tot + 0.27 * mf_mp.e_corr)
        # DH object from SCF object
        mf_dh = dh.energy.RDH(mf, "B2PLYP").run()
        self.assertTrue(np.allclose(mf_dh.e_tot, -76.290985955007, rtol=0, atol=1e-6))
        # DH object from molecule and xc_code directly
        mf_dh = dh.energy.RDH(mol, "B2PLYP").run()
        self.assertTrue(np.allclose(mf_dh.e_tot, -76.290985955007, rtol=0, atol=1e-6))

    def test_XYG3(self):
        # SCF part of XYG3
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = dft.RKS(mol, xc="B3LYPg").density_fit(df.aug_etb(mol)).run()
        mf_n = dft.RKS(mol, xc="0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP").density_fit(df.aug_etb(mol))
        eng_nc = mf_n.energy_tot(dm=mf.make_rdm1())
        # Give DH energy by PySCF only
        mf_mp = mp.dfmp2.DFMP2(mf).run()
        print(eng_nc + 0.3211 * mf_mp.e_corr)
        # DH object from SCF object
        mf_dh = dh.energy.RDH(mf, "XYG3").run()
        self.assertTrue(np.allclose(mf_dh.e_tot, -76.29073335525743, rtol=0, atol=1e-6))
        # DH object from molecule and xc_code directly
        mf_dh = dh.energy.RDH(mol, "XYG3").run()
        self.assertTrue(np.allclose(mf_dh.e_tot, -76.29073335525743, rtol=0, atol=1e-6))