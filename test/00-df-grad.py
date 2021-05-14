from deriv_numerical import NucCoordDerivGenerator, NumericDiff
from dh import RDFDH
from pyscf import gto, scf
import numpy as np


def mol_to_eng_wrapper(xc):
    def fx(mol):
        mf = RDFDH(mol, xc).run()
        return mf.e_tot
    return fx


if __name__ == '__main__':
    mol = gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", verbose=0).build()
    # MP2                    True 3.4962221522327752e-06
    nde = NumericDiff(NucCoordDerivGenerator(mol, mol_to_eng_wrapper("MP2"))).derivative.reshape(-1, 3)
    de = RDFDH(mol, "MP2").nuc_grad_method().run().grad_tot
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # XYG3                   True 1.2455929719573655e-05
    nde = NumericDiff(NucCoordDerivGenerator(mol, mol_to_eng_wrapper("XYG3"))).derivative.reshape(-1, 3)
    de = RDFDH(mol, "XYG3").nuc_grad_method().run().grad_tot
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # B2PLYP                 False 2.4995459680487997e-05
    nde = NumericDiff(NucCoordDerivGenerator(mol, mol_to_eng_wrapper("B2PLYP"))).derivative.reshape(-1, 3)
    de = RDFDH(mol, "B2PLYP").nuc_grad_method().run().grad_tot
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # XYGJ-OS                True 1.3572484140050856e-05
    nde = NumericDiff(NucCoordDerivGenerator(mol, mol_to_eng_wrapper("XYGJ-OS"))).derivative.reshape(-1, 3)
    de = RDFDH(mol, "XYGJ-OS").nuc_grad_method().run().grad_tot
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # HF reference XYG3      True 1.2058441408377418e-05
    xc = ["HF", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", 0.3211, 1, 1]
    nde = NumericDiff(NucCoordDerivGenerator(mol, mol_to_eng_wrapper(xc))).derivative.reshape(-1, 3)
    de = RDFDH(mol, xc).nuc_grad_method().run().grad_tot
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # B3LYP reference MP2    True 5.036923255707926e-07
    xc = ["B3LYPg", "HF", 1, 1, 1]
    nde = NumericDiff(NucCoordDerivGenerator(mol, mol_to_eng_wrapper(xc))).derivative.reshape(-1, 3)
    de = RDFDH(mol, xc).nuc_grad_method().run().grad_tot
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # HF                     True 2.0872192862952943e-14
    xc = ["HF", "HF", 0, 1, 1]    # should not be used in actual program! dh does not check situation cc == 0
    ade = scf.RHF(mol).density_fit(auxbasis="cc-pVDZ-jkfit").run().nuc_grad_method().run().de
    de = RDFDH(mol, xc).nuc_grad_method().run().grad_tot
    print(np.allclose(ade, de, atol=1e-5, rtol=1e-4), abs(ade - de).max())
