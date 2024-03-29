from deriv_numerical import DipoleDerivGenerator, NumericDiff
from pyscf.dh import DFDH
from pyscf import gto, scf
import numpy as np


def mol_to_eng_wrapper(mol, xc):
    def fx(component, interval):
        mf = DFDH(mol, xc)
        def get_hcore(mol=mol):
            return scf.rhf.get_hcore(mol) - interval * mol.intor("int1e_r")[component]
        mf.mf_s.get_hcore = mf.mf_n.get_hcore = get_hcore
        return mf.run().dipole()
    return fx


if __name__ == '__main__':
    mol = gto.Mole(atom="C 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", spin=1, verbose=0).build()
    # MP2                    True 1.662427478521522e-05
    nde = - NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, "MP2"))).derivative
    de = DFDH(mol, "MP2").polar_method().run().pol_tot
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # XYG3                   True 1.8334131063890702e-06
    nde = - NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, "XYG3"))).derivative
    de = DFDH(mol, "XYG3").polar_method().run().pol_tot
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # B2PLYP                 True 1.9267892900742822e-05
    nde = - NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, "B2PLYP"))).derivative
    de = DFDH(mol, "B2PLYP").polar_method().run().pol_tot
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # XYGJ-OS                True 1.8346018799131336e-06
    nde = - NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, "XYGJ-OS"))).derivative
    de = DFDH(mol, "XYGJ-OS").polar_method().run().pol_tot
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # HF reference XYG3      True 4.658442302352128e-06
    xc = ["HF", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", 0.3211, 1, 1]
    nde = - NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, xc))).derivative
    de = DFDH(mol, xc).polar_method().run().pol_tot
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # B3LYP reference MP2    True 2.414159641361824e-06
    xc = ["B3LYPg", "HF", 1, 1, 1]
    nde = - NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, xc))).derivative
    de = DFDH(mol, xc).polar_method().run().pol_tot
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # HF                     True 1.5621570573998156e-05
    # xc = ["HF", "HF", 0, 1, 1]    # should not be used in actual program! dh does not check situation cc == 0
    # nde = - NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, xc))).derivative
    # de = DFDH(mol, xc).polar_method().run().pol_tot
    # print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())

