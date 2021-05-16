from deriv_numerical import DipoleDerivGenerator, NumericDiff
from dh import DFDH
from pyscf import gto, scf
import numpy as np


def mol_to_eng_wrapper(mol, xc):
    def fx(component, interval):
        mf = DFDH(mol, xc)
        def get_hcore(mol=mol):
            return scf.rhf.get_hcore(mol) - interval * mol.intor("int1e_r")[component]
        mf.mf_s.get_hcore = mf.mf_n.get_hcore = get_hcore
        mf.run()
        return mf.e_tot
    return fx


if __name__ == '__main__':
    mol = gto.Mole(atom="C 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", spin=1, verbose=0).build()
    dip_nuc = np.einsum("At, A-> t", mol.atom_coords(), mol.atom_charges())
    # MP2                    True 9.684032937773424e-07
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, "MP2"))).derivative + dip_nuc
    de = DFDH(mol, "MP2").run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # XYG3                   True 2.194632808816266e-08
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, "XYG3"))).derivative + dip_nuc
    de = DFDH(mol, "XYG3").run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # B2PLYP                 True 8.145511456447707e-07
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, "B2PLYP"))).derivative + dip_nuc
    de = DFDH(mol, "B2PLYP").run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # XYGJ-OS                True 5.5010989985504466e-08
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, "XYGJ-OS"))).derivative + dip_nuc
    de = DFDH(mol, "XYGJ-OS").run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # HF reference XYG3      True 6.136228667408261e-07
    xc = ["HF", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", 0.3211, 1, 1]
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, xc))).derivative + dip_nuc
    de = DFDH(mol, xc).run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # B3LYP reference MP2    True 8.358120329177154e-08
    xc = ["B3LYPg", "HF", 1, 1, 1]
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, xc))).derivative + dip_nuc
    de = DFDH(mol, xc).run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # HF                     True 1.0654892117489823e-06
    xc = ["HF", "HF", 0, 1, 1]    # should not be used in actual program! dh does not check situation cc == 0
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, xc))).derivative + dip_nuc
    de = DFDH(mol, xc).run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
