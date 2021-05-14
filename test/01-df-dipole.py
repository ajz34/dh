from deriv_numerical import DipoleDerivGenerator, NumericDiff
from dh import RDFDH
from pyscf import gto, scf
import numpy as np


def mol_to_eng_wrapper(mol, xc):
    def fx(component, interval):
        mf = RDFDH(mol, xc)
        def get_hcore(mol=mol):
            return scf.rhf.get_hcore(mol) - interval * mol.intor("int1e_r")[component]
        mf.mf_s.get_hcore = mf.mf_n.get_hcore = get_hcore
        mf.run()
        return mf.e_tot
    return fx


if __name__ == '__main__':
    mol = gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", verbose=0).build()
    dip_nuc = np.einsum("At, A-> t", mol.atom_coords(), mol.atom_charges())
    # MP2                    True 7.231918783823232e-07
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, "MP2"))).derivative + dip_nuc
    de = RDFDH(mol, "MP2").run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # XYG3                   True 2.1028082815011118e-06
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, "XYG3"))).derivative + dip_nuc
    de = RDFDH(mol, "XYG3").run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # B2PLYP                 True 2.1478260168183994e-06
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, "B2PLYP"))).derivative + dip_nuc
    de = RDFDH(mol, "B2PLYP").run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # XYGJ-OS                True 2.899501268860405e-06
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, "XYGJ-OS"))).derivative + dip_nuc
    de = RDFDH(mol, "XYGJ-OS").run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # HF reference XYG3      True 1.4813120352563658e-07
    xc = ["HF", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", 0.3211, 1, 1]
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, xc))).derivative + dip_nuc
    de = RDFDH(mol, xc).run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # B3LYP reference MP2    True 4.533104975390501e-06
    xc = ["B3LYPg", "HF", 1, 1, 1]
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, xc))).derivative + dip_nuc
    de = RDFDH(mol, xc).run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())
    # HF                     True 7.595880797683918e-07
    xc = ["HF", "HF", 0, 1, 1]    # should not be used in actual program! dh does not check situation cc == 0
    nde = NumericDiff(DipoleDerivGenerator(mol_to_eng_wrapper(mol, xc))).derivative + dip_nuc
    de = RDFDH(mol, xc).run().dipole()
    print(np.allclose(nde, de, atol=1e-5, rtol=1e-4), abs(nde - de).max())

