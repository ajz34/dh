from pyscf.dh import DFDH
from pyscf import gto, dft, mp, df
import numpy as np


def get_energy_xDH(mol, xc_scf, c_pt2, xc_dh=None, basis_jk=None, basis_ri=None):
    basis_jk = basis_jk if basis_jk else df.make_auxbasis(mol, mp2fit=False)
    basis_ri = basis_ri if basis_ri else df.make_auxbasis(mol, mp2fit=True)
    # SCF part calculation
    mf_scf = dft.KS(mol, xc=xc_scf).density_fit(auxbasis=basis_jk).run()
    e_nscf = mf_scf.e_tot
    # Non-consistent part calculation
    if xc_dh is not None:
        mf_nscf = dft.KS(mol, xc=xc_dh).density_fit(auxbasis=basis_jk)
        e_nscf = mf_nscf.energy_tot(dm=mf_scf.make_rdm1())
    # PT2 contribution
    mf_pt2 = mp.MP2(mf_scf)
    mf_pt2.with_df = df.DF(mol, auxbasis=basis_ri)
    mf_pt2.run()
    e_pt2 = c_pt2 * mf_pt2.e_corr
    return e_nscf + e_pt2


if __name__ == '__main__':
    mol = gto.Mole(atom="C 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", spin=1, verbose=0).build()
    # MP2                    True 2.318289517688754e-06
    e1 = get_energy_xDH(mol, "HF", 1)
    e2 = DFDH(mol, "MP2").run().eng_tot
    print(np.allclose(e1, e2, atol=1e-6, rtol=1e-4), abs(e1 - e2))
    # XYG3                   True 1.4033127015977698e-06
    e1 = get_energy_xDH(mol, "B3LYPg", 0.3211, "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP")
    e2 = DFDH(mol, "XYG3").run().eng_tot
    print(np.allclose(e1, e2, atol=1e-6, rtol=1e-4), abs(e1 - e2))
    # B2PLYP                 True 9.31761626077332e-07
    e1 = get_energy_xDH(mol, "0.53*HF + 0.47*B88, 0.73*LYP", 0.27, "0.53*HF + 0.47*B88, 0.73*LYP")
    e2 = DFDH(mol, "B2PLYP").run().eng_tot
    print(np.allclose(e1, e2, atol=1e-6, rtol=1e-4), abs(e1 - e2))
    # HF reference XYG3      True 7.444065914796738e-07
    xc = ["HF", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", 0.3211, 1, 1]
    e1 = get_energy_xDH(mol, xc[0], xc[2], xc[1])
    e2 = DFDH(mol, xc).run().eng_tot
    print(np.allclose(e1, e2, atol=1e-6, rtol=1e-4), abs(e1 - e2))
    # B3LYP reference MP2    True 4.3703185212962126e-06
    xc = ["B3LYPg", "HF", 1, 1, 1]
    e1 = get_energy_xDH(mol, xc[0], xc[2], xc[1])
    e2 = DFDH(mol, xc).run().eng_tot
    print(np.allclose(e1, e2, atol=1e-6, rtol=1e-4), abs(e1 - e2))
    # HF                     True 3.552713678800501e-14
    xc = ["HF", "HF", 0, 1, 1]    # should not be used in actual program! dh does not check situation cc == 0
    e1 = get_energy_xDH(mol, xc[0], xc[2], xc[1])
    e2 = DFDH(mol, xc).run().eng_tot
    print(np.allclose(e1, e2, atol=1e-6, rtol=1e-4), abs(e1 - e2))
