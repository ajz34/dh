from pyscf.dh import DFDH
from pyscf import gto
import numpy as np



if __name__ == '__main__':
    mol = gto.Mole(atom="C 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", spin=1, verbose=0).build()

    # test for restart
    # XYG3                   True    < 1e-9
    #e1 = get_energy_xDH(mol, "B3LYPg", 0.3211, "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP")
    mf = DFDH(mol, "XYG3").run()
    e2 = mf.eng_tot
    mf2 = DFDH(mol, "XYG3")
    dm = mf.mf_s.make_rdm1()
    mf2.run_scf(dm0=dm)
    mf2.run()
    e3 = mf2.eng_tot

    print(np.allclose(e2, e3, atol=1e-6, rtol=1e-4), abs(e2 - e3))
