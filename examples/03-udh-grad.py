from pyscf import gto, dh
#from dh import DFDH
import numpy as np

np.set_printoptions(5, suppress=True, linewidth=180)

if __name__ == '__main__':
    mol = gto.Mole(atom="C 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", spin=1, verbose=0).build()
    # XYG3
    # [[ 0.16885  0.01282 -0.074  ]
    #  [-0.25042  0.04956  0.04518]
    #  [ 0.04461 -0.09688  0.03794]
    #  [ 0.03696  0.03449 -0.00912]]
    print(dh.DFDH(mol, xc="XYG3").run().e_tot)
    mf = dh.DFDH(mol, xc="XYG3").nuc_grad_method().run()
    print(mf.e_tot)
    print(mf.de)
    # B2PLYP
    # [[ 0.17191  0.01703 -0.06912]
    #  [-0.25191  0.04861  0.04429]
    #  [ 0.04375 -0.09947  0.03721]
    #  [ 0.03624  0.03383 -0.01239]]
    print(dh.DFDH(mol, xc="B2PLYP").run().e_tot)
    mf = dh.DFDH(mol, xc="B2PLYP").nuc_grad_method().run()
    print(mf.e_tot)
    print(mf.de)
    # MP2
    # [[ 0.17625  0.01816 -0.07066]
    #  [-0.25667  0.04885  0.04455]
    #  [ 0.04397 -0.10097  0.03734]
    #  [ 0.03645  0.03395 -0.01123]]
    print(dh.DFDH(mol, xc="MP2").run().e_tot)
    mf = dh.DFDH(mol, xc="MP2").nuc_grad_method().run()
    print(mf.e_tot)
    print(mf.de)
