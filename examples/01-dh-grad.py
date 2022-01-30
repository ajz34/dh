# for development, import dh
# for general user, import pyscf.dh

from pyscf import gto, dh
#from dh import DFDH
import numpy as np

np.set_printoptions(5, suppress=True, linewidth=180)

if __name__ == '__main__':
    mol = gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", verbose=0).build()
    # XYG3
    # [[ 0.12606 -0.00929 -0.07822]
    #  [-0.16752  0.02649  0.02154]
    #  [ 0.02384 -0.03202  0.01631]
    #  [ 0.01762  0.01483  0.04038]]
    print(dh.DFDH(mol, xc="XYG3").run().e_tot)
    mf = dh.DFDH(mol, xc="XYG3").nuc_grad_method().run()
    print(mf.e_tot)
    print(mf.de)
    # B2PLYP
    # [[ 0.13023 -0.00424 -0.07277]
    #  [-0.17088  0.02598  0.0211 ]
    #  [ 0.02338 -0.03631  0.01602]
    #  [ 0.01726  0.01457  0.03565]]
    print(dh.DFDH(mol, xc="B2PLYP").run().e_tot)
    mf = dh.DFDH(mol, xc="B2PLYP").nuc_grad_method().run()
    print(mf.e_tot)
    print(mf.de)
    # MP2
    # [[ 0.1345  -0.00321 -0.07428]
    #  [-0.17439  0.02557  0.02063]
    #  [ 0.02301 -0.03636  0.01541]
    #  [ 0.01688  0.01401  0.03824]]
    print(dh.DFDH(mol, xc="MP2").run().e_tot)
    mf = dh.DFDH(mol, xc="MP2").nuc_grad_method().run()
    print(mf.e_tot)
    print(mf.de)
