# for development, import dh
# for general user, import pyscf.dh

from pyscf import gto, dh
#from dh import DFDH
import numpy as np

np.set_printoptions(5, suppress=True, linewidth=180)


if __name__ == '__main__':
    mol = gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", verbose=0).build()
    # XYG3
    # [0.51672 0.49716 0.4677 ]
    # [[ 7.26795 -0.15045 -0.21885]
    #  [-0.15045  8.75195 -0.367  ]
    #  [-0.21885 -0.367   10.47337]]
    mf = dh.DFDH(mol, xc="XYG3").polar_method().run()
    print(mf.dipole())
    print(mf.pol_tot)
    # B2PLYP
    # [0.50747 0.48579 0.45491]
    # [[ 7.38219 -0.15075 -0.21973]
    #  [-0.15075  8.86653 -0.36795]
    #  [-0.21973 -0.36795 10.5606 ]]
    mf = dh.DFDH(mol, xc="B2PLYP").polar_method().run()
    print(mf.dipole())
    print(mf.pol_tot)
    # MP2
    # [0.51397 0.49493 0.46905]
    # [[ 7.28733 -0.14367 -0.2035 ]
    #  [-0.14367  8.72405 -0.33421]
    #  [-0.2035  -0.33421 10.34546]]
    mf = dh.DFDH(mol, xc="MP2").polar_method().run()
    print(mf.dipole())
    print(mf.pol_tot)
