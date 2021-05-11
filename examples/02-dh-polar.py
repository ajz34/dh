from pyscf import gto
from dh.polar.rdfdh import Polar
import numpy as np

np.set_printoptions(5, suppress=True, linewidth=180)


if __name__ == '__main__':
    from time import time
    mol = gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", verbose=0).build()
    # XYG3
    # [0.51672 0.49716 0.4677 ]
    # [[ 7.26795 -0.15045 -0.21885]
    #  [-0.15045  8.75195 -0.367  ]
    #  [-0.21885 -0.367   10.47337]]
    mf = Polar(mol, xc="XYG3").run()
    print(mf.dipole())
    print(mf.pol_tot)
    # B2PLYP
    # [0.50747 0.48579 0.45491]
    # [[ 7.38219 -0.15075 -0.21973]
    #  [-0.15075  8.86653 -0.36795]
    #  [-0.21973 -0.36795 10.5606 ]]
    mf = Polar(mol, xc="B2PLYP").run()
    print(mf.dipole())
    print(mf.pol_tot)
    # MP2
    # [0.51397 0.49493 0.46905]
    # [[ 7.28733 -0.14367 -0.2035 ]
    #  [-0.14367  8.72405 -0.33421]
    #  [-0.2035  -0.33421 10.34546]]
    mf = Polar(mol, xc="MP2").run()
    print(mf.dipole())
    print(mf.pol_tot)