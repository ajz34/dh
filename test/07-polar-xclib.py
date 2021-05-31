from pyscf import gto, dft
from dh import DFDH
from dh.dhutil import TicToc
import numpy as np

np.set_printoptions(5, suppress=True, linewidth=180)

# For PySCF v1.7.6-post, the closed-shell case result should be (because of libxc=4.3.2 in pyscf<=1.7.6)
# [[ 7.27381 -0.15118 -0.2198 ], [-0.15118  8.75867 -0.36819], [-0.2198  -0.36819 10.48013]]
# which is somehow different and in principle wrong
# After PySCF v2.0.0a, libxc is updated to v5, thus results are no longer problematic

if __name__ == '__main__':
    # closed-shell test
    mol = gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", verbose=0).build()
    # XYG3
    # [[ 7.26795 -0.15045 -0.21885]
    #  [-0.15045  8.75195 -0.367  ]
    #  [-0.21885 -0.367   10.47337]]
    tictoc = TicToc(); toc = tictoc.toc

    mf_libxc = DFDH(mol, xc="XYG3").polar_method().run()
    toc("libxc Run")  # 1.7273 sec

    mf_xcfun = DFDH(mol, xc="XYG3").polar_method()
    mf_xcfun.ni.libxc = dft.xcfun
    mf_xcfun.run()
    toc("xcfun Run")  # 2.1685 sec
    # xcfun is usually somehow slower than libxc, no offensive (QλQ)

    print(mf_libxc.pol_tot)
    print((mf_libxc.pol_tot - mf_xcfun.pol_tot) * 1e+10)  # should be close to zero

    # open-shell test
    mol = gto.Mole(atom="C 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", spin=1, verbose=0).build()
    # XYG3
    # [[ 8.49168 -0.09102 -0.14373]
    #  [-0.09102  9.79167 -0.27855]
    #  [-0.14373 -0.27855 11.35302]]
    tictoc = TicToc(); toc = tictoc.toc
    mf_libxc = DFDH(mol, xc="XYG3").polar_method().run()
    toc("libxc Run")  # 5.2012 sec

    mf_xcfun = DFDH(mol, xc="XYG3").polar_method()
    mf_xcfun.ni.libxc = dft.xcfun
    mf_xcfun.run()
    toc("xcfun Run")  # 7.2819 sec

    print(mf_libxc.pol_tot)
    print((mf_libxc.pol_tot - mf_xcfun.pol_tot) * 1e+10)  # should be close to zero
