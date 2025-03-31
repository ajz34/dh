# for development, import dh
# for general user, import pyscf.dh

from pyscf import gto, dh
#from dh import DFDH



mol = gto.Mole(atom="N 0. 0. 0.; H .9 0. 0.; H 0. 1. 0.; H 0. 0. 1.1", basis="cc-pVDZ", verbose=4).build()

# default initial guess
mf = dh.DFDH(mol, xc="XYG3")
mf.run()

# use the density matrix from the previous calculation as the initial guess
mf = dh.DFDH(mol, xc="XYG3")
dm = mf.mf_s.init_guess_by_chkfile(chkfile='some.pchk', project=True)
mf.run_scf(dm0=dm)
mf.run()

