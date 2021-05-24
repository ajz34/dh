# for development, import dh
# for general user, import pyscf.dh

from pyscf import gto, lib
from dh import DFDH
from pyscf.geomopt.geometric_solver import optimize
import numpy as np

np.set_printoptions(6, suppress=True)


if __name__ == '__main__':

    # Restricted Case
    mol = gto.Mole(atom="O; H 1 1.0; H 1 1.0 2 104.5", basis="6-31G", verbose=0).build()
    mf = DFDH(mol, xc="XYG3").nuc_grad_method()
    mol_eq = optimize(mf)
    print(mol_eq.atom_coords() * lib.param.BOHR)  # print optimized geom in Angstrom
    print(DFDH(mol_eq, xc="XYG3").nuc_grad_method().run().de)  # should be allclose to zero
    # [[ 0.023265 -0.        0.030072]
    #  [ 0.988122 -0.       -0.014877]
    #  [-0.261803 -0.        0.952951]]
    # converge in 3 steps

    # Unrestricted Case
    mol = gto.Mole(atom="O; O 1 1.2", basis="6-31G", spin=2, verbose=0).build()
    mf = DFDH(mol, xc="XYG3").nuc_grad_method()
    mol_eq = optimize(mf)
    print(mol_eq.atom_coords() * lib.param.BOHR)  # print optimized geom in Angstrom
    print(DFDH(mol_eq, xc="XYG3").nuc_grad_method().run().de)  # should be allclose to zero
    # [[-0.056206 -0.       -0.      ]
    #  [ 1.256206  0.        0.      ]]
    # converge in 5 steps
