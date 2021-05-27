# for development, import dh
# for general user, import pyscf.dh

from pyscf import gto
from dh import DFDH


if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = """
    N                  0.00000000    0.00000000    0.10882700
    H                  0.00000000    0.94745500   -0.25393100
    H                  0.82052000   -0.47372700   -0.25393100
    H                 -0.82052000   -0.47372700   -0.25393100
    """
    mol.basis = "6-311+G(3df,2p)"
    mol.verbose = 0
    mol.build()
    bas_jk = "aug-cc-pVTZ-jkfit"
    bas_ri = "aug-cc-pVTZ-ri"
    # xDH4Gau   -56.55573659
    # DFDH      -56.55572801605074
    print(DFDH(mol, xc="XYG3", auxbasis_jk=bas_jk, auxbasis_ri=bas_ri).run().e_tot)
    # xDH4Gau   -56.56076651
    # DFDH      -56.560754653794866
    print(DFDH(mol, xc="revXYG3", auxbasis_jk=bas_jk, auxbasis_ri=bas_ri).run().e_tot)
    # xDH4Gau   -56.46905628
    # DFDH      -56.46903544759781
    print(DFDH(mol, xc="XYG5", auxbasis_jk=bas_jk, auxbasis_ri=bas_ri).run().e_tot)
    # xDH4Gau   -56.45743365
    # DFDH      -56.45741034264096
    print(DFDH(mol, xc="XYG6", auxbasis_jk=bas_jk, auxbasis_ri=bas_ri).run().e_tot)
    # xDH4Gau   -56.26135862
    # DFDH      -56.26134013446673
    print(DFDH(mol, xc="XYG7", auxbasis_jk=bas_jk, auxbasis_ri=bas_ri).run().e_tot)
    # DFDH      -56.438271156661436
    print(DFDH(mol, xc="XYGJ-OS", auxbasis_jk=bas_jk, auxbasis_ri=bas_ri).run().e_tot)
    # DFDH      -56.41584756974222
    print(DFDH(mol, xc="revXYGJ-OS", auxbasis_jk=bas_jk, auxbasis_ri=bas_ri).run().e_tot)
    # DFDH      -56.35174227955246
    print(DFDH(mol, xc="XYGJ-OS5", auxbasis_jk=bas_jk, auxbasis_ri=bas_ri).run().e_tot)
