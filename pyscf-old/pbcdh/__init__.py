from . import kdh
from pyscf.pbc import gto


def KDH(mol: gto.Cell, *args, **kwargs):
    if mol.spin != 0:
        return kdh.KDH(mol, *args, **kwargs)
    else:
        return kdh.KDH(mol, *args, **kwargs)


