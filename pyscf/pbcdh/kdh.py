from pyscf import dh
# typing import
from typing import Tuple, TYPE_CHECKING
from pyscf import lib
from pyscf.pbc import gto, scf, df, mp
import numpy as np

class KDH(dh.RDFDH):

    def __init__(self,
                 mol: gto.Cell,
                 xc: str or tuple = "XYG3",
                 #auxbasis_jk: str or dict or None = None,
                 #auxbasis_ri: str or dict or None = None,
                 #grids: dft.Grids = None,
                 #grids_cpks: dft.Grids = None,
                 unrestricted: bool = False,  # only for class initialization
                 kpts = np.zeros((1,3))
                 ):
        self.max_memory = mol.max_memory
        # Parse xc code
        # It's tricky to say that, self.xc refers to SCF xc, and self.xc_dh refers to double hybrid xc
        # There should be three kinds of possible inputs:
        # 1) String: "XYG3"
        # 2) Tuple: ("B3LYPg", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", 0.3211, 1, 1)
        # 3) Additional: (("0.69*HF + 0.31*PBE, 0.44*P86", None, 1, 0.52, 0.22), {"D3": ([0.48, 0, 0, 5.6, 0], 4)})
        self.xc_dh = xc
        if isinstance(xc, str):
            xc_list, xc_add = parse_xc_dh(xc)
        elif len(xc) == 5:  # here should assert xc is a tuple/list with 2 or 5 elements
            xc_list = xc
            xc_add = {}
        else:  # assert len(xc) == 2
            xc_list, xc_add = xc
        self.xc, self.xc_n, self.cc, self.c_os, self.c_ss = xc_list
        self.xc_add = xc_add
        # parse auxiliary basis
        #self.auxbasis_jk = auxbasis_jk = auxbasis_jk if auxbasis_jk else df.make_auxbasis(mol, mp2fit=False)
        #self.auxbasis_ri = auxbasis_ri = auxbasis_ri if auxbasis_ri else df.make_auxbasis(mol, mp2fit=True)
        #self.same_aux = True if auxbasis_jk == auxbasis_ri or auxbasis_ri is None else False
        # parse scf method
        self.unrestricted = unrestricted
        if unrestricted:
            mf_s = dft.KUKS(mol, kpts, xc=self.xc_n).rs_density_fit()
        else:
            mf_s = dft.KRKS(mol, kpts, xc=self.xc_n).rs_density_fit()
        #self.grids = grids if grids else mf_s.grids                        # type: dft.grid.Grids
        #self.grids_cpks = grids_cpks if grids_cpks else self.grids         # type: dft.grid.Grids
        self.mf_s = mf_s                                                   # type: dft.rks.RKS
        #self.mf_s.grids = self.grids
