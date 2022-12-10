from pyscf import lib, gto, dft, df
import numpy as np
from pyscf.dh.util import Params, HybridDict, get_default_options
from pyscf.dh.energy.rmp2 import driver_energy_mp2


class RDH(lib.StreamObject):
    """ Restricted doubly hybrid class.
    """
    mf: dft.rks.RKS
    """ Self consistent object. """
    params: Params
    """ Params object consisting of flags, tensors and results. """
    df_ri: df.DF or None
    """ Density fitting object. """
    df_ri_2: df.DF or None
    """ Density fitting object for customize ERI. Used in magnetic computation. """

    @property
    def mol(self) -> gto.Mole: return self.mf.mol

    @property
    def mo_coeff(self) -> np.ndarray: return self.mf.mo_coeff

    @property
    def mo_occ(self) -> np.ndarray: return self.mf.mo_occ

    @property
    def mo_energy(self) -> np.ndarray: return self.mf.mo_energy

    @property
    def nao(self): return self.mol.nao

    @property
    def nocc(self): return self.mol.nelec[0]

    @property
    def nmo(self): return self.mo_coeff.shape[-1]

    @property
    def nvir(self): return self.nmo - self.nocc

    def __init__(self, mf_s=NotImplemented, mf_n=NotImplemented, params=None, df_ri=None):
        self.mf = mf_s
        self.mf_n = mf_n
        self.df_ri = df_ri
        self.df_ri_2 = None
        if params:
            self.params = params
        else:
            self.params = Params(get_default_options(), HybridDict(), {})

    driver_energy_mp2 = driver_energy_mp2
