from pyscf import lib, gto, dft, df
import numpy as np
from pyscf.dh.util import Params, HybridDict, get_default_options
from pyscf.dh.energy.rmp2 import driver_energy_rmp2


class RDH(lib.StreamObject):
    """ Restricted doubly hybrid class. """
    mf: dft.rks.RKS
    """ Self consistent object. """
    params: Params
    """ Params object consisting of flags, tensors and results. """
    df_ri: df.DF or None
    """ Density fitting object. """
    df_ri_2: df.DF or None
    """ Density fitting object for customize ERI. Used in magnetic computation. """

    @property
    def mol(self) -> gto.Mole:
        """ Molecular object. """
        return self.mf.mol

    @property
    def mo_coeff(self) -> np.ndarray:
        """ Molecular orbital coefficient. """
        return self.mf.mo_coeff

    @property
    def mo_occ(self) -> np.ndarray:
        """ Molecular orbital occupation number. """
        return self.mf.mo_occ

    @property
    def mo_energy(self) -> np.ndarray:
        """ Molecular orbital energy. """
        return self.mf.mo_energy

    @property
    def nao(self) -> int:
        """ Number of atomic orbitals. """
        return self.mol.nao

    @property
    def nocc(self) -> int:
        """ Number of occupied molecular orbitals. """
        return self.mol.nelec[0]

    @property
    def nmo(self) -> int:
        """ Number of molecular orbitals. """
        return self.mo_coeff.shape[-1]

    @property
    def nvir(self) -> int:
        """ Number of virtual molecular orbitals. """
        return self.nmo - self.nocc

    @property
    def restricted(self) -> bool:
        """ Restricted closed-shell or unrestricted open-shell. """
        return True

    def __init__(self, mf=NotImplemented, params=None, df_ri=None):
        self.mf = mf
        self.df_ri = df_ri
        self.df_ri_2 = None
        if params:
            self.params = params
        else:
            self.params = Params(get_default_options(), HybridDict(), {})
        self.verbose = self.mol.verbose

    driver_energy_mp2 = driver_energy_rmp2
