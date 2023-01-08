from pyscf.dh.energy.rdh import RDH
from pyscf.dh.energy.ump2 import driver_energy_ump2
from pyscf.dh.energy.uiepa import driver_energy_uiepa
from typing import Tuple


class UDH(RDH):
    """ Unrestricted doubly hybrid class. """

    @property
    def restricted(self) -> bool:
        return False

    @property
    def nocc(self) -> Tuple[int, int]:
        """ Number of occupied (alpha, beta) molecular orbitals. """
        return self.mol.nelec

    @property
    def nvir(self) -> Tuple[int, int]:
        """ Number of virtual (alpha, beta) molecular orbitals. """
        return self.nmo - self.nocc[0], self.nmo - self.nocc[1]

    driver_energy_mp2 = driver_energy_ump2
    driver_energy_iepa = driver_energy_uiepa
